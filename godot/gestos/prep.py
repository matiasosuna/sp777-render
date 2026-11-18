# Capas del encapuchado por partes, para animarlo con gestos en Godot (fase 2, 17/11/2026).
# Corre en GitHub Actions (sp777-render, godot.yml con carpeta=godot/gestos), nunca en la Mac.
# Entra fuente.png (imagen_0 de FLUX), cortes.json (polígonos a mano, en píxeles de 1080x1920) y piezas/*.png (FLUX).
# Sale en proj/capas/:
#   cuerpo.png     el personaje sin los brazos; lo que tapaban las mangas se rellena con LaMa y se cierra con tinta
#   brazoL/R.png   cada brazo con su manga campana y su mano (L = el de la izquierda de la imagen)
#   manto.png      la esclavina (copia de esa zona, va delante de los hombros de los brazos)
#   noche.jpg, anillo.png, anillo_glow.png   fondo de noche sin personaje y el círculo de runas completo
#   <pieza>.png/.jpg   cada pieza de piezas/ (recortada o como fondo, según cortes.json)
#   layout.json
# En out/prep/ quedan grillas con coordenadas (para trazar los cortes) y controles.
import json, os, sys, time, urllib.request, glob
import numpy as np
import cv2
from PIL import Image

W, H = 1080, 1920
D = sys.argv[1]
args = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
CAPAS = os.path.join(D, 'proj', 'capas')
CTL = 'out/prep'
os.makedirs(CAPAS, exist_ok=True)
os.makedirs(CTL, exist_ok=True)
open(os.path.join(CAPAS, '.gdignore'), 'w').close()
CORTES = json.load(open(os.path.join(D, 'cortes.json'))) if os.path.exists(os.path.join(D, 'cortes.json')) else {}
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


def bajar(url, path):
    if not os.path.exists(path):
        log('bajando', url)
        urllib.request.urlretrieve(url, path)
    return path


import torch
torch.set_num_threads(os.cpu_count() or 4)
_esr = None


def esr(src):
    """Real-ESRGAN anime x4 (PIL RGB -> PIL RGB x4); si falla, Lanczos."""
    global _esr
    try:
        if _esr is None:
            from spandrel import ModelLoader
            mp = bajar('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', '/tmp/esr.pth')
            _esr = ModelLoader().load_from_file(mp).eval()
        sw, sh = src.size
        x = torch.from_numpy(np.asarray(src, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        T, O, S = 192, 16, 4
        out = torch.zeros(1, 3, sh * S, sw * S)
        with torch.no_grad():
            for ty in range(0, sh, T):
                for tx in range(0, sw, T):
                    ya, yb = max(ty - O, 0), min(ty + T + O, sh)
                    xa, xb = max(tx - O, 0), min(tx + T + O, sw)
                    y = _esr(x[:, :, ya:yb, xa:xb])
                    cy0, cy1 = (ty - ya) * S, (min(ty + T, sh) - ya) * S
                    cx0, cx1 = (tx - xa) * S, (min(tx + T, sw) - xa) * S
                    out[:, :, ty * S:min(ty + T, sh) * S, tx * S:min(tx + T, sw) * S] = y[:, :, cy0:cy1, cx0:cx1]
        return Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8))
    except Exception as e:
        log('ESR falló, Lanczos:', repr(e))
        return src.resize((src.size[0] * 4, src.size[1] * 4), Image.LANCZOS)


def cubrir(img, w, h):
    """escala para cubrir w x h y recorta al centro"""
    sw, sh = img.size
    k = max(w / sw, h / sh)
    nw, nh = round(sw * k), round(sh * k)
    img = img.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return img.crop((x0, y0, x0 + w, y0 + h))


from rembg import new_session, remove
_ses = {}


def alfa(img, modelo='birefnet-general'):
    if modelo not in _ses:
        _ses[modelo] = new_session(modelo)
    m = remove(img, session=_ses[modelo], only_mask=True)
    a = np.asarray(m.convert('L'), dtype=np.float32) / 255
    # componente principal y sin agujeros chicos
    bin_ = (a > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bin_, 8)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        a = a * cv2.dilate((lab == k).astype(np.uint8), np.ones((9, 9), np.uint8))
    inv = (a <= 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    hh, ww = a.shape
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 1500 and x > 0 and y > 0 and x + w < ww and y + h < hh:
            a[lab == i] = 1.0
    return a


_lama = None


def lama(img_np, mask):
    global _lama
    if _lama is None:
        lp = bajar('https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt', '/tmp/big-lama.pt')
        _lama = torch.jit.load(lp, map_location='cpu').eval()
    h, w = mask.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    im = np.pad(img_np, ((0, ph), (0, pw), (0, 0)), mode='edge')
    mk = np.pad(mask, ((0, ph), (0, pw)))
    with torch.no_grad():
        r = _lama(torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1)[None],
                  torch.from_numpy((mk > 0).astype(np.float32))[None, None])
    out = (r[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)[:h, :w]
    return np.where(mask[..., None] > 0, out, img_np)


def poli(pts, shape, pluma=1.2):
    """polígono (lista de [x, y]) o lista de polígonos, rellenos y con borde suave"""
    m = np.zeros(shape, np.uint8)
    if pts:
        varios = pts if isinstance(pts[0][0], (list, tuple)) else [pts]
        cv2.fillPoly(m, [np.array(p_, np.int32).reshape(-1, 1, 2) for p_ in varios], 255, cv2.LINE_AA)
    m = m.astype(np.float32) / 255
    if pluma > 0:
        m = cv2.GaussianBlur(m, (0, 0), pluma)
    return m


def grilla(img_np, nombre, paso=50, tile=(540, 640)):
    g = img_np.copy()
    if g.shape[2] == 4:
        a = g[..., 3:4].astype(np.float32) / 255
        g = (g[..., :3] * a + np.array([255, 0, 200]) * (1 - a)).astype(np.uint8)
    g = np.ascontiguousarray(g)
    h, w = g.shape[:2]
    for x in range(0, w, paso):
        cv2.line(g, (x, 0), (x, h - 1), (255, 255, 255) if x % 100 == 0 else (120, 120, 120), 1)
    for y in range(0, h, paso):
        cv2.line(g, (0, y), (w - 1, y), (255, 255, 255) if y % 100 == 0 else (120, 120, 120), 1)
    tw, th = tile
    for ty in range(0, h, th):
        for tx in range(0, w, tw):
            c = g[ty:ty + th, tx:tx + tw].copy()
            for x in range((tx // 100) * 100, tx + tw, 100):
                if x >= tx:
                    cv2.putText(c, str(x), (x - tx + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
            for y in range((ty // 100) * 100, ty + th, 100):
                if y >= ty:
                    cv2.putText(c, str(y), (2, y - ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            Image.fromarray(c).save(f'{CTL}/grilla_{nombre}_{ty:04d}_{tx:04d}.jpg', quality=88)


def correr_color(col, a, it=40):
    """lleva el color de adentro hacia la zona transparente (sin halos al filtrar la textura)"""
    val = (a > 0.5).astype(np.float32)
    cc = col * val[..., None]
    v = val.copy()
    for _ in range(it):
        s_ = cv2.blur(cc, (3, 3))
        n_ = cv2.blur(v, (3, 3))
        nuevo = (v == 0) & (n_ > 1e-4)
        if not nuevo.any():
            break
        cc[nuevo] = s_[nuevo] / n_[nuevo][:, None]
        v[nuevo] = 1
    return np.where(val[..., None] > 0, col, cc)


def guardar_rgba(col, a, path):
    col = correr_color(col.astype(np.float32), a)
    Image.fromarray(np.dstack([np.clip(col, 0, 255).astype(np.uint8), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)]), 'RGBA').save(path, optimize=True)


def descontaminar(col, a, bg):
    af = a[..., None]
    borde = (af > 0.03) & (af < 0.97)
    return np.where(borde, np.clip((col - (1 - af) * bg) / np.maximum(af, 0.03), 0, 255), col)


# ================= 1. personaje base =================
src = Image.open(os.path.join(D, 'fuente.png')).convert('RGB')
base = cubrir(esr(src), W, H)
base_np = np.asarray(base).copy()
a = alfa(base)
corte_y = float(CORTES.get('corte_y', 1828))
a = a * np.clip((corte_y - np.arange(H, dtype=np.float32)) / 14, 0, 1)[:, None]
fig = (a > 0.02).astype(np.uint8)
grilla(base_np, 'base')
log('base y recorte')

# ================= 2. fondo de noche y anillo (como la fase 1) =================
hueco = cv2.dilate(fig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
bi = base_np.astype(np.int32)
amar = (bi[..., 0] + bi[..., 1]) / 2 - bi[..., 2]
oro = (amar > 45) & (np.maximum(bi[..., 0], bi[..., 1]) > 140) & (hueco == 0)
ys, xs = np.nonzero(oro)


def ajusta(xs, ys):
    A_ = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    c, *_ = np.linalg.lstsq(A_, xs ** 2 + ys ** 2, rcond=None)
    return c[0], c[1], np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)


cx, cy, rr = ajusta(xs.astype(float), ys.astype(float))
for _ in range(3):
    d = np.hypot(xs - cx, ys - cy)
    ok = np.abs(d - rr) < 90
    cx, cy, rr = ajusta(xs[ok].astype(float), ys[ok].astype(float))
d = np.hypot(xs - cx, ys - cy)
ok = np.abs(d - rr) < 110
r_in, r_out = float(np.percentile(d[ok], 1)), float(np.percentile(d[ok], 99))
yy, xx = np.mgrid[0:H, 0:W]
dd = np.hypot(xx - cx, yy - cy)
MARG = 34
banda = ((dd > r_in - MARG) & (dd < r_out + MARG)).astype(np.uint8)
noche = lama(base_np, np.maximum(hueco, banda))
Image.fromarray(noche).save(f'{CAPAS}/noche.jpg', quality=94)
luz = np.clip(base_np.astype(np.float32) - noche.astype(np.float32), 0, 255) * (banda * (1 - hueco))[..., None]
K = 8
maxR = int(np.ceil(r_out + MARG + 6))
A = K * int(round(2 * np.pi * maxR / K))
FL = cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR
pol = cv2.warpPolar(luz, (maxR, A), (cx, cy), maxR, FL)
occ = cv2.warpPolar(hueco.astype(np.float32), (maxR, A), (cx, cy), maxR, FL)
dentro = cv2.warpPolar(np.ones((H, W), np.float32), (maxR, A), (cx, cy), maxR, FL)
ra, rb = int(max(r_in - MARG, 0)), min(int(r_out + MARG), maxR)
malo = (occ[:, ra:rb].max(axis=1) > 0.01) | (dentro[:, ra:rb].min(axis=1) < 0.99)
malo = np.convolve(np.r_[malo[-8:], malo, malo[:8]].astype(float), np.ones(17), 'same')[8:-8] > 0
P = A // K
F = max(P // 5, 8)
bueno = ~malo
mejor, ini, cur, cs = 0, 0, 0, 0
for i, v in enumerate(np.r_[bueno, bueno]):
    if v:
        if cur == 0:
            cs = i
        cur += 1
        if cur > mejor and cur <= A:
            mejor, ini = cur, cs
    else:
        cur = 0
if mejor < P + F:
    F = max(mejor - P, 0)
s0 = ini + max((mejor - (P + F)) // 2, 0)
seg = pol[(s0 + np.arange(P + F)) % A]
tile = seg[:P].copy()
for i in range(F):
    tile[i] = seg[i] * (i / F) + seg[P + i] * (1 - i / F)
S2 = 2 * maxR
anillo = cv2.warpPolar(np.tile(tile, (K, 1, 1)), (S2, S2), (maxR, maxR), maxR, FL | cv2.WARP_INVERSE_MAP)
ry, rx = np.mgrid[0:S2, 0:S2]
rd = np.hypot(rx - maxR + 0.5, ry - maxR + 0.5)
anillo *= np.clip((r_out + MARG - rd) / 6, 0, 1)[..., None] * np.clip((rd - (r_in - MARG)) / 6, 0, 1)[..., None]
anillo = np.clip(anillo, 0, 255)
Image.fromarray(anillo.astype(np.uint8)).save(f'{CAPAS}/anillo.png')
Image.fromarray(np.clip(cv2.GaussianBlur(anillo, (0, 0), 9) * 1.8, 0, 255).astype(np.uint8)).save(f'{CAPAS}/anillo_glow.png')
log('noche y anillo')

# ================= 3. capas del personaje =================
col = descontaminar(base_np.astype(np.float32), a, noche.astype(np.float32))
capas_info = {}
mL = poli(CORTES.get('brazoL'), (H, W))
mR = poli(CORTES.get('brazoR'), (H, W))
casco = np.maximum(poli(CORTES.get('cascoL'), (H, W)), poli(CORTES.get('cascoR'), (H, W)))
manto = poli(CORTES.get('manto'), (H, W))
brazos = np.maximum(mL, mR) * (a > 0.02)
oculto = cv2.dilate((brazos > 0.3).astype(np.uint8), np.ones((7, 7), np.uint8))
# LaMa con el afuera del personaje pintado del color del abrigo: rellena tela, no cielo
zona = (fig > 0) & (oculto == 0) & (casco > 0.5)
tela = np.median(base_np[zona], axis=0) if zona.sum() > 100 else np.array([30, 34, 60])
log('color de tela', tela)
para_lama = np.where(fig[..., None] > 0, base_np, tela.astype(np.uint8))
relleno = lama(para_lama.astype(np.uint8), oculto)
cuerpo_col = np.where(oculto[..., None] > 0, relleno.astype(np.float32), col)
# alfa del cuerpo: donde estaban los brazos, solo lo que queda dentro del casco (silueta del abrigo)
cuerpo_a = np.where(oculto > 0, casco * (fig > 0), a)
# tinta en el borde nuevo del casco (solo dentro de lo que tapaban los brazos)
cb = (casco > 0.5).astype(np.uint8)
borde = (cb - cv2.erode(cb, np.ones((5, 5), np.uint8))) * oculto
tinta = cv2.GaussianBlur(borde.astype(np.float32), (0, 0), 0.8)
cuerpo_col = cuerpo_col * (1 - tinta[..., None]) + np.array([16, 14, 26], np.float32) * tinta[..., None]
guardar_rgba(cuerpo_col, cuerpo_a, f'{CAPAS}/cuerpo.png')
hL = poli(CORTES.get('manoL'), (H, W), 0.8)
hR = poli(CORTES.get('manoR'), (H, W), 0.8)
guardar_rgba(col, a * mL * (1 - hL), f'{CAPAS}/brazoL.png')
guardar_rgba(col, a * mR * (1 - hR), f'{CAPAS}/brazoR.png')
guardar_rgba(col, a * hL, f'{CAPAS}/manoL.png')
guardar_rgba(col, a * hR, f'{CAPAS}/manoR.png')
if CORTES.get('cintura'):
    guardar_rgba(cuerpo_col, cuerpo_a * poli(CORTES['cintura'], (H, W), 6.0), f'{CAPAS}/cuerpo_sentado.png')
guardar_rgba(col, a * manto, f'{CAPAS}/manto.png')
mag = np.array([255, 0, 200], np.float32)
for nom, cc_, aa in [('cuerpo', cuerpo_col, cuerpo_a), ('brazos', col, a * np.maximum(mL, mR)), ('manto', col, a * manto)]:
    Image.fromarray((cc_ * aa[..., None] + mag * (1 - aa[..., None])).astype(np.uint8)).save(f'{CTL}/capa_{nom}.jpg', quality=88)
grilla(np.dstack([np.clip(cuerpo_col, 0, 255).astype(np.uint8), (cuerpo_a * 255).astype(np.uint8)]), 'cuerpo')
log('capas del personaje')

# ================= 4. piezas de FLUX =================
piezas = {}
for p in sorted(glob.glob(os.path.join(D, 'piezas', '*.png'))):
    nom = os.path.splitext(os.path.basename(p))[0]
    cfg = CORTES.get('piezas', {}).get(nom, {})
    im = Image.open(p).convert('RGB')
    if cfg.get('tipo', 'recorte') == 'fondo':
        f = cubrir(esr(im), W, H)
        f.save(f'{CAPAS}/{nom}.jpg', quality=94)
        fn = np.asarray(f)
        grilla(fn, nom)
        # primer plano opcional: polígono que va delante del personaje (escritorio, bote)
        if cfg.get('frente'):
            m = poli(cfg['frente'], (H, W), 1.0)
            guardar_rgba(fn.astype(np.float32), m, f'{CAPAS}/{nom}_frente.png')
            Image.fromarray((fn * m[..., None] + mag * (1 - m[..., None])).astype(np.uint8)).save(f'{CTL}/frente_{nom}.jpg', quality=88)
        piezas[nom] = dict(tipo='fondo', w=W, h=H)
    else:
        esc = float(cfg.get('escala', 1.4))
        big = esr(im)
        tw, th = round(im.size[0] * esc), round(im.size[1] * esc)
        g = big.resize((tw, th), Image.LANCZOS)
        gn = np.asarray(g).copy()
        aa = alfa(g)
        if cfg.get('quitar'):
            aa = aa * (1 - poli(cfg['quitar'], aa.shape))
        if cfg.get('dejar'):
            aa = aa * poli(cfg['dejar'], aa.shape)
        bg = cv2.GaussianBlur(gn.astype(np.float32), (0, 0), 25)
        cc_ = descontaminar(gn.astype(np.float32), aa, bg)
        guardar_rgba(cc_, aa, f'{CAPAS}/{nom}.png')
        grilla(np.dstack([gn, (aa * 255).astype(np.uint8)]), nom)
        piezas[nom] = dict(tipo='recorte', w=tw, h=th)
    log('pieza', nom, piezas[nom])

layout = dict(anillo=dict(cx=round(float(cx), 2), cy=round(float(cy), 2), r=round(float(rr), 1), k=K, radio_tex=maxR),
              ojos=CORTES.get('ojos', [[512.4, 371.1], [588.1, 371.0]]), gema=CORTES.get('gema', [545.8, 761.7]),
              piezas=piezas)
json.dump(layout, open(f'{CAPAS}/layout.json', 'w'))
print(json.dumps(layout, indent=1))
log('listo')
