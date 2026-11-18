# Capas para el encapuchado en Godot (17/11/2026). Corre en GitHub Actions (sp777-render, godot.yml), nunca en la Mac.
# Entra fuente.png (768x1344, FLUX.2 klein) y deja en proj/capas/, a 1080x1920:
#   personaje.png  el personaje recortado (BiRefNet con rembg), bordes descontaminados y color corrido hacia afuera
#   fondo.jpg      el fondo sin personaje y sin el círculo de runas (LaMa rellena los huecos)
#   anillo.png     la luz del círculo de runas, completo (lo tapado se rehace con un tramo visible que se repite K veces),
#                  para sumarlo (aditivo) y girarlo 360/K grados por loop sin corte
#   anillo_glow.png  el mismo, desenfocado (pulso de brillo)
#   layout.json    centro y radios del anillo, ojos, gema, estrellas, caja del personaje
# Controles para revisar en out/prep/.
import json, os, sys, time, urllib.request
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
K = int(args.get('k', '8'))
DIL = int(args.get('dilatar', '18'))
CORTE = args.get('corte', 'birefnet-general')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


def bajar(url, path):
    if not os.path.exists(path):
        log('bajando', url)
        urllib.request.urlretrieve(url, path)
    return path


src = Image.open(os.path.join(D, 'fuente.png')).convert('RGB')
sw, sh = src.size
nw = round(sw * H / sh)
x0 = (nw - W) // 2

# ---------- 1. escala (Real-ESRGAN anime x4 y Lanczos a 1080x1920) ----------
import torch
torch.set_num_threads(os.cpu_count() or 4)
base = src.resize((nw, H), Image.LANCZOS).crop((x0, 0, x0 + W, H))
try:
    from spandrel import ModelLoader
    mp = bajar('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', '/tmp/esr.pth')
    model = ModelLoader().load_from_file(mp).eval()
    x = torch.from_numpy(np.asarray(src, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    T, O, S = 192, 16, 4
    out = torch.zeros(1, 3, sh * S, sw * S)
    with torch.no_grad():
        for ty in range(0, sh, T):
            for tx in range(0, sw, T):
                ya, yb = max(ty - O, 0), min(ty + T + O, sh)
                xa, xb = max(tx - O, 0), min(tx + T + O, sw)
                y = model(x[:, :, ya:yb, xa:xb])
                cy0, cy1 = (ty - ya) * S, (min(ty + T, sh) - ya) * S
                cx0, cx1 = (tx - xa) * S, (min(tx + T, sw) - xa) * S
                out[:, :, ty * S:min(ty + T, sh) * S, tx * S:min(tx + T, sw) * S] = y[:, :, cy0:cy1, cx0:cx1]
    big = Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8))
    base = big.resize((nw, H), Image.LANCZOS).crop((x0, 0, x0 + W, H))
    log('esr ok')
except Exception as e:
    log('ESR falló, uso Lanczos:', repr(e))
base_np = np.asarray(base).copy()
base.save(f'{CTL}/base.jpg', quality=92)

# ---------- 2. recorte ----------
from rembg import new_session, remove
a = None
for nombre in dict.fromkeys([CORTE, 'isnet-anime']):
    try:
        m = remove(base, session=new_session(nombre), only_mask=True)
        a = np.asarray(m.convert('L'), dtype=np.float32) / 255
        log('corte', nombre)
        break
    except Exception as e:
        log('corte falló', nombre, repr(e))
bin_ = (a > 0.5).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(bin_, 8)
if n > 1:
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    a = a * cv2.dilate((lab == k).astype(np.uint8), np.ones((9, 9), np.uint8))
inv = (a <= 0.5).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
for i in range(1, n):
    x, y, w, h, area = st[i]
    if area < 1500 and x > 0 and y > 0 and x + w < W and y + h < H:
        a[lab == i] = 1.0
# el piso oscuro entre las botas y debajo del ruedo no es personaje
CORTE_Y = float(args.get('corte_y', '1828'))
a = a * np.clip((CORTE_Y - np.arange(H, dtype=np.float32)) / 14, 0, 1)[:, None]
fig = (a > 0.02).astype(np.uint8)
hueco = cv2.dilate(fig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1)))
Image.fromarray((a * 255).astype(np.uint8)).save(f'{CTL}/mascara.png')

# ---------- 3. círculo de runas: ajuste ----------
bi = base_np.astype(np.int32)
Rr, Gg, Bb = bi[..., 0], bi[..., 1], bi[..., 2]
amar = (Rr + Gg) / 2 - Bb
oro = (amar > 45) & (np.maximum(Rr, Gg) > 140) & (hueco == 0)
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
log('anillo', round(cx, 1), round(cy, 1), round(rr, 1), round(r_in, 1), round(r_out, 1))
yy, xx = np.mgrid[0:H, 0:W]
dd = np.hypot(xx - cx, yy - cy)
MARG = int(args.get('margen_anillo', '34'))
banda = ((dd > r_in - MARG) & (dd < r_out + MARG)).astype(np.uint8)

# ---------- 4. fondo (LaMa: sin personaje y sin anillo) ----------
mask = np.maximum(hueco, banda)
lp = bajar('https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt', '/tmp/big-lama.pt')
lama = torch.jit.load(lp, map_location='cpu').eval()
with torch.no_grad():
    r = lama(torch.from_numpy(base_np.astype(np.float32) / 255).permute(2, 0, 1)[None],
             torch.from_numpy(mask.astype(np.float32))[None, None])
fondo = (r[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)[:H, :W]
fondo = np.where(mask[..., None] > 0, fondo, base_np)
Image.fromarray(fondo).save(f'{CAPAS}/fondo.jpg', quality=94)
log('lama')

# ---------- 5. anillo completo y periódico ----------
luz = np.clip(base_np.astype(np.float32) - fondo.astype(np.float32), 0, 255)
luz *= (banda * (1 - hueco))[..., None]
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
# tramo válido más largo (circular)
bueno = ~malo
ext = np.r_[bueno, bueno]
mejor, ini, cur, cs = 0, 0, 0, 0
for i, v in enumerate(ext):
    if v:
        if cur == 0:
            cs = i
        cur += 1
        if cur > mejor and cur <= A:
            mejor, ini = cur, cs
    else:
        cur = 0
log('tramo válido', mejor, 'filas de', A, '; hace falta', P + F)
if mejor < P + F:
    F = max(mejor - P, 0)
s0 = ini + max((mejor - (P + F)) // 2, 0)
seg = pol[(s0 + np.arange(P + F)) % A]
tile = seg[:P].copy()
for i in range(F):
    w = i / F
    tile[i] = seg[i] * w + seg[P + i] * (1 - w)
tira = np.tile(tile, (K, 1, 1))
S2 = 2 * maxR
anillo = cv2.warpPolar(tira, (S2, S2), (maxR, maxR), maxR, FL | cv2.WARP_INVERSE_MAP)
ry, rx = np.mgrid[0:S2, 0:S2]
rd = np.hypot(rx - maxR + 0.5, ry - maxR + 0.5)
anillo *= np.clip((r_out + MARG - rd) / 6, 0, 1)[..., None] * np.clip((rd - (r_in - MARG)) / 6, 0, 1)[..., None]
anillo = np.clip(anillo, 0, 255)
Image.fromarray(anillo.astype(np.uint8)).save(f'{CAPAS}/anillo.png')
glow = np.clip(cv2.GaussianBlur(anillo, (0, 0), 9) * 1.8, 0, 255)
Image.fromarray(glow.astype(np.uint8)).save(f'{CAPAS}/anillo_glow.png')
log('anillo listo', S2)

# ---------- 6. personaje: bordes descontaminados y color corrido hacia afuera ----------
af = a[..., None]
col = base_np.astype(np.float32)
bg = fondo.astype(np.float32)
borde = (af > 0.03) & (af < 0.97)
col = np.where(borde, np.clip((col - (1 - af) * bg) / np.maximum(af, 0.03), 0, 255), col)
val = (a > 0.5).astype(np.float32)
cc = col * val[..., None]
v = val.copy()
for _ in range(40):
    s_ = cv2.blur(cc, (3, 3))
    n_ = cv2.blur(v, (3, 3))
    nuevo = (v == 0) & (n_ > 1e-4)
    if not nuevo.any():
        break
    cc[nuevo] = s_[nuevo] / n_[nuevo][:, None]
    v[nuevo] = 1
col = np.where(val[..., None] > 0, col, cc)
rgba = np.dstack([np.clip(col, 0, 255).astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])
Image.fromarray(rgba, 'RGBA').save(f'{CAPAS}/personaje.png', optimize=True)
mag = np.array([255, 0, 200], np.float32)
Image.fromarray((col * af + mag * (1 - af)).astype(np.uint8)).save(f'{CTL}/control_magenta.jpg', quality=90)
# composición estática (como queda en el cuadro 0, sin efectos): fondo + anillo + personaje
comp = fondo.astype(np.float32)
x1, y1 = int(round(cx)) - maxR, int(round(cy)) - maxR
ax0, ay0 = max(x1, 0), max(y1, 0)
ax1, ay1 = min(x1 + S2, W), min(y1 + S2, H)
comp[ay0:ay1, ax0:ax1] += anillo[ay0 - y1:ay1 - y1, ax0 - x1:ax1 - x1]
comp = np.clip(comp, 0, 255)
Image.fromarray(comp.astype(np.uint8)).save(f'{CTL}/control_fondo_anillo.jpg', quality=90)
comp = col * af + comp * (1 - af)
Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(f'{CTL}/control_compuesto.jpg', quality=90)
log('personaje')

# ---------- 7. ojos, gema, estrellas ----------
cian = ((bi[..., 2] > 170) & (bi[..., 1] > 150) & (bi[..., 0] < 150) & (bi[..., 2] - bi[..., 0] > 70)).astype(np.uint8) * fig
n, lab, st, cen = cv2.connectedComponentsWithStats(cian, 8)
manchas = [dict(x=float(cen[i][0]), y=float(cen[i][1]), area=int(st[i][4]), w=int(st[i][2]), h=int(st[i][3]))
           for i in range(1, n) if st[i][4] >= 8]
# ojos: manchas chicas y compactas, las más cercanas a donde están en la ilustración (no el borde cian de la capucha)
ojos = []
for ex, ey in [(510.0, 374.0), (588.0, 374.0)]:
    cand = [m for m in manchas if m['w'] <= 50 and m['h'] <= 30 and np.hypot(m['x'] - ex, m['y'] - ey) < 40]
    cand.sort(key=lambda m: np.hypot(m['x'] - ex, m['y'] - ey))
    ojos.append([round(cand[0]['x'], 1), round(cand[0]['y'], 1)] if cand else [ex, ey])
gemas = sorted([m for m in manchas if 680 < m['y'] < 840 and 470 < m['x'] < 630], key=lambda m: -m['area'])
gema = [round(gemas[0]['x'], 1), round(gemas[0]['y'], 1)] if gemas else [549.0, 757.0]
gris = cv2.cvtColor(fondo, cv2.COLOR_RGB2GRAY).astype(np.float32)
fF = fondo.astype(np.float32)
satu = fF.max(axis=2) - fF.min(axis=2)
loc = (gris == cv2.dilate(gris, np.ones((15, 15), np.uint8))) & (gris > 170) & (satu < 80) & (hueco == 0) & (banda == 0)
sy, sx = np.nonzero(loc)
estrellas = []
for i in np.argsort(-gris[sy, sx]):
    x, y = int(sx[i]), int(sy[i])
    if y > 1250:
        continue
    if all((x - e[0]) ** 2 + (y - e[1]) ** 2 > 28 ** 2 for e in estrellas):
        estrellas.append([x, y, int(gris[y, x])])
    if len(estrellas) >= 80:
        break
ys2, xs2 = np.nonzero(fig)
layout = dict(anillo=dict(cx=round(float(cx), 2), cy=round(float(cy), 2), r=round(float(rr), 1), rin=round(r_in, 1),
                          rout=round(r_out, 1), radio_tex=maxR, k=K),
              ojos=ojos, gema=gema, estrellas=estrellas,
              caja=[int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())],
              manchas=sorted(manchas, key=lambda m: -m['area'])[:12])
with open(f'{CAPAS}/layout.json', 'w') as fh:
    json.dump(layout, fh)
print(json.dumps({k_: v_ for k_, v_ in layout.items() if k_ != 'estrellas'}, indent=1))
log('listo')
