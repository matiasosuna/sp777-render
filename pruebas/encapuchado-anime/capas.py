# Capas para animar una ilustración al estilo Live2D (encapuchado-anime, 17/11/2026).
# Entra fuente.png (768x1344, FLUX.2 klein) y salen, a 1080x1920:
#   base.jpg       la ilustración escalada (Real-ESRGAN anime x4 y después Lanczos)
#   personaje.png  el personaje recortado (BiRefNet o isnet-anime con rembg), bordes descontaminados
#   fondo.jpg      el fondo sin el personaje (LaMa rellena el hueco)
#   anillo.png     solo el trazo dorado del círculo de runas (para que brille encima)
#   capas.js       window.CAPAS con el centro y radio del anillo, ojos, gema y estrellas
# Corre en GitHub Actions (capas.yml), nunca en la Mac.
import json, os, sys, time, urllib.request
import numpy as np
import cv2
from PIL import Image

W, H = 1080, 1920
args = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
SRC = sys.argv[1]
OUT = 'out'
os.makedirs(OUT, exist_ok=True)
ESCALA = args.get('escala', 'esr')
CORTE = args.get('corte', 'birefnet-general')
DIL = int(args.get('dilatar', '20'))
T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)

def bajar(url, path):
    if not os.path.exists(path):
        log('bajando', url)
        urllib.request.urlretrieve(url, path)
    return path

src = Image.open(os.path.join(SRC, 'fuente.png')).convert('RGB')
sw, sh = src.size
nw = round(sw * H / sh)
x0 = (nw - W) // 2

# ---------- 1. escala ----------
import torch
torch.set_num_threads(os.cpu_count() or 4)
lanc = src.resize((nw, H), Image.LANCZOS).crop((x0, 0, x0 + W, H))
lanc.save(f'{OUT}/base_lanczos.png')
base = lanc
if ESCALA == 'esr':
    try:
        from spandrel import ModelLoader
        mp = bajar('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', '/tmp/esr.pth')
        model = ModelLoader().load_from_file(mp).eval()
        x = torch.from_numpy(np.asarray(src, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        T, O, S = 192, 16, 4
        out = torch.zeros(1, 3, sh * S, sw * S); wsum = torch.zeros(1, 1, sh * S, sw * S)
        with torch.no_grad():
            for ty in range(0, sh, T):
                for tx in range(0, sw, T):
                    ya, yb = max(ty - O, 0), min(ty + T + O, sh)
                    xa, xb = max(tx - O, 0), min(tx + T + O, sw)
                    y = model(x[:, :, ya:yb, xa:xb])
                    # se queda solo con el centro de cada mosaico (sin el borde de solape)
                    cy0, cy1 = (ty - ya) * S, (min(ty + T, sh) - ya) * S
                    cx0, cx1 = (tx - xa) * S, (min(tx + T, sw) - xa) * S
                    out[:, :, ty * S:min(ty + T, sh) * S, tx * S:min(tx + T, sw) * S] = y[:, :, cy0:cy1, cx0:cx1]
        big = (out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
        big = Image.fromarray(big)
        log('esr', big.size)
        base = big.resize((nw, H), Image.LANCZOS).crop((x0, 0, x0 + W, H))
        base.save(f'{OUT}/base_esr.png')
    except Exception as e:
        log('ESR falló, uso Lanczos:', repr(e))
base_np = np.asarray(base).copy()
base.save(f'{OUT}/base.jpg', quality=93)

# ---------- 2. recorte ----------
from rembg import new_session, remove
alphas = {}
for nombre in dict.fromkeys([CORTE, 'isnet-anime']):
    try:
        ses = new_session(nombre)
        m = remove(base, session=ses, only_mask=True)
        alphas[nombre] = np.asarray(m.convert('L'), dtype=np.float32) / 255
        Image.fromarray((alphas[nombre] * 255).astype(np.uint8)).save(f'{OUT}/mascara_{nombre}.png')
        log('corte', nombre)
    except Exception as e:
        log('corte falló', nombre, repr(e))
a = alphas.get(CORTE, next(iter(alphas.values())))

# limpiar: solo el componente principal (y lo que lo toca), sin agujeros chicos
bin_ = (a > 0.5).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(bin_, 8)
if n > 1:
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = (lab == k).astype(np.uint8)
    near = cv2.dilate(keep, np.ones((9, 9), np.uint8))
    a = a * near
inv = (a <= 0.5).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
for i in range(1, n):
    x, y, w, h, area = st[i]
    if area < 1500 and x > 0 and y > 0 and x + w < W and y + h < H:
        a[lab == i] = 1.0
fig = (a > 0.02).astype(np.uint8)

# ---------- 3. fondo (LaMa) ----------
hueco = cv2.dilate(fig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1)))
lp = bajar('https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt', '/tmp/big-lama.pt')
lama = torch.jit.load(lp, map_location='cpu').eval()
img_t = torch.from_numpy(base_np.astype(np.float32) / 255).permute(2, 0, 1)[None]
msk_t = torch.from_numpy(hueco.astype(np.float32))[None, None]
with torch.no_grad():
    r = lama(img_t, msk_t)
fondo = (r[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)[:H, :W]
fondo = np.where(hueco[..., None] > 0, fondo, base_np)
Image.fromarray(fondo).save(f'{OUT}/fondo.jpg', quality=93)
log('lama')

# ---------- 4. personaje con bordes descontaminados ----------
af = a[..., None]
col = base_np.astype(np.float32)
bg = fondo.astype(np.float32)
borde = (af > 0.03) & (af < 0.97)
uncol = np.clip((col - (1 - af) * bg) / np.maximum(af, 0.03), 0, 255)
col = np.where(borde, uncol, col)
rgba = np.dstack([col.astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])
Image.fromarray(rgba, 'RGBA').save(f'{OUT}/personaje.png', optimize=True)
# controles: el recorte sobre magenta y sobre el fondo rehecho
mag = np.array([255, 0, 200], np.float32)
Image.fromarray((col * af + mag * (1 - af)).astype(np.uint8)).save(f'{OUT}/control_magenta.jpg', quality=90)
log('personaje')

# ---------- 5. anillo de runas ----------
f = fondo.astype(np.int32)
R, G, B = f[..., 0], f[..., 1], f[..., 2]
amar = (R + G) / 2 - B
oro = ((amar > 45) & (np.maximum(R, G) > 140)).astype(np.uint8) * (1 - hueco)
ys, xs = np.nonzero(oro)
# ajuste de círculo (Kasa) y un refinamiento con los puntos cercanos
def ajusta(xs, ys):
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    return cx, cy, np.sqrt(c[2] + cx ** 2 + cy ** 2)
cx, cy, rr = ajusta(xs.astype(float), ys.astype(float))
for _ in range(3):
    d = np.hypot(xs - cx, ys - cy)
    ok = np.abs(d - rr) < 90
    cx, cy, rr = ajusta(xs[ok].astype(float), ys[ok].astype(float))
d = np.hypot(xs - cx, ys - cy)
ok = np.abs(d - rr) < 110
r_in, r_out = np.percentile(d[ok], 1), np.percentile(d[ok], 99)
log('anillo', cx, cy, rr, r_in, r_out)
yy, xx = np.mgrid[0:H, 0:W]
dd = np.hypot(xx - cx, yy - cy)
anul = np.clip(1 - np.maximum(r_in - 8 - dd, dd - r_out - 8) / 10, 0, 1)
fondoF = fondo.astype(np.float32)
lum = fondoF.max(axis=2)
amarF = (fondoF[..., 0] + fondoF[..., 1]) / 2 - fondoF[..., 2]
brillo = np.clip((lum - 110) / 120, 0, 1) * np.clip((amarF - 15) / 50, 0, 1) * anul
anillo = np.dstack([fondo * (brillo[..., None] > 0.004), (brillo * 255).astype(np.uint8)])
Image.fromarray(anillo, 'RGBA').save(f'{OUT}/anillo.png', optimize=True)

# ---------- 6. ojos, gema, estrellas ----------
fb = base_np.astype(np.int32)
cian = ((fb[..., 2] > 170) & (fb[..., 1] > 150) & (fb[..., 0] < 150) & (fb[..., 2] - fb[..., 0] > 70)).astype(np.uint8) * fig
n, lab, st, cen = cv2.connectedComponentsWithStats(cian, 8)
manchas = sorted([dict(x=round(float(cen[i][0]), 1), y=round(float(cen[i][1]), 1), area=int(st[i][4]),
                       w=int(st[i][2]), h=int(st[i][3])) for i in range(1, n) if st[i][4] >= 12], key=lambda m: -m['area'])[:20]
gris = cv2.cvtColor(fondo, cv2.COLOR_RGB2GRAY).astype(np.float32)
satu = fondoF.max(axis=2) - fondoF.min(axis=2)
loc = (gris == cv2.dilate(gris, np.ones((15, 15), np.uint8))) & (gris > 175) & (satu < 70) & (hueco == 0) & (np.abs(dd - rr) > 80)
sy, sx = np.nonzero(loc)
orden = np.argsort(-gris[sy, sx])
estrellas = []
for i in orden:
    x, y = int(sx[i]), int(sy[i])
    if y > 1150:
        continue
    if all((x - e[0]) ** 2 + (y - e[1]) ** 2 > 30 ** 2 for e in estrellas):
        estrellas.append([x, y, int(gris[y, x])])
    if len(estrellas) >= 70:
        break
ys2, xs2 = np.nonzero(fig)
caja = [int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())]
capas = dict(anillo=dict(cx=round(float(cx), 1), cy=round(float(cy), 1), r=round(float(rr), 1),
                         rin=round(float(r_in), 1), rout=round(float(r_out), 1)),
             cian=manchas, estrellas=estrellas, caja=caja, escala=ESCALA, corte=CORTE)
with open(f'{OUT}/capas.js', 'w') as fh:
    fh.write('window.CAPAS = ' + json.dumps(capas) + ';\n')
print(json.dumps(capas, indent=1))
log('listo')
