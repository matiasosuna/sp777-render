# Mago encapuchado (estilo FLUX/Pixar) riggeado y animado con mocap de tai chi (CMU 12_04), en Blender headless.
#
# Etapas (--stage):
#   sim    : arma todo, simula la tela (capa, túnica, mangas) siguiendo el esqueleto y guarda las posiciones en sim.npz
#   render : rearma la escena (rápido), carga sim.npz y renderiza un tramo de cuadros (--chunk i --nchunks n)
#   stills : igual que render pero 4 fotos fijas con más muestras
#
# Retarget: el BVH se importa tal cual (armadura A, pose T). Se copia a B, se bajan los brazos a pose A y esa
# pasa a ser la pose de reposo; cada hueso de B copia la rotación de mundo del mismo hueso de A. La ropa se
# modela en el reposo de B, se pesa por cercanía a los huesos y la tela se simula con la parte de arriba fijada.

import bpy, bmesh, math, os, sys, time, random, argparse
import numpy as np
from mathutils import Vector, Matrix

T0 = time.time()
def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='out')
ap.add_argument('--stage', default='sim', choices=['sim', 'render', 'stills'])
ap.add_argument('--cache', default='')
ap.add_argument('--chunk', type=int, default=0)
ap.add_argument('--nchunks', type=int, default=1)
ap.add_argument('--samples', type=int, default=0)
ap.add_argument('--look', default='pixar', choices=['pixar', 'anime'])
ap.add_argument('--bvh', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'taichi_cmu12_04.bvh'))
args = ap.parse_args(argv)
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
random.seed(7)
LOOK = args.look

FPS = 30
PRE = 31            # cuadros 1..30: la tela se acomoda mientras la pose pasa del reposo al mocap
CLIP_BVH0 = 90      # cuadro del BVH (60 fps) donde arranca el clip
NCLIP = 180         # 6 s a 30 fps
LAST = PRE + NCLIP - 1
SCALE = 0.066       # unidades del BVH -> metros (~1.70 m)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS

def new_coll(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c
C_CHAR = new_coll('personaje')
C_COL = new_coll('colisionadores')
C_ENV = new_coll('entorno')
CC = {k: bpy.data.collections.new('cc_' + k) for k in ('robe', 'coat', 'sleeve')}

# ======================================================================= utilidades de malla
def mesh_obj(name, verts, faces, coll=C_CHAR):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob

def smooth(ob):
    try:
        ob.data.shade_smooth()
    except Exception:
        ob.data.polygons.foreach_set('use_smooth', [True] * len(ob.data.polygons))

def catmull(P, n=24):
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(n):
            t = k / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    return np.array(out)

def resample(dense, n, cols=None):
    c = dense if cols is None else dense[:, cols]
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, dense[:, k]) for k in range(dense.shape[1])], axis=1)

def grid_faces(nv, nu, closed_u=False):
    f = []
    for i in range(nv - 1):
        for j in range(nu if closed_u else nu - 1):
            j2 = (j + 1) % nu
            f.append((i * nu + j, i * nu + j2, (i + 1) * nu + j2, (i + 1) * nu + j))
    return f

def arc_lengths(P, closed_u=False):
    du = np.linalg.norm(np.diff(P, axis=1), axis=2)
    su = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(du, axis=1)], axis=1)
    dv = np.linalg.norm(np.diff(P, axis=0), axis=2)
    sv = np.concatenate([np.zeros((1, P.shape[1])), np.cumsum(dv, axis=0)], axis=0)
    utot = su[:, -1] + (np.linalg.norm(P[:, 0] - P[:, -1], axis=1) if closed_u else 0)
    return su, sv, utot

def set_uv(ob, name, uv_vert):
    me = ob.data
    uvl = me.uv_layers.new(name=name)
    vidx = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get('vertex_index', vidx)
    uvl.data.foreach_set('uv', np.asarray(uv_vert, np.float32).reshape(-1, 2)[vidx].ravel())

def flip_if(ob, ref_fn):
    me = ob.data
    me.update()
    tot = sum(p.normal.dot(p.center - Vector(ref_fn(p))) for p in me.polygons)
    if tot < 0:
        bm = bmesh.new(); bm.from_mesh(me)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        bm.to_mesh(me); bm.free(); me.update()

def vgroup(ob, name, idx, w):
    vg = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
    if np.isscalar(w):
        vg.add([int(i) for i in idx], float(w), 'REPLACE')
    else:
        for i, ww in zip(idx, w):
            if ww > 1e-4:
                vg.add([int(i)], float(ww), 'REPLACE')
    return vg

def bm_to_obj(bm, name, coll=C_CHAR):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob

def bm_ellipsoid(bm, c, r, seg=24, rings=12, M=None):
    res = bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0)
    for v in res['verts']:
        p = Vector((v.co.x * r[0], v.co.y * r[1], v.co.z * r[2]))
        if M is not None:
            p = M @ p
        v.co = Vector(c) + p
    return res['verts']

def bm_capsule(bm, a, b, ra, rb, seg=16, caps=True):
    a, b = Vector(a), Vector(b)
    d = b - a
    if d.length < 1e-6:
        return bm_ellipsoid(bm, a, (ra, ra, ra), seg, seg // 2)
    res = bmesh.ops.create_cone(bm, cap_ends=False, segments=seg, radius1=ra, radius2=rb, depth=d.length)
    m = Matrix.Translation((a + b) / 2) @ d.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=m, verts=res['verts'])
    vs = list(res['verts'])
    if caps:
        vs += bm_ellipsoid(bm, a, (ra, ra, ra), seg, seg // 2)
        vs += bm_ellipsoid(bm, b, (rb, rb, rb), seg, seg // 2)
    return vs

# ======================================================================= runas (textura)
RUNES = [
    [(.3, 0, .3, 1), (.3, .5, .75, .75), (.3, .78, .75, 1.0)], [(.25, 0, .25, 1), (.25, 1, .75, .7), (.75, .7, .75, 0)],
    [(.3, 0, .3, 1), (.3, .75, .7, .5), (.7, .5, .3, .25)], [(.3, 0, .3, 1), (.3, 1, .7, .75), (.3, .7, .7, .45)],
    [(.3, 0, .3, 1), (.3, 1, .7, .78), (.7, .78, .3, .55), (.3, .55, .72, 0)], [(.65, 1, .3, .5), (.3, .5, .65, 0)],
    [(.25, 0, .75, 1), (.25, 1, .75, 0)], [(.25, 0, .25, 1), (.75, 0, .75, 1), (.25, .65, .75, .35)],
    [(.5, 0, .5, 1), (.3, .65, .7, .35)], [(.5, 0, .5, 1), (.5, 1, .75, .8), (.5, 0, .25, .2)],
    [(.5, 0, .5, 1), (.5, .6, .2, .95), (.5, .6, .8, .95)], [(.7, 1, .3, .7), (.3, .7, .7, .3), (.7, .3, .3, 0)],
    [(.5, 0, .5, 1), (.5, 1, .2, .7), (.5, 1, .8, .7)], [(.3, 0, .3, 1), (.3, 1, .7, .75), (.7, .75, .3, .5), (.3, .5, .7, .25), (.7, .25, .3, 0)],
    [(.25, 0, .25, 1), (.75, 0, .75, 1), (.25, 1, .75, .6), (.75, 1, .25, .6)], [(.2, 0, .2, 1), (.8, 0, .8, 1), (.2, 1, .8, 0), (.2, 0, .8, 1)],
]

def raster_segments(H, W, segs, width):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.zeros((H, W), np.float32)
    for (x0, y0, x1, y1) in segs:
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
        d = np.hypot(xx - x0 - t * dx, yy - y0 - t * dy)
        m = np.maximum(m, np.clip(width * 0.5 + 0.75 - d, 0, 1))
    return m

def pattern_image(name, vertical, kind='runes'):
    """Máscara gris 128 x 2048 (o al revés). kind='runes': fila de runas; kind='knot': trenza (panel central)."""
    Lpx, Wpx = 2048, 128
    m = np.zeros((Wpx, Lpx), np.float32)
    segs = []
    if kind == 'runes':
        cell = int(Wpx * 0.75)
        n = Lpx // cell
        gap = (Lpx - n * cell) / n
        rng = random.Random(11 if vertical else 5)
        for k in range(n):
            g = rng.choice(RUNES)
            ox = k * (cell + gap) + gap / 2 + cell * 0.12
            oy = Wpx * 0.5 - cell * 0.38
            sz = cell * 0.76
            for (x0, y0, x1, y1) in g:
                if vertical:
                    segs.append((ox + (1 - y0) * sz, oy + x0 * sz, ox + (1 - y1) * sz, oy + x1 * sz))
                else:
                    segs.append((ox + x0 * sz, oy + y0 * sz, ox + x1 * sz, oy + y1 * sz))
        m = np.maximum(m, raster_segments(Wpx, Lpx, segs, width=9))
    else:
        # trenza: dos senoides cruzadas + rombos
        xs = np.arange(Lpx, dtype=np.float32)
        per = Wpx * 1.4
        for ph in (0, math.pi):
            yc = Wpx * 0.5 + Wpx * 0.30 * np.sin(2 * math.pi * xs / per + ph)
            yy = np.arange(Wpx, dtype=np.float32)[:, None]
            m = np.maximum(m, np.clip(6.5 - np.abs(yy - yc[None, :]), 0, 1))
        for k in range(int(Lpx / per) + 1):
            cx = k * per + per / 4
            segs += [(cx - 12, Wpx / 2, cx, Wpx / 2 - 12), (cx, Wpx / 2 - 12, cx + 12, Wpx / 2),
                     (cx + 12, Wpx / 2, cx, Wpx / 2 + 12), (cx, Wpx / 2 + 12, cx - 12, Wpx / 2)]
        m = np.maximum(m, raster_segments(Wpx, Lpx, segs, width=4))
    if vertical:
        m = m.T
    h, w = m.shape
    img = bpy.data.images.new(name, w, h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    px = np.ones((h, w, 4), np.float32)
    px[..., 0] = px[..., 1] = px[..., 2] = m
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    return img

IMG_RUNES = pattern_image('runas_h', vertical=False)
IMG_KNOT = pattern_image('trenza_v', vertical=True, kind='knot')

# ======================================================================= materiales (look Pixar: tela mate con brillo de pelusa)
def nodes_clear(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return nt

def N(nt, typ, **kw):
    n = nt.nodes.new(typ)
    for k, v in kw.items():
        setattr(n, k, v)
    return n

def math_(nt, op, a, b=None, val=None):
    n = N(nt, 'ShaderNodeMath', operation=op)
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            nt.links.new(x, n.inputs[i])
    if val is not None:
        n.inputs[1].default_value = val
    return n.outputs[0]

def mixc(nt, fac, c1, c2):
    n = N(nt, 'ShaderNodeMixRGB', blend_type='MIX')
    nt.links.new(fac, n.inputs['Fac'])
    for sock, c in ((n.inputs['Color1'], c1), (n.inputs['Color2'], c2)):
        if isinstance(c, tuple):
            sock.default_value = (*c, 1)
        else:
            nt.links.new(c, sock)
    return n.outputs[0]

class _Wrap:
    def __init__(self, out):
        self.outputs = [out]; self.inputs = {}

RIM_VECS = []
def toon_bsdf(nt, color, gold=None):
    """Anime: difuso plano (el corte en bandas se hace en el compositor sobre la pasada DiffDir) + borde cian emisivo."""
    df = N(nt, 'ShaderNodeBsdfDiffuse')
    if isinstance(color, tuple):
        df.inputs['Color'].default_value = (*color, 1)
    else:
        nt.links.new(color, df.inputs['Color'])
    lw = N(nt, 'ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.5
    geo = N(nt, 'ShaderNodeNewGeometry')
    rv = N(nt, 'ShaderNodeCombineXYZ'); RIM_VECS.append(rv)
    dp = N(nt, 'ShaderNodeVectorMath', operation='DOT_PRODUCT')
    nt.links.new(geo.outputs['Normal'], dp.inputs[0]); nt.links.new(rv.outputs[0], dp.inputs[1])
    side = math_(nt, 'GREATER_THAN', dp.outputs['Value'], val=0.05)
    edge = math_(nt, 'GREATER_THAN', lw.outputs['Facing'], val=0.74)
    rim = math_(nt, 'MULTIPLY', math_(nt, 'MULTIPLY', side, edge), val=0.8)
    em = N(nt, 'ShaderNodeEmission'); em.inputs['Color'].default_value = (0.22, 0.85, 1.0, 1)
    nt.links.new(rim, em.inputs['Strength'])
    add = N(nt, 'ShaderNodeAddShader')
    nt.links.new(df.outputs[0], add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1])
    if gold is not None:
        eg = N(nt, 'ShaderNodeEmission'); eg.inputs['Color'].default_value = (*GOLD, 1)
        nt.links.new(math_(nt, 'MULTIPLY', gold, val=0.30), eg.inputs['Strength'])
        add2 = N(nt, 'ShaderNodeAddShader')
        nt.links.new(add.outputs[0], add2.inputs[0]); nt.links.new(eg.outputs[0], add2.inputs[1])
        add = add2
    return _Wrap(add.outputs[0])

def fabric_bsdf(nt, color, gold=None, rough=0.85, sheen=0.35, bump=0.25):
    if LOOK == 'anime':
        return toon_bsdf(nt, color, gold)
    bs = N(nt, 'ShaderNodeBsdfPrincipled')
    if isinstance(color, tuple):
        bs.inputs['Base Color'].default_value = (*color, 1)
    else:
        nt.links.new(color, bs.inputs['Base Color'])
    bs.inputs['Roughness'].default_value = rough
    bs.inputs['Sheen Weight'].default_value = sheen
    bs.inputs['Sheen Roughness'].default_value = 0.4
    bs.inputs['Sheen Tint'].default_value = (0.55, 0.6, 0.9, 1)
    if gold is not None:
        nt.links.new(gold, bs.inputs['Metallic'])
        r = N(nt, 'ShaderNodeMapRange')
        nt.links.new(gold, r.inputs['Value'])
        r.inputs['To Min'].default_value = rough; r.inputs['To Max'].default_value = 0.32
        nt.links.new(r.outputs[0], bs.inputs['Roughness'])
    if bump > 0:
        tc = N(nt, 'ShaderNodeTexCoord')
        nz = N(nt, 'ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 900.0; nz.inputs['Detail'].default_value = 3.0
        nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
        bp = N(nt, 'ShaderNodeBump'); bp.inputs['Strength'].default_value = bump; bp.inputs['Distance'].default_value = 0.001
        nt.links.new(nz.outputs['Fac'], bp.inputs['Height'])
        nt.links.new(bp.outputs['Normal'], bs.inputs['Normal'])
    return bs

GOLD = (0.80, 0.55, 0.22)
PURPLE = (0.075, 0.030, 0.17)
NAVY = (0.020, 0.028, 0.068)
ROBE = (0.012, 0.016, 0.050)

def band_mat(name, base, facing=PURPLE, face_w=0.028, rune_w=0.05, runes=True):
    """UV 'trim' = (distancia al borde en m, posición a lo largo en m); UV 'kind' x: 0 = ruedo/puño (runas), 1 = borde delantero."""
    mat = bpy.data.materials.new(name)
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    uvt = N(nt, 'ShaderNodeUVMap', uv_map='trim'); uvk = N(nt, 'ShaderNodeUVMap', uv_map='kind')
    st = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(uvt.outputs['UV'], st.inputs[0])
    sk = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(uvk.outputs['UV'], sk.inputs[0])
    a = st.outputs['X']; along = st.outputs['Y']
    is_front = math_(nt, 'GREATER_THAN', sk.outputs['X'], val=0.5)
    is_hem = math_(nt, 'SUBTRACT', 1.0, is_front)
    face = math_(nt, 'MULTIPLY', math_(nt, 'LESS_THAN', a, val=face_w), math_(nt, 'GREATER_THAN', a, val=-0.001))
    g0 = face_w + 0.001
    line1 = math_(nt, 'MULTIPLY', math_(nt, 'GREATER_THAN', a, val=g0), math_(nt, 'LESS_THAN', a, val=g0 + 0.005))
    gold = line1
    if runes:
        r0 = g0 + 0.009
        inband = math_(nt, 'MULTIPLY', math_(nt, 'GREATER_THAN', a, val=r0), math_(nt, 'LESS_THAN', a, val=r0 + rune_w))
        # uv de la imagen: x = a lo largo / (16 * ancho), y = (a - r0) / ancho
        ux = math_(nt, 'DIVIDE', along, val=16 * rune_w)
        uy = math_(nt, 'DIVIDE', math_(nt, 'SUBTRACT', a, val=r0), val=rune_w)
        cv = N(nt, 'ShaderNodeCombineXYZ'); nt.links.new(ux, cv.inputs['X']); nt.links.new(uy, cv.inputs['Y'])
        tx = N(nt, 'ShaderNodeTexImage', image=IMG_RUNES, extension='REPEAT', interpolation='Linear')
        nt.links.new(cv.outputs[0], tx.inputs['Vector'])
        rmask = math_(nt, 'MULTIPLY', math_(nt, 'MULTIPLY', tx.outputs['Color'], inband), is_hem)
        line2 = math_(nt, 'MULTIPLY', math_(nt, 'MULTIPLY', math_(nt, 'GREATER_THAN', a, val=r0 + rune_w), math_(nt, 'LESS_THAN', a, val=r0 + rune_w + 0.004)), is_hem)
        gold = math_(nt, 'MAXIMUM', math_(nt, 'MAXIMUM', line1, rmask), line2)
        bandbg = math_(nt, 'MULTIPLY', inband, is_hem)
    col = (base[0], base[1], base[2])
    c = mixc(nt, face, col, facing)
    if runes:
        c = mixc(nt, bandbg, c, tuple(x * 0.7 for x in base))
    c = mixc(nt, gold, c, GOLD)
    bs = fabric_bsdf(nt, c, gold=gold)
    nt.links.new(bs.outputs[0], out.inputs['Surface'])
    return mat

def plain_mat(name, color, rough=0.85, sheen=0.5, metal=0.0, bump=0.25):
    mat = bpy.data.materials.new(name)
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    if LOOK == 'anime' and metal > 0:
        one = N(nt, 'ShaderNodeValue'); one.outputs[0].default_value = 1.0
        bs = toon_bsdf(nt, color, gold=one.outputs[0])
    else:
        bs = fabric_bsdf(nt, color, rough=rough, sheen=sheen, bump=bump)
    if 'Metallic' in bs.inputs:
        bs.inputs['Metallic'].default_value = metal
    nt.links.new(bs.outputs[0], out.inputs['Surface'])
    return mat

def emit_mat(name, color, strength):
    mat = bpy.data.materials.new(name)
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    e = N(nt, 'ShaderNodeEmission'); e.inputs['Color'].default_value = (*color, 1); e.inputs['Strength'].default_value = strength
    nt.links.new(e.outputs[0], out.inputs['Surface'])
    return mat

def robe_mat():
    """Túnica con panel central bordado: UV 'trim' = (distancia firmada al centro en m, a lo largo en m)."""
    mat = bpy.data.materials.new('tunica')
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    uvt = N(nt, 'ShaderNodeUVMap', uv_map='trim')
    st = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(uvt.outputs['UV'], st.inputs[0])
    sd = st.outputs['X']; along = st.outputs['Y']
    HW = 0.058
    ad = math_(nt, 'ABSOLUTE', sd)
    panel = math_(nt, 'LESS_THAN', ad, val=HW)
    border = math_(nt, 'MULTIPLY', math_(nt, 'GREATER_THAN', ad, val=HW - 0.007), panel)
    inner = math_(nt, 'MULTIPLY', math_(nt, 'GREATER_THAN', ad, val=HW - 0.012), math_(nt, 'LESS_THAN', ad, val=HW - 0.0095))
    ux = math_(nt, 'DIVIDE', math_(nt, 'ADD', sd, val=HW - 0.013), val=2 * (HW - 0.013))
    uy = math_(nt, 'DIVIDE', along, val=-16 * 2 * (HW - 0.013))
    cv = N(nt, 'ShaderNodeCombineXYZ'); nt.links.new(ux, cv.inputs['X']); nt.links.new(uy, cv.inputs['Y'])
    tx = N(nt, 'ShaderNodeTexImage', image=IMG_KNOT, extension='REPEAT', interpolation='Linear')
    nt.links.new(cv.outputs[0], tx.inputs['Vector'])
    patt = math_(nt, 'MULTIPLY', tx.outputs['Color'], math_(nt, 'LESS_THAN', ad, val=HW - 0.013))
    gold = math_(nt, 'MAXIMUM', math_(nt, 'MAXIMUM', border, inner), patt)
    c = mixc(nt, panel, ROBE, (0.006, 0.008, 0.02))
    c = mixc(nt, gold, c, GOLD)
    bs = fabric_bsdf(nt, c, gold=gold)
    nt.links.new(bs.outputs[0], out.inputs['Surface'])
    return mat

M = {}
M['coat'] = band_mat('capa', NAVY)
M['coat_in'] = plain_mat('capa_forro', PURPLE)
M['sleeve'] = band_mat('manga', NAVY)
M['hood'] = band_mat('capucha', NAVY, face_w=0.034, runes=False)
M['hood_in'] = plain_mat('capucha_dentro', (0.004, 0.003, 0.008), sheen=0.0)
M['robe'] = robe_mat()
M['robe_in'] = plain_mat('tunica_dentro', (0.005, 0.005, 0.012))
M['rope'] = plain_mat('soga', (0.30, 0.17, 0.075), rough=0.7, sheen=0.3, bump=0.6)
M['gold'] = plain_mat('oro', (0.85, 0.60, 0.25), rough=0.3, sheen=0.0, metal=1.0, bump=0.0)
M['skin'] = plain_mat('piel', (0.78, 0.58, 0.48), rough=0.6, sheen=0.1, bump=0.05)
M['face'] = emit_mat('cara', (0.0, 0.0, 0.0), 0.0)
M['eye'] = emit_mat('ojos', (0.25, 1.0, 0.92), 14.0)
M['gem'] = emit_mat('gema', (0.05, 0.75, 0.95), 1.6)

# ======================================================================= esqueleto: importar BVH y retarget
fps_save = scene.render.fps
bpy.ops.import_anim.bvh(filepath=args.bvh, global_scale=SCALE, frame_start=1, use_fps_scale=True,
                        update_scene_fps=False, update_scene_duration=False, rotate_mode='NATIVE',
                        axis_forward='-Z', axis_up='Y')
A = bpy.context.active_object
A.name = 'mocap'
log('BVH importado; matriz A identidad:', A.matrix_world == Matrix.Identity(4), 'huesos', len(A.data.bones))
shift = PRE - (1 + CLIP_BVH0 * FPS / 120.0 * 2)  # cuadro BVH k -> 1 + k/2 a 30 fps
for fc in A.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.co.x += shift; kp.handle_left.x += shift; kp.handle_right.x += shift
    fc.update()
log('desplazamiento de clave', shift)

def joint_world(arm, name, tail=False):
    pb = arm.pose.bones[name]
    return arm.matrix_world @ (pb.tail if tail else pb.head)

# orientación del mocap al comenzar el clip y altura del piso
scene.frame_set(PRE)
lw = joint_world(A, 'LeftArm') - joint_world(A, 'RightArm'); lw.z = 0; lw.normalize()
YAW = math.atan2(lw.y, lw.x)
HIPS0 = joint_world(A, 'Hips')
zs = []
for f in range(PRE, LAST + 1, 3):
    scene.frame_set(f)
    for n in ('LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot'):
        zs.append(joint_world(A, n).z)
FLOOR = min(zs) - 0.03
log('yaw %.1f°, cadera %s, piso %.3f' % (math.degrees(YAW), tuple(round(v, 3) for v in HIPS0), FLOOR))
scene.frame_set(1)

# B = copia de A, sin animación, en pose A
B = A.copy(); B.data = A.data.copy(); B.name = 'rig'; B.data.name = 'rig'
scene.collection.objects.link(B)
B.animation_data_clear()
for pb in B.pose.bones:
    pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.rotation_euler = (0, 0, 0); pb.scale = (1, 1, 1)
B.matrix_world = Matrix.Identity(4)
bpy.context.view_layer.objects.active = B
for o in bpy.context.view_layer.objects:
    o.select_set(o == B)
bpy.ops.object.mode_set(mode='EDIT')
eb = B.data.edit_bones
# 1) que la izquierda del personaje quede en +x (mira hacia -y)
L = eb['LeftArm'].head - eb['RightArm'].head
rot0 = Matrix.Rotation(-math.atan2(L.y, L.x), 4, 'Z')
for e in eb:
    e.transform(rot0)
# 2) brazos abajo (pose A, 30° del cuerpo) con el antebrazo apenas hacia adelante
def chain(root):
    out = [eb[root]]
    for e in eb:
        p = e.parent
        while p is not None:
            if p.name == root:
                out.append(e); break
            p = p.parent
    return out
for side, sgn in (('Left', 1), ('Right', -1)):
    piv = eb[side + 'Arm'].head.copy()
    d = (eb[side + 'ForeArm'].head - piv).normalized()
    target = Vector((sgn * math.sin(math.radians(28)), -0.05, -math.cos(math.radians(28)))).normalized()
    R = d.rotation_difference(target).to_matrix().to_4x4()
    Mx = Matrix.Translation(piv) @ R @ Matrix.Translation(-piv)
    for e in chain(side + 'Arm'):
        e.transform(Mx)
    # codo apenas doblado hacia adelante
    piv2 = eb[side + 'ForeArm'].head.copy()
    R2 = Matrix.Rotation(math.radians(-12), 4, Vector((1, 0, 0)))
    Mx2 = Matrix.Translation(piv2) @ R2 @ Matrix.Translation(-piv2)
    for e in chain(side + 'ForeArm'):
        e.transform(Mx2)
bpy.ops.object.mode_set(mode='OBJECT')

JN = {b.name: Vector(b.head_local) for b in B.data.bones}
JT = {b.name: Vector(b.tail_local) for b in B.data.bones}
up = (JN['Head'] - JN['Hips']).normalized()
log('reposo B: arriba', tuple(round(v, 2) for v in up), 'hombro izq', tuple(round(v, 3) for v in JN['LeftArm']))
toe_rest = min(JN['LeftToeBase'].z, JN['RightToeBase'].z)
F0 = toe_rest - 0.03  # piso en coordenadas de B
B.location = Vector((HIPS0.x, HIPS0.y, FLOOR - F0))
B.rotation_euler = (0, 0, YAW)
_rim = Matrix.Rotation(YAW, 3, 'Z') @ Vector((0.0, 1.0, 0.3)).normalized()
for rv in RIM_VECS:
    rv.inputs[0].default_value, rv.inputs[1].default_value, rv.inputs[2].default_value = _rim
if LOOK == 'anime':
    M['gem'].node_tree.nodes['Emission'].inputs['Strength'].default_value = 6.0

# B copia las rotaciones de mundo de A; entra de a poco en los cuadros de preparación.
# Pies plantados: cadera y piernas quedan en reposo (sin pasos ni postura ancha); la torsión sale de la columna.
LEG_BONES = {'Hips', 'LHipJoint', 'RHipJoint', 'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg',
             'LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase'}
GAIN = {'LowerBack': 0.75, 'Spine': 0.85}
for pb in B.pose.bones:
    if pb.name in LEG_BONES:
        continue
    c = pb.constraints.new('COPY_ROTATION'); c.target = A; c.subtarget = pb.name
    c.owner_space = 'WORLD'; c.target_space = 'WORLD'
    g = GAIN.get(pb.name, 1.0)
    c.influence = 0.0; c.keyframe_insert('influence', frame=1)
    c.influence = 0.0; c.keyframe_insert('influence', frame=2)
    c.influence = g; c.keyframe_insert('influence', frame=PRE - 11)
if B.animation_data and B.animation_data.action:
    for fc in B.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'; kp.easing = 'EASE_IN_OUT'
A.hide_render = True; B.hide_render = True

# ======================================================================= pesos por cercanía a segmentos de hueso
def seg(n):
    return {
        'Hips': (JN['Hips'], JN['Hips'] + Vector((0, 0, -0.09))),
        'LowerBack': (JN['Hips'], JN['Spine']),
        'Spine': (JN['Spine'], JN['Spine1']),
        'Spine1': (JN['Spine1'], JN['Spine1'] + Vector((0, 0, 0.07))),
        'Neck': (JN['Spine1'] + Vector((0, 0, 0.05)), JN['Neck1']),
        'Neck1': (JN['Neck1'], JN['Head']),
        'Head': (JN['Head'], JT['Head']),
        'LeftShoulder': (JN['Spine1'] + (JN['LeftArm'] - JN['Spine1']) * 0.4, JN['LeftArm']),
        'RightShoulder': (JN['Spine1'] + (JN['RightArm'] - JN['Spine1']) * 0.4, JN['RightArm']),
        'LeftArm': (JN['LeftArm'], JN['LeftForeArm']), 'RightArm': (JN['RightArm'], JN['RightForeArm']),
        'LeftForeArm': (JN['LeftForeArm'], JN['LeftHand']), 'RightForeArm': (JN['RightForeArm'], JN['RightHand']),
        'LeftHand': (JN['LeftHand'], JT['LeftHandIndex1']), 'RightHand': (JN['RightHand'], JT['RightHandIndex1']),
        'LHipJoint': (JN['Hips'], JN['LeftUpLeg']), 'RHipJoint': (JN['Hips'], JN['RightUpLeg']),
        'LeftUpLeg': (JN['LeftUpLeg'], JN['LeftLeg']), 'RightUpLeg': (JN['RightUpLeg'], JN['RightLeg']),
        'LeftLeg': (JN['LeftLeg'], JN['LeftFoot']), 'RightLeg': (JN['RightLeg'], JN['RightFoot']),
        'LeftFoot': (JN['LeftFoot'], JN['LeftToeBase']), 'RightFoot': (JN['RightFoot'], JN['RightToeBase']),
        'LeftToeBase': (JN['LeftToeBase'], JT['LeftToeBase']), 'RightToeBase': (JN['RightToeBase'], JT['RightToeBase']),
    }[n]

def seg_dist(P, a, b):
    a = np.array(a); b = np.array(b)
    d = b - a
    t = np.clip(((P - a) @ d) / (d @ d + 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)

def skin_prox(ob, bones, power=4.0, maxinf=3, eps=0.03):
    me = ob.data
    P = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', P); P = P.reshape(-1, 3)
    D = np.stack([seg_dist(P, *seg(b)) for b in bones], 1)
    W = 1.0 / (D + eps) ** power
    if len(bones) > maxinf:
        thr = -np.sort(-W, axis=1)[:, maxinf - 1:maxinf]
        W = np.where(W >= thr, W, 0)
    W /= W.sum(1, keepdims=True)
    for k, b in enumerate(bones):
        idx = np.nonzero(W[:, k] > 1e-3)[0]
        vgroup(ob, b, idx, W[idx, k])

def skin_rigid(ob, bone):
    vgroup(ob, bone, range(len(ob.data.vertices)), 1.0)

def add_armature(ob):
    ob.parent = B
    m = ob.modifiers.new('Armature', 'ARMATURE')
    m.object = B
    m.use_vertex_groups = True
    return m

# ======================================================================= cuerpo invisible (colisionadores que se mueven con el rig)
def collider(name, specs, extra=0.0):
    bm = bmesh.new()
    groups = []
    for bone, a, b, ra, rb in specs:
        vs = bm_capsule(bm, a, b, ra + extra, rb + extra, seg=14)
        groups.append((bone, vs))
    bm.verts.index_update()
    gi = [(bone, [v.index for v in vs]) for bone, vs in groups]
    ob = bm_to_obj(bm, name, C_COL)
    for bone, idx in gi:
        vgroup(ob, bone, idx, 1.0)
    add_armature(ob)
    ob.modifiers.new('Collision', 'COLLISION')
    ob.collision.thickness_outer = 0.006
    ob.collision.cloth_friction = 0.8
    ob.hide_render = True
    return ob

def head_center():
    return JN['Head'] + (JT['Head'] - JN['Head']) * 0.45 + Vector((0, -0.01, 0))

torso = [('LowerBack', JN['Hips'], JN['Spine'], 0.125, 0.12), ('Spine', JN['Spine'], JN['Spine1'], 0.12, 0.125),
         ('Spine1', JN['Spine1'], JN['Spine1'] + Vector((0, 0, 0.06)), 0.125, 0.10), ('Neck1', JN['Neck1'], JN['Head'], 0.05, 0.05),
         ('Head', head_center(), head_center(), 0.10, 0.10)]
legs = []
arms = []
for s in ('Left', 'Right'):
    legs += [(s + 'UpLeg', JN[s + 'UpLeg'], JN[s + 'Leg'], 0.075, 0.058), (s + 'Leg', JN[s + 'Leg'], JN[s + 'Foot'], 0.055, 0.042),
             (s + 'Foot', JN[s + 'Foot'], JN[s + 'ToeBase'], 0.04, 0.03)]
    arms += [(s + 'Arm', JN[s + 'Arm'], JN[s + 'ForeArm'], 0.048, 0.042), (s + 'ForeArm', JN[s + 'ForeArm'], JN[s + 'Hand'], 0.04, 0.032),
             (s + 'Hand', JN[s + 'Hand'], JT[s + 'HandIndex1'], 0.03, 0.025)]
hipsph = [('Hips', JN['Hips'] + Vector((0, 0, -0.04)), JN['Hips'] + Vector((0, 0, -0.04)), 0.13, 0.13)]
COL_BODY = collider('col_cuerpo', torso + legs + hipsph)
COL_ARMS = collider('col_brazos', arms)
COL_FAT = collider('col_piernas_anchas', legs + hipsph, extra=0.035)
COL_SH = collider('col_hombros', torso[:3] + [a for a in arms if a[0].endswith('Arm')], extra=0.05)
CC['cape'] = bpy.data.collections.new('cc_cape')

bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, FLOOR))
FLOOR_OB = bpy.context.active_object; FLOOR_OB.name = 'piso'
for c in list(FLOOR_OB.users_collection):
    c.objects.unlink(FLOOR_OB)
C_ENV.objects.link(FLOOR_OB)
FLOOR_OB.modifiers.new('Collision', 'COLLISION')
FLOOR_OB.collision.thickness_outer = 0.004
FLOOR_OB.collision.cloth_friction = 4.0

for k, obs in (('robe', [COL_BODY, FLOOR_OB]), ('coat', [COL_FAT, COL_BODY, COL_ARMS, FLOOR_OB]), ('sleeve', [COL_ARMS, COL_BODY]), ('cape', [COL_SH, COL_ARMS])):
    for o in obs:
        CC[k].objects.link(o)

# ======================================================================= ropa
J = JN
Z1, ZS, ZH, ZN, ZHD = J['Spine1'].z, J['Spine'].z, J['Hips'].z, J['Neck1'].z, J['Head'].z
ZSH = J['LeftArm'].z
YC = J['Spine1'].y
HEM = F0 + 0.03
log('alturas (B): piso %.3f cadera %.3f spine %.3f spine1 %.3f hombro %.3f cuello %.3f cabeza %.3f' % (F0, ZH, ZS, Z1, ZSH, ZN, ZHD))

def ring_rows(keys, n, nu, open_angle=False):
    rows = resample(catmull(keys, 30), n, cols=[0, 1])
    P = np.zeros((n, nu, 3))
    for i, r in enumerate(rows):
        z, rx, ry, yc = r[:4]
        if open_angle:
            a = np.linspace(r[4], 2 * np.pi - r[4], nu)
        else:
            a = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        P[i, :, 0] = rx * np.sin(a)
        P[i, :, 1] = yc - ry * np.cos(a)
        P[i, :, 2] = z
    return P

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def cloth_mod(ob, key, *, mass=0.3, tension=15, bending=0.5, shear=5, air=1.5, quality=6, self_col=False, dist=0.006, pin_stiff=1.0):
    mod = ob.modifiers.new('Cloth', 'CLOTH')
    s = mod.settings
    s.quality = quality; s.mass = mass; s.air_damping = air
    s.tension_stiffness = tension; s.compression_stiffness = tension; s.shear_stiffness = shear; s.bending_stiffness = bending
    s.tension_damping = 5; s.compression_damping = 5; s.shear_damping = 5; s.bending_damping = 0.5
    s.vertex_group_mass = 'pin'; s.pin_stiffness = pin_stiff
    c = mod.collision_settings
    c.use_collision = True; c.collection = CC[key]; c.distance_min = dist; c.collision_quality = 4
    c.use_self_collision = self_col
    if self_col:
        c.self_distance_min = 0.005
    mod.point_cache.frame_start = 1; mod.point_cache.frame_end = LAST
    return mod

# --- túnica interior
def make_robe():
    keys = [(Z1 + 0.03, 0.150, 0.110, YC), (ZS, 0.150, 0.115, YC), (ZH + 0.03, 0.160, 0.118, J['Hips'].y),
            (ZH - 0.08, 0.200, 0.150, J['Hips'].y), (F0 + 0.45, 0.255, 0.200, J['Hips'].y), (HEM - 0.02, 0.300, 0.230, J['Hips'].y)]
    NU, NV = 96, 54
    P = ring_rows(keys, NV, NU)
    ob = mesh_obj('tunica', P.reshape(-1, 3), grid_faces(NV, NU, closed_u=True))
    flip_if(ob, lambda p: (0, YC, p.center.z))
    su, sv, ut = arc_lengths(P, closed_u=True)
    sd = np.where(su < ut[:, None] / 2, su, su - ut[:, None])
    set_uv(ob, 'trim', np.stack([np.clip(sd, -0.5, 0.5), sv], -1))
    z = P[:, 0, 2]
    pw = np.maximum(smoothstep(ZH - 0.02, ZH + 0.05, z), smoothstep(Z1 - 0.02, Z1 + 0.02, z))
    pw = np.repeat(pw, NU)
    vgroup(ob, 'pin', np.arange(len(pw)), pw)
    skin_prox(ob, ['Hips', 'LowerBack', 'Spine', 'Spine1', 'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg'])
    ob.data.materials.append(M['robe']); ob.data.materials.append(M['robe_in'])
    smooth(ob)
    return ob

# --- capa (abierta adelante)
def make_coat():
    ycs = YC
    keys = [  # z, rx, ry, yc, media apertura
        (ZN - 0.005, 0.072, 0.068, ycs + 0.005, 0.30),
        (ZSH + 0.040, 0.160, 0.115, ycs, 0.36),
        (ZSH - 0.025, 0.186, 0.130, ycs, 0.40),
        (Z1 - 0.06, 0.190, 0.135, ycs, 0.44),
        (ZH, 0.228, 0.172, J['Hips'].y, 0.48),
        (F0 + 0.40, 0.290, 0.225, J['Hips'].y, 0.52),
        (HEM - 0.012, 0.340, 0.265, J['Hips'].y, 0.54),
    ]
    NU, NV = 124, 60
    P = ring_rows(keys, NV, NU, open_angle=True)
    ob = mesh_obj('capa', P.reshape(-1, 3), grid_faces(NV, NU))
    flip_if(ob, lambda p: (0, ycs, p.center.z))
    su, sv, ut = arc_lengths(P)
    dfront = np.minimum(su, ut[:, None] - su)
    dhem = sv[-1][None, :] - sv
    kind = (dfront < dhem).astype(float)
    across = np.where(kind > 0.5, dfront, dhem)
    along = np.where(kind > 0.5, sv, su)
    set_uv(ob, 'trim', np.stack([across, along], -1))
    set_uv(ob, 'kind', np.stack([kind, np.zeros_like(kind)], -1))
    z = P[:, :, 2]
    pw = smoothstep(ZS - 0.05, Z1 - 0.02, z)
    vgroup(ob, 'pin', np.arange(pw.size), pw.ravel())
    skin_prox(ob, ['Hips', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'LeftShoulder', 'RightShoulder', 'LeftUpLeg', 'RightUpLeg'])
    ob.data.materials.append(M['coat']); ob.data.materials.append(M['coat_in'])
    smooth(ob)
    return ob

# --- mangas acampanadas
def make_sleeve(side, sgn):
    S, E, Wr = J[side + 'Arm'], J[side + 'ForeArm'], J[side + 'Hand']
    fwd = (Wr - E).normalized()
    top = S + Vector((-sgn * 0.010, 0, 0.035))
    C = Wr + fwd * 0.02
    pts = [tuple(top), tuple(S + (E - S) * 0.5), tuple(E), tuple(E + (Wr - E) * 0.5), tuple(C)]
    nv, nu = 40, 40
    path = resample(catmull(pts, 30), nv)
    T = np.gradient(path, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    Nn = np.cross(T[0], [0, -1.0, 0]); Nn /= np.linalg.norm(Nn)
    seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seglen)]); fr = s / s[-1]
    rk = np.array([(0, 0.072), (0.30, 0.070), (0.55, 0.085), (0.80, 0.122), (1.0, 0.160)])
    r = np.interp(fr, rk[:, 0], rk[:, 1])
    a = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    P = np.zeros((nv, nu, 3))
    for i in range(nv):
        if i > 0:
            Nn = Nn - T[i] * np.dot(Nn, T[i]); Nn /= np.linalg.norm(Nn)
        Bn = np.cross(T[i], Nn)
        P[i] = path[i] + r[i] * (np.cos(a)[:, None] * Nn + np.sin(a)[:, None] * Bn)
    ob = mesh_obj('manga_' + side, P.reshape(-1, 3), grid_faces(nv, nu, closed_u=True))
    me = ob.data; me.update()
    tot = sum(p.normal.dot(p.center - Vector(path[min(p.vertices[0] // nu, nv - 1)])) for p in me.polygons)
    if tot < 0:
        bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.reverse_faces(bm, faces=bm.faces[:]); bm.to_mesh(me); bm.free()
    su, sv, ut = arc_lengths(P, closed_u=True)
    dcuff = sv[-1][None, :] - sv
    set_uv(ob, 'trim', np.stack([dcuff, su], -1))
    set_uv(ob, 'kind', np.zeros((nv * nu, 2)))
    pw = 1.0 - smoothstep(0.10, 0.42, fr)
    vgroup(ob, 'pin', np.arange(nv * nu), np.repeat(pw, nu))
    skin_prox(ob, [side + 'Shoulder', side + 'Arm', side + 'ForeArm'])
    ob.data.materials.append(M['sleeve']); ob.data.materials.append(M['coat_in'])
    smooth(ob)
    return ob

# --- capucha (esculpida, rígida con la cabeza; abajo se mezcla con el cuello)
def make_hood():
    HC = head_center()
    K = 1.18  # capucha grande, estilo Pixar
    W, R_arc = 0.142, 0.227
    zb, zs = 1.48, 1.62  # coordenadas de diseño (cabeza en 1.66)
    nb, na = 16, 34
    rim = []
    for k in range(nb):
        h = k / nb
        rim.append((-(0.112 + 0.030 * math.sin(h * math.pi / 2)), zb + (zs - zb) * h))
    amax = math.acos((R_arc - W) / R_arc)
    for k in range(na + 1):
        al = amax * k / na
        rim.append(((R_arc - W) - R_arc * math.cos(al), zs + R_arc * math.sin(al)))
    rim = np.array(rim); rim[-1, 0] = 0.0
    hf = (rim[:, 1] - zb) / (rim[-1, 1] - zb)
    ry = -0.080 - 0.11 * hf ** 1.3 + 0.035 * (np.abs(rim[:, 0]) / W) ** 2
    O_left = np.stack([rim[:, 0], ry, rim[:, 1]], 1)
    seam_pts = [(0, -0.20, 1.83), (0, -0.13, 1.93), (0, -0.045, 2.02), (0, 0.06, 2.10),
                (0, 0.11, 1.95), (0, 0.175, 1.78), (0, 0.18, 1.60), (0, 0.15, 1.46)]
    seam = catmull(seam_pts, 40)
    def arcfrac(Pp):
        d = np.linalg.norm(np.diff(Pp, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(d)])
        return s / s[-1]
    qr = arcfrac(O_left); qs = arcfrac(seam)
    Sx = np.stack([np.interp(1 - qr, qs, seam[:, k]) for k in range(3)], 1)
    O_full = np.vstack([O_left, (O_left[:-1] * [-1, 1, 1])[::-1]])
    S_full = np.vstack([Sx, Sx[:-1][::-1]])
    q_full = np.concatenate([qr, qr[:-1][::-1]])
    nrim = len(O_full)
    NT = 30
    Hc = np.array([0, 0.03, 1.69])
    fspan = np.clip(np.linalg.norm(O_full - S_full, axis=1) / 0.06, 0, 1)
    P = np.zeros((NT, nrim, 3))
    for j in range(NT):
        t = j / (NT - 1)
        base = (1 - t) * O_full + t * S_full
        v = base - Hc
        dist = np.linalg.norm(v, axis=1)
        n = v / dist[:, None]
        w = math.sin(math.pi * t) ** 0.7
        Rt = 0.180 + 0.022 * (1 - q_full)
        r = dist + fspan * (w * np.maximum(0, Rt - dist) + 0.014 * math.sin(math.pi * t))
        p = Hc + n * r[:, None]
        fold = fspan * 0.010 * np.sin(2 * math.pi * (3.0 * t + 2.0 * q_full)) * (1 - q_full) ** 1.6 * math.sin(math.pi * t)
        p = p + n * fold[:, None]
        zr = np.clip((p[:, 2] - 1.64) / 0.44, 0, 1)
        p[:, 0] *= 1 - 0.55 * zr ** 1.1
        p[:, 1] = 0.03 + (p[:, 1] - 0.03) * (1 - 0.30 * zr ** 1.1)
        # abajo se abre un poco sobre los hombros
        lo = np.clip((1.56 - p[:, 2]) / 0.10, 0, 1)
        p[:, 0] *= 1 + 0.35 * lo
        p[:, 1] = 0.03 + (p[:, 1] - 0.03) * (1 + 0.15 * lo)
        P[j] = p
    P = (P - np.array([0, 0, 1.66])) * K + np.array(HC)
    ob = mesh_obj('capucha', P.reshape(-1, 3), grid_faces(NT, nrim))
    su, sv, ut = arc_lengths(P)
    set_uv(ob, 'trim', np.stack([sv, su], -1))
    set_uv(ob, 'kind', np.ones((NT * nrim, 2)))
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0015)
    bm.to_mesh(me); bm.free(); me.update()
    flip_if(ob, lambda p: tuple(HC))
    Pz = np.array([v.co.z for v in me.vertices])
    th = smoothstep(ZN - 0.01, ZHD + 0.06, Pz)
    idx = np.arange(len(Pz))
    vgroup(ob, 'Head', idx, th)
    vgroup(ob, 'Neck1', idx, (1 - th) * 0.45)
    vgroup(ob, 'Spine1', idx, (1 - th) * 0.55)
    ob.data.materials.append(M['hood']); ob.data.materials.append(M['hood_in'])
    smooth(ob)
    return ob

# --- cabeza oscura y ojos
def make_head():
    HC = head_center()
    bm = bmesh.new()
    bm_ellipsoid(bm, HC, (0.088, 0.098, 0.112), 28, 14)
    ob = bm_to_obj(bm, 'cara'); ob.data.materials.append(M['face']); smooth(ob); skin_rigid(ob, 'Head')
    bm = bmesh.new()
    for sx in (-1, 1):
        R = Matrix.Rotation(math.radians(-14) * sx, 3, 'Y')
        bm_ellipsoid(bm, HC + Vector((sx * 0.040, -0.090, 0.004)), (0.022, 0.006, 0.0075), 16, 8, M=R)
    eyes = bm_to_obj(bm, 'ojos'); eyes.data.materials.append(M['eye']); skin_rigid(eyes, 'Head')
    return [ob, eyes]

# --- manos (piel) con dedos
def voxelize(ob, size):
    rm = ob.modifiers.new('Remesh', 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = size; rm.adaptivity = 0.0
    sm = ob.modifiers.new('Suave', 'SMOOTH'); sm.factor = 0.6; sm.iterations = 10
    deps = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(deps))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me2
    for m in old.materials:
        me2.materials.append(m)
    bpy.data.meshes.remove(old)
    smooth(ob)

def make_hand(side, sgn):
    bm = bmesh.new()
    def blob(c, r, s=12, rr=6):
        bm_ellipsoid(bm, c, r, s, rr)
    def segm(a, b, ra, rb):
        bm_capsule(bm, a, b, ra, rb, seg=12, caps=False)
    blob((0.048, 0.0, 0.0), (0.052, 0.042, 0.017), 24, 12)
    blob((0.0, 0.0, 0.0), (0.030, 0.031, 0.020), 16, 8)
    specs = [(0.028, (0.036, 0.024, 0.020), 0.0092, 6), (0.0098, (0.041, 0.027, 0.021), 0.0096, 1),
             (-0.0092, (0.039, 0.026, 0.020), 0.0094, -4), (-0.0265, (0.031, 0.020, 0.017), 0.0084, -9)]
    for (yy, Ls, rad, spread) in specs:
        p = Vector((0.094, yy, 0.002)); ang = 0.0; sp = math.radians(spread); r = rad
        for Lk, curl in zip(Ls, (12, 22, 20)):
            ang += math.radians(curl)
            d = Vector((math.cos(ang) * math.cos(sp), math.sin(sp), -math.sin(ang)))
            q = p + d * Lk
            blob(p, (r, r, r * 0.92)); segm(p, q, r, r * 0.92)
            p = q; r *= 0.92
        blob(p, (r, r, r * 0.9))
    t0 = Vector((0.022, 0.032, -0.010)); t1 = t0 + Vector((0.026, 0.022, -0.012)); t2 = t1 + Vector((0.025, 0.008, -0.009))
    for a_, b_, ra in ((t0, t1, 0.013), (t1, t2, 0.011)):
        blob(a_, (ra, ra, ra)); segm(a_, b_, ra, ra * 0.9)
    blob(t2, (0.0098, 0.0098, 0.0092))
    # orientación en el reposo: dedos siguiendo el hueso de la mano, dorso hacia afuera, pulgar hacia adelante
    Wr = J[side + 'Hand']
    fwd = (JT[side + 'HandIndex1'] - Wr).normalized()
    back = Vector((sgn, 0, 0)); back = (back - fwd * back.dot(fwd)).normalized()
    sidev = back.cross(fwd)
    Rm = Matrix((fwd, sidev, back)).transposed().to_4x4()
    if sidev.dot(Vector((0, -1, 0))) < 0:
        Rm = Rm @ Matrix.Diagonal((1, -1, 1, 1))
    Mw = Matrix.Translation(Wr + fwd * 0.012) @ Rm @ Matrix.Diagonal((1.08, 1.08, 1.08, 1))
    bmesh.ops.transform(bm, matrix=Mw, verts=bm.verts[:])
    if Mw.determinant() < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    ob = bm_to_obj(bm, 'mano_' + side)
    ob.data.materials.append(M['skin'])
    voxelize(ob, 0.0022)
    me = ob.data
    P = np.array([v.co for v in me.vertices])
    t = smoothstep(-0.005, 0.03, (P - np.array(Wr)) @ np.array(fwd))
    idx = np.arange(len(P))
    vgroup(ob, side + 'Hand', idx, t); vgroup(ob, side + 'ForeArm', idx, 1 - t)
    sb = ob.modifiers.new('Sub', 'SUBSURF'); sb.levels = 1; sb.render_levels = 1
    return ob

# --- pies descalzos
def make_foot(side, sgn):
    bm = bmesh.new()
    ank, toe, toe_end = J[side + 'Foot'], J[side + 'ToeBase'], JT[side + 'ToeBase']
    sole = F0 + 0.004
    fwd = (toe_end - ank); fwd.z = 0; fwd.normalize()
    lat = Vector((fwd.y, -fwd.x, 0)) * -sgn  # hacia afuera
    heel = Vector((ank.x, ank.y, sole)) - fwd * 0.045
    tip = Vector((toe_end.x, toe_end.y, sole)) + fwd * 0.01
    L = (tip - heel).length
    Rm = Matrix((fwd, Vector((0, 0, 1)).cross(fwd), Vector((0, 0, 1)))).transposed()
    c = heel + fwd * (L * 0.45) + Vector((0, 0, 0.030))
    bm_ellipsoid(bm, c, (L * 0.47, 0.043, 0.032), 24, 12, M=Rm)
    bm_ellipsoid(bm, heel + fwd * 0.045 + Vector((0, 0, 0.05)), (0.040, 0.036, 0.055), 16, 8, M=Rm)  # tobillo
    # dedos
    base = Vector((toe.x, toe.y, sole)) + fwd * 0.035
    for k, off in enumerate((-0.026, -0.011, 0.002, 0.014, 0.025)):
        rr = (0.013, 0.0095, 0.009, 0.0085, 0.008)[k]
        pos = base + fwd * (0.012 - abs(off) * 0.55) + Rm.col[1] * (off * -sgn) + Vector((0, 0, rr * 0.9))
        bm_ellipsoid(bm, pos, (rr * 1.5, rr, rr * 0.9), 12, 6, M=Rm)
    # canilla (por si la túnica se levanta)
    bm_capsule(bm, J[side + 'Leg'] + (ank - J[side + 'Leg']) * 0.25, ank + Vector((0, 0, 0.02)), 0.042, 0.033, seg=16)
    ob = bm_to_obj(bm, 'pie_' + side)
    ob.data.materials.append(M['skin'])
    voxelize(ob, 0.003)
    P = np.array([v.co for v in ob.data.vertices])
    t = smoothstep(-0.01, 0.025, (P - np.array(toe)) @ np.array(fwd))
    up = smoothstep(ank.z + 0.0, ank.z + 0.06, P[:, 2])
    idx = np.arange(len(P))
    vgroup(ob, side + 'ToeBase', idx, t * (1 - up)); vgroup(ob, side + 'Foot', idx, (1 - t) * (1 - up)); vgroup(ob, side + 'Leg', idx, up)
    sb = ob.modifiers.new('Sub', 'SUBSURF'); sb.levels = 1; sb.render_levels = 1
    return ob

# --- faja de soga y medallón
def make_rope():
    bm = bmesh.new()
    zc = ZH + 0.085
    rx, ry, yc = 0.158, 0.121, (YC + J['Hips'].y) / 2
    n = 160
    for k, dz in enumerate((-0.033, -0.011, 0.011, 0.033)):
        prev = None
        ring = []
        for i in range(n):
            a = 2 * math.pi * i / n
            ring.append(Vector((rx * (1 + 0.012 * k) * math.sin(a), yc - ry * (1 + 0.012 * k) * math.cos(a), zc + dz)))
        # tubo trenzado: radio con ondas en hélice
        nu = 10
        verts = []
        for i in range(n):
            p = ring[i]; tng = (ring[(i + 1) % n] - ring[i - 1]).normalized()
            out = Vector((p.x, p.y - yc, 0)).normalized()
            upv = tng.cross(out).normalized()
            for j in range(nu):
                th = 2 * math.pi * j / nu
                rr = 0.0115 * (1 + 0.16 * math.sin(3 * th + i * 0.9 + k))
                verts.append(bm.verts.new(p + (out * math.cos(th) + upv * math.sin(th)) * rr))
        for i in range(n):
            for j in range(nu):
                a1 = verts[i * nu + j]; a2 = verts[i * nu + (j + 1) % nu]
                b1 = verts[((i + 1) % n) * nu + j]; b2 = verts[((i + 1) % n) * nu + (j + 1) % nu]
                bm.faces.new((a1, a2, b2, b1))
    ob = bm_to_obj(bm, 'soga')
    flip_if(ob, lambda p: (0, yc, p.center.z))
    ob.data.materials.append(M['rope']); smooth(ob)
    skin_prox(ob, ['Hips', 'LowerBack', 'Spine'])
    return ob

def make_medallion():
    bm = bmesh.new()
    zc = ZSH - 0.035
    yfront = YC - 0.130 - 0.022
    # rombo con bisel: dos marcos y punta arriba/abajo
    def diamond(rw, rh, depth, y):
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=rw, radius2=rw * 0.92, depth=depth)
        m = Matrix.Translation((0, y, zc)) @ Matrix.Rotation(math.pi / 2, 4, 'X') @ Matrix.Diagonal((1, rh / rw, 1, 1))
        bmesh.ops.transform(bm, matrix=m, verts=res['verts'])
    diamond(0.046, 0.058, 0.010, yfront + 0.004)
    diamond(0.034, 0.044, 0.012, yfront - 0.002)
    ob = bm_to_obj(bm, 'medallon')
    ob.data.materials.append(M['gold'])
    bev = ob.modifiers.new('Bisel', 'BEVEL'); bev.width = 0.003; bev.segments = 2
    bm = bmesh.new()
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.026, radius2=0.012, depth=0.016)
    m = Matrix.Translation((0, yfront - 0.012, zc)) @ Matrix.Rotation(math.pi / 2, 4, 'X') @ Matrix.Diagonal((1, 1.28, 1, 1))
    bmesh.ops.transform(bm, matrix=m, verts=res['verts'])
    gem = bm_to_obj(bm, 'gema'); gem.data.materials.append(M['gem'])
    for o in (ob, gem):
        skin_rigid(o, 'Spine1')
    return [ob, gem]

def make_cape():
    NU, NV = 96, 22
    keys = [(ZN + 0.025, 0.080, 0.078, YC + 0.01, 0.34), (ZSH + 0.045, 0.165, 0.135, YC + 0.005, 0.40),
            (ZSH - 0.02, 0.235, 0.180, YC, 0.46), (ZSH - 0.085, 0.265, 0.205, YC, 0.50)]
    P = ring_rows(keys, NV, NU, open_angle=True)
    a = np.arctan2(P[..., 0], -(P[..., 1] - YC))  # 0 adelante, ±pi atrás
    back = np.clip(-np.cos(a), 0, 1) ** 2
    t = np.linspace(0, 1, NV)[:, None] ** 1.5
    P[..., 2] -= 0.17 * back * t
    ob = mesh_obj('esclavina', P.reshape(-1, 3), grid_faces(NV, NU))
    flip_if(ob, lambda p: (0, YC, p.center.z))
    su, sv, ut = arc_lengths(P)
    dfront = np.minimum(su, ut[:, None] - su)
    dhem = sv[-1][None, :] - sv
    set_uv(ob, 'trim', np.stack([np.minimum(dfront, dhem), np.where(dfront < dhem, sv, su)], -1))
    set_uv(ob, 'kind', np.ones((NV * NU, 2)))
    z = P[..., 2]
    pw = smoothstep(ZSH + 0.005, ZSH + 0.05, z)
    vgroup(ob, 'pin', np.arange(pw.size), pw.ravel())
    skin_prox(ob, ['Spine1', 'Neck', 'Neck1', 'LeftShoulder', 'RightShoulder', 'LeftArm', 'RightArm'])
    ob.data.materials.append(M['hood']); ob.data.materials.append(M['coat_in'])
    smooth(ob)
    return ob

log('modelando')
ROBE_OB = make_robe()
COAT = make_coat()
SLEEVES = [make_sleeve('Left', 1), make_sleeve('Right', -1)]
HOOD = make_hood()
HEADS = make_head()
HANDS = [make_hand('Left', 1), make_hand('Right', -1)]
FEET = [make_foot('Left', 1), make_foot('Right', -1)]
ROPE = make_rope()
MEDS = make_medallion()
CAPE = make_cape()
CLOTH = [ROBE_OB, COAT] + SLEEVES + [CAPE]
SKINNED = [HOOD, ROPE] + HEADS + HANDS + FEET + MEDS
for ob in SKINNED:
    add_armature(ob)

# ======================================================================= simulación o caché
if args.stage == 'sim':
    for ob in CLOTH:
        add_armature(ob)
    cloth_mod(ROBE_OB, 'robe', mass=0.25, tension=25, bending=2.0, shear=8, air=1.5)
    cloth_mod(COAT, 'coat', mass=0.30, tension=25, bending=1.6, shear=8, air=1.5, quality=7, self_col=True, dist=0.007)
    for sl in SLEEVES:
        cloth_mod(sl, 'sleeve', mass=0.18, tension=15, bending=0.5, air=1.2, self_col=True, pin_stiff=0.6)
    cloth_mod(CAPE, 'cape', mass=0.2, tension=20, bending=2.0, air=1.2, dist=0.008)
    # la capa choca también con la túnica (ya simulada) y con la soga
    for o in (ROBE_OB, ROPE):
        cm = o.modifiers.new('Collision', 'COLLISION')
        o.collision.thickness_outer = 0.006; o.collision.cloth_friction = 0.5
        CC['coat'].objects.link(o)
    scene.frame_start = 1; scene.frame_end = LAST
    bpy.context.view_layer.update()
    cache = {ob.name: np.zeros((NCLIP, len(ob.data.vertices), 3), np.float32) for ob in CLOTH}
    t = time.time()
    for f in range(1, LAST + 1):
        scene.frame_set(f)
        if f >= PRE:
            deps = bpy.context.evaluated_depsgraph_get()
            for ob in CLOTH:
                ev = ob.evaluated_get(deps)
                m = ev.to_mesh()
                arr = np.empty(len(m.vertices) * 3, np.float32)
                m.vertices.foreach_get('co', arr)
                cache[ob.name][f - PRE] = arr.reshape(-1, 3)
                ev.to_mesh_clear()
        if f % 15 == 0:
            log('  tela cuadro %d/%d (%.1fs)' % (f, LAST, time.time() - t))
    np.savez_compressed(os.path.join(OUT, 'sim.npz'), **cache)
    log('simulación guardada (%.1fs)' % (time.time() - t))
    for ob in CLOTH:
        for m in list(ob.modifiers):
            ob.modifiers.remove(m)
    CACHE = cache
else:
    CACHE = dict(np.load(args.cache))
    log('caché cargada', {k: v.shape for k, v in CACHE.items()})

def apply_cache(f):
    i = min(max(f - PRE, 0), NCLIP - 1)
    for ob in CLOTH:
        ob.data.vertices.foreach_set('co', CACHE[ob.name][i].ravel())
        ob.data.update()

# terminación: grosor + subdivisión
def finish(ob, thick, sub=1):
    so = ob.modifiers.new('Grosor', 'SOLIDIFY')
    so.thickness = thick; so.offset = -1; so.use_rim = True; so.material_offset = 1; so.material_offset_rim = 0
    sb = ob.modifiers.new('Sub', 'SUBSURF'); sb.levels = 1; sb.render_levels = sub
for ob in CLOTH:
    ob.parent = B  # ya está; las posiciones de la caché están en el espacio del objeto
finish(COAT, 0.010, 1); finish(ROBE_OB, 0.006, 1)
for sl in SLEEVES:
    finish(sl, 0.008, 1)
finish(CAPE, 0.009, 1)
finish(HOOD, 0.014, 2)
for o in (ROBE_OB, ROPE):
    for m in list(o.modifiers):
        if m.type == 'COLLISION':
            o.modifiers.remove(m)
ROPE.modifiers.new('Sub', 'SUBSURF').render_levels = 1
for o in (COL_BODY, COL_ARMS, COL_FAT, COL_SH):
    o.hide_viewport = True

# ======================================================================= entorno: cielo estrellado, piso, luces
def sky():
    world = bpy.data.worlds.new('noche'); scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = N(nt, 'ShaderNodeOutputWorld')
    bg = N(nt, 'ShaderNodeBackground')
    tc = N(nt, 'ShaderNodeTexCoord')
    sxyz = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Generated'], sxyz.inputs[0])
    # degradé vertical del cielo (Generated en el mundo = dirección)
    tcw = N(nt, 'ShaderNodeTexCoord')
    sep = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(tcw.outputs['Generated'], sep.inputs[0])
    ramp = N(nt, 'ShaderNodeValToRGB')
    el = ramp.color_ramp.elements
    el[0].position = 0.45; el[0].color = (0.020, 0.035, 0.075, 1)
    el[1].position = 0.75; el[1].color = (0.004, 0.006, 0.016, 1)
    nt.links.new(sep.outputs['Z'], ramp.inputs['Fac'])
    vo = N(nt, 'ShaderNodeTexVoronoi'); vo.inputs['Scale'].default_value = 90.0
    nt.links.new(tcw.outputs['Generated'], vo.inputs['Vector'])
    mr = N(nt, 'ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.11; mr.inputs['From Max'].default_value = 0.0
    nt.links.new(vo.outputs['Distance'], mr.inputs['Value'])
    sc = N(nt, 'ShaderNodeSeparateColor'); nt.links.new(vo.outputs['Color'], sc.inputs[0])
    keep = math_(nt, 'GREATER_THAN', sc.outputs[0], val=0.88)
    star = math_(nt, 'MULTIPLY', math_(nt, 'MULTIPLY', mr.outputs[0], keep), math_(nt, 'POWER', sc.outputs[1], val=3.0))
    star = math_(nt, 'MULTIPLY', star, val=6.0)
    above = math_(nt, 'GREATER_THAN', sep.outputs['Z'], val=-0.15)
    star = math_(nt, 'MULTIPLY', star, above)
    add = N(nt, 'ShaderNodeMixRGB', blend_type='ADD'); add.inputs['Fac'].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], add.inputs['Color1'])
    cs = N(nt, 'ShaderNodeCombineXYZ')
    for k in range(3):
        nt.links.new(star, cs.inputs[k])
    nt.links.new(cs.outputs[0], add.inputs['Color2'])
    nt.links.new(add.outputs[0], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = 1.0
    if LOOK == 'anime':
        lp = N(nt, 'ShaderNodeLightPath'); bk = N(nt, 'ShaderNodeBackground'); bk.inputs['Color'].default_value = (0, 0, 0, 1)
        mx = N(nt, 'ShaderNodeMixShader')
        nt.links.new(lp.outputs['Is Camera Ray'], mx.inputs['Fac'])
        nt.links.new(bk.outputs[0], mx.inputs[1]); nt.links.new(bg.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs['Surface'])
    else:
        nt.links.new(bg.outputs[0], out.inputs['Surface'])
sky()

def floor_material():
    mat = bpy.data.materials.new('piso')
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    geo = N(nt, 'ShaderNodeNewGeometry')
    sx = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sx.inputs[0])
    cx = N(nt, 'ShaderNodeCombineXYZ')
    hx, hy = B.location.x, B.location.y
    dx = math_(nt, 'SUBTRACT', sx.outputs['X'], val=hx); dy = math_(nt, 'SUBTRACT', sx.outputs['Y'], val=hy)
    nt.links.new(dx, cx.inputs['X']); nt.links.new(dy, cx.inputs['Y'])
    ln = N(nt, 'ShaderNodeVectorMath', operation='LENGTH'); nt.links.new(cx.outputs[0], ln.inputs[0])
    bs = fabric_bsdf(nt, (0.012, 0.015, 0.028), rough=0.9, sheen=0.0, bump=0.0)
    fade = N(nt, 'ShaderNodeMapRange'); fade.inputs['From Min'].default_value = 1.6; fade.inputs['From Max'].default_value = 5.0
    fade.interpolation_type = 'SMOOTHSTEP'
    nt.links.new(ln.outputs['Value'], fade.inputs['Value'])
    tr = N(nt, 'ShaderNodeBsdfTransparent')
    mx = N(nt, 'ShaderNodeMixShader')
    nt.links.new(fade.outputs[0], mx.inputs['Fac'])
    nt.links.new(bs.outputs[0], mx.inputs[1]); nt.links.new(tr.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    return mat
FLOOR_OB.data.materials.append(floor_material())

def light(name, typ, loc_local, energy, color, size=1.0, target_local=(0, 0, 0.0)):
    ld = bpy.data.lights.new(name, typ); ld.energy = energy; ld.color = color
    if typ == 'AREA':
        ld.size = size
    else:
        ld.shadow_soft_size = size
    lo = bpy.data.objects.new(name, ld); scene.collection.objects.link(lo)
    Mw = B.matrix_world
    lo.location = Mw @ Vector(loc_local)
    lo.rotation_euler = ((Mw @ Vector(target_local)) - lo.location).to_track_quat('-Z', 'Y').to_euler()
    return lo

bpy.context.view_layer.update()
light('clave', 'AREA', (-1.6, -2.6, 1.6), 380, (0.82, 0.86, 1.0), size=2.2, target_local=(0, 0, 0.2))
light('relleno', 'AREA', (2.4, -2.2, 0.4), 50, (0.45, 0.55, 0.95), size=2.5, target_local=(0, 0, 0.0))
light('borde_izq', 'AREA', (-1.5, 1.8, 1.4), 650, (0.35, 0.65, 1.0), size=1.5, target_local=(0, 0, 0.4))
light('borde_der', 'AREA', (1.6, 1.7, 1.2), 550, (0.45, 0.55, 1.0), size=1.5, target_local=(0, 0, 0.3))
light('techo', 'AREA', (0, -0.4, 2.6), 40, (0.6, 0.7, 1.0), size=1.2, target_local=(0, 0, 0.5))
light('ojos_luz', 'POINT', tuple(head_center() + Vector((0, -0.13, 0.0))), 0.6, (0.3, 1.0, 0.9), size=0.03)

# ======================================================================= look anime: sol duro + círculo de runas pintado
def stroke(m, x0, y0, x1, y1, w):
    H, W = m.shape
    pad = w + 2
    xa, xb = int(max(0, min(x0, x1) - pad)), int(min(W, max(x0, x1) + pad + 1))
    ya, yb = int(max(0, min(y0, y1) - pad)), int(min(H, max(y0, y1) + pad + 1))
    if xa >= xb or ya >= yb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
    d = np.hypot(xx - x0 - t * dx, yy - y0 - t * dy)
    m[ya:yb, xa:xb] = np.maximum(m[ya:yb, xa:xb], np.clip(w * 0.5 + 0.75 - d, 0, 1))

def box_blur(a, r):
    for ax in (0, 1):
        c = np.cumsum(np.pad(a, [(r + 1, r) if k == ax else (0, 0) for k in range(2)], mode='edge'), axis=ax)
        a = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return a

def rune_circle_image(S=2048, seed=3):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    c = S / 2
    r = np.hypot(xx - c, yy - c) / (S / 2)  # 0 centro, 1 borde
    m = np.zeros((S, S), np.float32)
    px = S / 2
    def ring(R, w):
        np.maximum(m, np.clip(w * 0.5 + 0.75 - np.abs(r - R) * px, 0, 1), out=m)
    for R, w in ((0.96, 7), (0.93, 3), (0.74, 5), (0.71, 2.5), (0.50, 3), (0.47, 2), (0.22, 3)):
        ring(R, w)
    rng = random.Random(seed)
    # runas entre 0.75 y 0.92
    n = 28
    for k in range(n):
        g = rng.choice(RUNES)
        th = 2 * math.pi * k / n
        rad = np.array([math.cos(th), math.sin(th)]); tan = np.array([-math.sin(th), math.cos(th)])
        hgt, wid = 0.13 * px, 0.085 * px
        for (u0, v0, u1, v1) in g:
            p0 = c + rad * (0.765 * px + v0 * hgt) + tan * ((u0 - 0.5) * wid)
            p1 = c + rad * (0.765 * px + v1 * hgt) + tan * ((u1 - 0.5) * wid)
            stroke(m, p0[0], p0[1], p1[0], p1[1], 6)
    # marcas radiales entre los anillos medios
    for k in range(72):
        th = 2 * math.pi * k / 72
        L = 0.06 if k % 6 == 0 else 0.03
        a0 = c + np.array([math.cos(th), math.sin(th)]) * 0.50 * px
        a1 = c + np.array([math.cos(th), math.sin(th)]) * (0.50 + L) * px
        stroke(m, a0[0], a0[1], a1[0], a1[1], 3 if k % 6 else 5)
    # puntos sobre el anillo interior y arcos de órbita
    for k in range(12):
        th = 2 * math.pi * (k + 0.5) / 12
        q = c + np.array([math.cos(th), math.sin(th)]) * 0.355 * px
        d = np.hypot(xx - q[0], yy - q[1])
        np.maximum(m, np.clip(9 - d, 0, 1), out=m)
    glow = box_blur(box_blur(box_blur(m, 10), 10), 10)
    fill = np.clip(1.0 - r, 0, 1) ** 0.6 * (r < 0.96) * 0.10
    alpha = np.clip(m + glow * 1.4 + fill, 0, 1)
    col = np.array([0.25, 0.85, 1.0], np.float32)
    rgb = (m[..., None] * 1.6 + glow[..., None] * 1.2) * col + fill[..., None] * np.array([0.10, 0.12, 0.45], np.float32)
    px4 = np.concatenate([rgb, alpha[..., None]], -1).astype(np.float32)
    img = bpy.data.images.new('circulo_runas', S, S, alpha=True, float_buffer=True)
    img.pixels.foreach_set(px4.ravel())
    img.update(); img.pack()
    return img

def decal_mat(name, img, strength):
    mat = bpy.data.materials.new(name)
    mat.blend_method = 'BLEND'
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    tx = N(nt, 'ShaderNodeTexImage', image=img, interpolation='Cubic')
    em = N(nt, 'ShaderNodeEmission'); em.inputs['Strength'].default_value = strength
    nt.links.new(tx.outputs['Color'], em.inputs['Color'])
    tr = N(nt, 'ShaderNodeBsdfTransparent')
    mx = N(nt, 'ShaderNodeMixShader')
    nt.links.new(tx.outputs['Alpha'], mx.inputs['Fac'])
    nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    return mat

def quad(name, size, mat):
    h = size / 2
    ob = mesh_obj(name, [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [(0, 1, 2, 3)], C_ENV)
    set_uv(ob, 'UVMap', [(0, 0), (1, 0), (1, 1), (0, 1)])
    ob.data.materials.append(mat)
    ob.visible_diffuse = False; ob.visible_glossy = False; ob.visible_shadow = False
    return ob

BG_PLANE = None
if LOOK == 'anime':
    for o in list(scene.objects):
        if o.type == 'LIGHT':
            o.hide_render = True
    sd = bpy.data.lights.new('sol', 'SUN'); sd.energy = 1.0; sd.angle = math.radians(1.0); sd.color = (1.0, 0.97, 0.92)
    so = bpy.data.objects.new('sol', sd); scene.collection.objects.link(so)
    so.rotation_euler = (B.matrix_world.to_3x3() @ Vector((-1.4, -2.2, 2.0))).to_track_quat('Z', 'Y').to_euler()
    FLOOR_OB.hide_render = True
    t = time.time()
    IMG_CIRC = rune_circle_image()
    log('círculo de runas %.1fs' % (time.time() - t))
    BG_PLANE = quad('fondo_circulo', 2.7, decal_mat('fondo_circulo', IMG_CIRC, 0.7))
    ground = quad('piso_circulo', 2.0, decal_mat('piso_circulo', IMG_CIRC, 0.6))
    ground.location = (B.location.x, B.location.y, FLOOR + 0.004)

def place_bg(camo):
    if BG_PLANE is None:
        return
    bpy.context.view_layer.update()
    ctr = B.matrix_world @ Vector((0, 0, ZHD - 0.20))
    d = (ctr - camo.location).normalized()
    BG_PLANE.location = ctr + d * 2.3
    BG_PLANE.rotation_euler = (-d).to_track_quat('Z', 'Y').to_euler()

# ======================================================================= cámaras (en el marco del personaje)
def cam(name, az_deg, dist, height, target_z, lens):
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.sensor_fit = 'VERTICAL'; cd.sensor_height = 36.0
    co = bpy.data.objects.new(name, cd); scene.collection.objects.link(co)
    az = math.radians(az_deg)
    Mw = B.matrix_world
    tgt = Mw @ Vector((0, 0, target_z))
    co.location = Mw @ Vector((dist * math.sin(az), -dist * math.cos(az), height))
    co.rotation_euler = (tgt - co.location).to_track_quat('-Z', 'Y').to_euler()
    return co

zmid = (F0 + (ZHD + 0.55)) / 2
CAM_VIDEO = cam('cam_video', 18, 4.0, zmid - 0.05, zmid, 50)
CAMS_STILL = [
    (cam('cam_frente', 12, 3.8, zmid - 0.05, zmid, 50), PRE + 2, 'foto1_frente.png'),
    (cam('cam_tres_cuartos', 35, 3.9, zmid, zmid, 50), PRE + 75, 'foto2_movimiento.png'),
    (cam('cam_capucha', 20, 1.9, ZHD - 0.05, ZHD - 0.02, 75), PRE + 120, 'foto3_capucha.png'),
    (cam('cam_costado', 140, 3.9, zmid, zmid, 50), PRE + 160, 'foto4_costado.png'),
]

# ======================================================================= render
scene.render.engine = 'CYCLES'
cy = scene.cycles
cy.device = 'CPU'
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.04
cy.use_denoising = True
cy.denoiser = 'OPENIMAGEDENOISE'
cy.max_bounces = 4; cy.diffuse_bounces = 2; cy.glossy_bounces = 2; cy.transmission_bounces = 0; cy.volume_bounces = 0
cy.transparent_max_bounces = 4
cy.caustics_reflective = False; cy.caustics_refractive = False
cy.blur_glossy = 1.0
scene.render.use_persistent_data = True
scene.render.resolution_x = 1080; scene.render.resolution_y = 1920; scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGB'
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'None'
scene.view_settings.exposure = -0.35

scene.use_nodes = True
ct = scene.node_tree
for n in list(ct.nodes):
    ct.nodes.remove(n)
rl = ct.nodes.new('CompositorNodeRLayers')
gl = ct.nodes.new('CompositorNodeGlare'); gl.glare_type = 'FOG_GLOW'; gl.quality = 'HIGH'; gl.size = 7; gl.threshold = 1.0; gl.mix = 0.0
comp = ct.nodes.new('CompositorNodeComposite')
VIEWER = None
if LOOK == 'pixar':
    ct.links.new(rl.outputs['Image'], gl.inputs['Image'])
else:
    vl = scene.view_layers[0]
    vl.use_pass_diffuse_direct = True; vl.use_pass_diffuse_color = True; vl.use_pass_emit = True; vl.use_pass_environment = True
    vl.cycles.denoising_store_passes = True
    cy.use_denoising = False
    scene.view_settings.view_transform = 'Standard'; scene.view_settings.exposure = 0.0
    # tinta: Freestyle como pasada aparte
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = 'ABSOLUTE'; scene.render.line_thickness = 1.0
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    fs.as_render_pass = True
    fs.crease_angle = math.radians(128)
    ls = fs.linesets.active if len(fs.linesets) else fs.linesets.new('tinta')
    ls.select_by_visibility = True; ls.visibility = 'VISIBLE'
    ls.select_by_edge_types = True
    ls.select_silhouette = True; ls.select_border = True; ls.select_crease = True; ls.select_contour = True
    ls.select_by_collection = True; ls.collection = C_CHAR; ls.collection_negation = 'INCLUSIVE'
    lst = ls.linestyle or bpy.data.linestyles.new('tinta')
    ls.linestyle = lst
    lst.color = (0.004, 0.005, 0.018); lst.thickness = 4.2; lst.alpha = 1.0
    try:
        tm = lst.thickness_modifiers.new('afinar', 'ALONG_STROKE'); tm.mapping = 'CURVE'; tm.blend = 'MULTIPLY'
        cm = tm.curve; pts = cm.curves[0].points
        pts[0].location = (0, 0.55); pts[1].location = (1, 0.55); pts.new(0.5, 1.0); cm.update()
    except Exception as e:
        log('modificador de grosor no disponible:', e)
    dn = ct.nodes.new('CompositorNodeDenoise')
    ct.links.new(rl.outputs['DiffDir'], dn.inputs['Image'])
    ct.links.new(rl.outputs['Denoising Normal'], dn.inputs['Normal'])
    bw = ct.nodes.new('CompositorNodeRGBToBW'); ct.links.new(dn.outputs[0], bw.inputs[0])
    ramp = ct.nodes.new('CompositorNodeValToRGB')
    cr = ramp.color_ramp; cr.interpolation = 'CONSTANT'
    cr.elements[0].position = 0.0; cr.elements[0].color = (0.30, 0.27, 0.55, 1)
    cr.elements[1].position = 0.10; cr.elements[1].color = (0.68, 0.68, 0.86, 1)
    e3 = cr.elements.new(0.42); e3.color = (1.0, 1.0, 1.0, 1)
    ct.links.new(bw.outputs[0], ramp.inputs['Fac'])
    mul = ct.nodes.new('CompositorNodeMixRGB'); mul.blend_type = 'MULTIPLY'; mul.inputs['Fac'].default_value = 1.0
    ct.links.new(ramp.outputs['Image'], mul.inputs[1]); ct.links.new(rl.outputs['DiffCol'], mul.inputs[2])
    gain = ct.nodes.new('CompositorNodeMixRGB'); gain.blend_type = 'MULTIPLY'; gain.inputs['Fac'].default_value = 1.0
    ct.links.new(mul.outputs[0], gain.inputs[1]); gain.inputs[2].default_value = (1.7, 1.7, 1.7, 1)
    a1 = ct.nodes.new('CompositorNodeMixRGB'); a1.blend_type = 'ADD'; a1.inputs['Fac'].default_value = 1.0
    ct.links.new(gain.outputs[0], a1.inputs[1]); ct.links.new(rl.outputs['Emit'], a1.inputs[2])
    a2 = ct.nodes.new('CompositorNodeMixRGB'); a2.blend_type = 'ADD'; a2.inputs['Fac'].default_value = 1.0
    ct.links.new(a1.outputs[0], a2.inputs[1]); ct.links.new(rl.outputs['Env'], a2.inputs[2])
    ao = ct.nodes.new('CompositorNodeAlphaOver')
    ct.links.new(a2.outputs[0], ao.inputs[1]); ct.links.new(rl.outputs['Freestyle'], ao.inputs[2])
    ct.links.new(ao.outputs[0], gl.inputs['Image'])
    VIEWER = ct.nodes.new('CompositorNodeViewer'); ct.links.new(bw.outputs[0], VIEWER.inputs[0])
ct.links.new(gl.outputs['Image'], comp.inputs['Image'])

def render_frame(f, camo, path, samples):
    scene.camera = camo
    place_bg(camo)
    cy.samples = samples
    scene.frame_set(f)
    apply_cache(f)
    bpy.context.view_layer.update()
    scene.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    log('render cuadro %d en %.1fs -> %s' % (f, time.time() - t, os.path.basename(path)))
    if VIEWER is not None and args.stage != 'render':
        try:
            v = np.array(bpy.data.images['Viewer Node'].pixels[:], np.float32).reshape(-1, 4)[:, 0]
            v = v[v > 1e-4]
            log('  luz difusa (percentiles 10/50/90/99):', np.round(np.percentile(v, [10, 50, 90, 99]), 3))
        except Exception as e:
            log('  sin datos del visor:', e)

if args.stage == 'sim':
    # una foto rápida de control (media resolución)
    scene.render.resolution_percentage = 50
    render_frame(PRE + 60, CAMS_STILL[0][0], os.path.join(OUT, 'control.png'), 24)
    scene.render.resolution_percentage = 100
elif args.stage == 'render':
    frames = list(range(PRE, LAST + 1))
    mine = frames[args.chunk::args.nchunks] if False else np.array_split(frames, args.nchunks)[args.chunk]
    os.makedirs(os.path.join(OUT, 'frames'), exist_ok=True)
    for f in mine:
        render_frame(int(f), CAM_VIDEO, os.path.join(OUT, 'frames', 'f_%04d.png' % (int(f) - PRE)), args.samples or 20)
elif args.stage == 'stills':
    for camo, f, name in CAMS_STILL:
        render_frame(f, camo, os.path.join(OUT, name), args.samples or 64)
log('fin')
