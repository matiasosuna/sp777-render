# Encapuchado (Spiritual Practices 777) en Blender, desde cero.
# Corre headless en GitHub Actions:  blender -b --factory-startup --python blender/encapuchado.py -- --out out
#
# Idea: un cuerpo simple invisible (colisionadores) y encima la ropa hecha con simulación de tela real,
# en etapas (cada etapa se congela y pasa a ser colisionador de la siguiente):
#   1. túnica interior (colgada del pecho y de la cintura)
#   2. mangas acampanadas + puntas del fajín
#   3. capa larga abierta adelante (cae sobre hombros, brazos y mangas, se apoya en el piso)
#   4. cuello/capelina (los pliegues que bajan de la capucha a los hombros)
# La capucha es geometría esculpida por código (dos paños unidos por una costura central que termina en punta).
# Look ilustrado: Toon BSDF de Cycles + borde de luz + líneas Freestyle; glow con el compositor.

import bpy, bmesh, math, os, sys, time, random, argparse
import numpy as np
from mathutils import Vector, Matrix, Quaternion

T0 = time.time()
def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='out')
ap.add_argument('--samples', type=int, default=96)
ap.add_argument('--views', default='full,close,side')
ap.add_argument('--scale', type=int, default=100, help='resolution percentage')
ap.add_argument('--simscale', type=float, default=1.0, help='multiplica la cantidad de cuadros simulados')
args = ap.parse_args(argv)
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
random.seed(7)
np.random.seed(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 24

# ---------------------------------------------------------------- colecciones
def new_coll(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or scene.collection).children.link(c)
    return c

C_CHAR = new_coll('personaje')      # lo que lleva contorno Freestyle
C_FX = new_coll('fx')               # ojos, gema, runas, partículas, fondo (sin contorno)
C_COL = new_coll('colisionadores')  # cuerpo invisible

def free_coll(name):
    # colección suelta (no vinculada a la escena) solo para elegir con qué choca cada tela
    return bpy.data.collections.new(name)

CC = {k: free_coll('cc_' + k) for k in ('robe', 'sleeve', 'cloak', 'cowl', 'sash')}

# ---------------------------------------------------------------- utilidades
def link(ob, coll):
    coll.objects.link(ob)
    return ob

def mesh_obj(name, verts, faces, coll):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob

def smooth(ob):
    me = ob.data
    try:
        me.shade_smooth()
    except Exception:
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.update()

def catmull(P, n_per_seg=24):
    """Catmull-Rom centrípeta-ish (uniforme) por una lista de puntos (k, d). Devuelve puntos densos."""
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(n_per_seg):
            t = k / n_per_seg
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return np.array(out)

def resample(dense, n, cols=None):
    """Re-muestrea n puntos equiespaciados por longitud de arco (medida en las columnas cols)."""
    c = dense if cols is None else dense[:, cols]
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, dense[:, k]) for k in range(dense.shape[1])], axis=1)

def grid_faces(nv, nu, closed_u=False):
    f = []
    nuf = nu if closed_u else nu - 1
    for i in range(nv - 1):
        for j in range(nuf):
            j2 = (j + 1) % nu
            f.append((i * nu + j, i * nu + j2, (i + 1) * nu + j2, (i + 1) * nu + j))
    return f

def arc_lengths(P, closed_u=False):
    """P (nv, nu, 3) -> su (arco a lo largo de u en cada fila), sv (arco a lo largo de v en cada columna)."""
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
    uvl.data.foreach_set('uv', np.asarray(uv_vert, np.float32)[vidx].ravel())
    return uvl

def orient(ob, center_fn):
    """Da vuelta las caras si las normales no apuntan hacia afuera (según center_fn(punto) -> punto de referencia)."""
    me = ob.data
    me.update()
    tot = 0.0
    for p in me.polygons:
        c = p.center
        tot += p.normal.dot(c - Vector(center_fn(c)))
    if tot < 0:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
        me.update()

def axis_center(yc=0.0):
    return lambda c: (0.0, yc, c.z)

def add_pin(ob, idx, w=1.0, name='pin'):
    vg = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
    vg.add([int(i) for i in idx], w, 'REPLACE')
    return vg

def set_materials(ob, mats, face_mat_idx=None):
    for m in mats:
        ob.data.materials.append(m)
    if face_mat_idx is not None:
        ob.data.polygons.foreach_set('material_index', list(map(int, face_mat_idx)))

# ---------------------------------------------------------------- texturas de runas
RUNES = [
    [(.3, 0, .3, 1), (.3, .5, .75, .75), (.3, .78, .75, 1.0)],
    [(.25, 0, .25, 1), (.25, 1, .75, .7), (.75, .7, .75, 0)],
    [(.3, 0, .3, 1), (.3, .75, .7, .5), (.7, .5, .3, .25)],
    [(.3, 0, .3, 1), (.3, 1, .7, .75), (.3, .7, .7, .45)],
    [(.3, 0, .3, 1), (.3, 1, .7, .78), (.7, .78, .3, .55), (.3, .55, .72, 0)],
    [(.65, 1, .3, .5), (.3, .5, .65, 0)],
    [(.25, 0, .75, 1), (.25, 1, .75, 0)],
    [(.3, 0, .3, 1), (.3, 1, .7, .78), (.7, .78, .3, .55)],
    [(.25, 0, .25, 1), (.75, 0, .75, 1), (.25, .65, .75, .35)],
    [(.5, 0, .5, 1), (.3, .65, .7, .35)],
    [(.45, .95, .2, .65), (.2, .65, .45, .4), (.55, .6, .8, .35), (.8, .35, .55, .05)],
    [(.5, 0, .5, 1), (.5, 1, .75, .8), (.5, 0, .25, .2)],
    [(.5, 0, .5, 1), (.5, .6, .2, .95), (.5, .6, .8, .95)],
    [(.7, 1, .3, .7), (.3, .7, .7, .3), (.7, .3, .3, 0)],
    [(.5, 0, .5, 1), (.5, 1, .2, .7), (.5, 1, .8, .7)],
    [(.3, 0, .3, 1), (.3, 1, .7, .75), (.7, .75, .3, .5), (.3, .5, .7, .25), (.7, .25, .3, 0)],
    [(.25, 0, .25, 1), (.75, 0, .75, 1), (.25, 1, .75, .6), (.75, 1, .25, .6)],
    [(.5, 1, .8, .65), (.8, .65, .5, .3), (.5, .3, .2, .65), (.2, .65, .5, 1), (.5, .3, .2, 0), (.5, .3, .8, 0)],
    [(.2, 0, .2, 1), (.8, 0, .8, 1), (.2, 1, .8, 0), (.2, 0, .8, 1)],
    [(.3, 0, .3, 1), (.3, .4, .7, .1), (.3, .6, .7, .9)],
]

def raster_segments(H, W, segs, width):
    """Máscara (H, W) con segmentos en coordenadas de píxel (x0, y0, x1, y1)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.zeros((H, W), np.float32)
    for (x0, y0, x1, y1) in segs:
        px, py = xx - x0, yy - y0
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-9
        t = np.clip((px * dx + py * dy) / L2, 0, 1)
        d = np.hypot(px - t * dx, py - t * dy)
        m = np.maximum(m, np.clip(width * 0.5 + 0.75 - d, 0, 1))
    return m

def trim_image(name, vertical):
    """Banda de 128 de ancho x 2048 de largo: dos filetes dorados y runas en el medio. Máscara en escala de grises.
    Horizontal: x = a lo largo, y = 0 en el borde de la tela. Vertical: x = 0 en el borde, y = a lo largo (runas derechas
    cuando 'a lo largo' crece hacia abajo en la tela)."""
    Lpx, Wpx = 2048, 128
    m = np.zeros((Wpx, Lpx), np.float32)  # filas = a través, columnas = a lo largo
    m[int(.05 * Wpx):int(.12 * Wpx), :] = 1
    m[int(.88 * Wpx):int(.95 * Wpx), :] = 1
    cell = int(Wpx * 0.62)
    n = Lpx // cell
    gap = (Lpx - n * cell) / n
    segs = []
    rng = random.Random(len(name) * 31 + (1 if vertical else 0))
    for k in range(n):
        g = rng.choice(RUNES)
        ox = k * (cell + gap) + gap / 2 + cell * 0.15
        oy = Wpx * 0.5 - cell * 0.35
        sz = cell * 0.7
        for (x0, y0, x1, y1) in g:
            if vertical:
                # 'arriba' de la runa = hacia columnas de menor índice (a lo largo decrece = hacia arriba en la tela)
                segs.append((ox + (1 - y0) * sz, oy + x0 * sz, ox + (1 - y1) * sz, oy + x1 * sz))
            else:
                segs.append((ox + x0 * sz, oy + y0 * sz, ox + x1 * sz, oy + y1 * sz))
    m = np.maximum(m, raster_segments(Wpx, Lpx, segs, width=6.5))
    if vertical:
        m = m.T  # filas = a lo largo, columnas = a través
        h, w = Lpx, Wpx
    else:
        h, w = Wpx, Lpx
    img = bpy.data.images.new(name, w, h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'  # antes de cargar píxeles: cambiarlo después recarga y borra la imagen
    px = np.ones((h, w, 4), np.float32)
    px[..., 0] = px[..., 1] = px[..., 2] = m
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    log('textura %s: media %.3f' % (name, float(np.mean(img.pixels[::4][:200000]))))
    return img

IMG_H = trim_image('trim_h', vertical=False)
IMG_V = trim_image('trim_v', vertical=True)
TILE = 16.0  # largo de la imagen = 16 anchos de banda

# ---------------------------------------------------------------- materiales
GOLD = (0.62, 0.40, 0.12)

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

def rgb_scale(nt, col_socket, k):
    m = N(nt, 'ShaderNodeVectorMath', operation='SCALE')
    nt.links.new(col_socket, m.inputs[0])
    m.inputs['Scale'].default_value = k
    return m.outputs[0]

def trim_color(nt, base_socket, gold, kind_uv='kind', trim_uv='trim', line_only=False, band_dark=0.65):
    """Devuelve (color, máscara_oro) mezclando la banda de runas sobre base_socket según los UV 'trim' y 'kind'."""
    uvt = N(nt, 'ShaderNodeUVMap', uv_map=trim_uv)
    uvk = N(nt, 'ShaderNodeUVMap', uv_map=kind_uv)
    sep = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(uvt.outputs['UV'], sep.inputs[0])
    sepk = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(uvk.outputs['UV'], sepk.inputs[0])
    isv = N(nt, 'ShaderNodeMath', operation='GREATER_THAN'); isv.inputs[1].default_value = 0.5
    nt.links.new(sepk.outputs['X'], isv.inputs[0])
    # coordenada 'a través' : B (horizontal) o A (vertical)
    across = N(nt, 'ShaderNodeMix', data_type='FLOAT')
    nt.links.new(isv.outputs[0], across.inputs['Factor'])
    nt.links.new(sep.outputs['Y'], across.inputs[2])
    nt.links.new(sep.outputs['X'], across.inputs[3])
    a_ge = N(nt, 'ShaderNodeMath', operation='GREATER_THAN'); a_ge.inputs[1].default_value = 0.0
    a_le = N(nt, 'ShaderNodeMath', operation='LESS_THAN'); a_le.inputs[1].default_value = 1.0
    nt.links.new(across.outputs[0], a_ge.inputs[0]); nt.links.new(across.outputs[0], a_le.inputs[0])
    band = N(nt, 'ShaderNodeMath', operation='MULTIPLY')
    nt.links.new(a_ge.outputs[0], band.inputs[0]); nt.links.new(a_le.outputs[0], band.inputs[1])
    if line_only:
        # solo un filete fino en el borde (lado del forro)
        l = N(nt, 'ShaderNodeMath', operation='LESS_THAN'); l.inputs[1].default_value = 0.14
        nt.links.new(across.outputs[0], l.inputs[0])
        g = N(nt, 'ShaderNodeMath', operation='MULTIPLY')
        nt.links.new(l.outputs[0], g.inputs[0]); nt.links.new(a_ge.outputs[0], g.inputs[1])
        goldmask = g.outputs[0]
        bandmask = goldmask
    else:
        th = N(nt, 'ShaderNodeTexImage', image=IMG_H, extension='REPEAT', interpolation='Linear')
        tv = N(nt, 'ShaderNodeTexImage', image=IMG_V, extension='REPEAT', interpolation='Linear')
        nt.links.new(uvt.outputs['UV'], th.inputs['Vector'])
        nt.links.new(uvt.outputs['UV'], tv.inputs['Vector'])
        sel = N(nt, 'ShaderNodeMix', data_type='FLOAT')
        nt.links.new(isv.outputs[0], sel.inputs['Factor'])
        nt.links.new(th.outputs['Color'], sel.inputs[2])
        nt.links.new(tv.outputs['Color'], sel.inputs[3])
        g = N(nt, 'ShaderNodeMath', operation='MULTIPLY')
        nt.links.new(sel.outputs[0], g.inputs[0]); nt.links.new(band.outputs[0], g.inputs[1])
        goldmask = g.outputs[0]
        bandmask = band.outputs[0]
    # base oscurecida dentro de la banda, oro encima
    dark = rgb_scale(nt, base_socket, band_dark)
    m1 = N(nt, 'ShaderNodeMixRGB', blend_type='MIX')
    nt.links.new(bandmask, m1.inputs['Fac']); nt.links.new(base_socket, m1.inputs['Color1']); nt.links.new(dark, m1.inputs['Color2'])
    m2 = N(nt, 'ShaderNodeMixRGB', blend_type='MIX')
    nt.links.new(goldmask, m2.inputs['Fac']); nt.links.new(m1.outputs[0], m2.inputs['Color1'])
    m2.inputs['Color2'].default_value = (*gold, 1)
    return m2.outputs[0], goldmask

def toon_mat(name, base, rim=(0.25, 0.65, 1.0), rim_k=0.35, amb=0.22, size=0.62, smooth=0.06,
             trim=False, line_only=False, gloss=0.0):
    mat = bpy.data.materials.new(name)
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    rgb = N(nt, 'ShaderNodeRGB'); rgb.outputs[0].default_value = (*base, 1)
    col = rgb.outputs[0]
    goldmask = None
    if trim:
        col, goldmask = trim_color(nt, col, GOLD, line_only=line_only)
    toon = N(nt, 'ShaderNodeBsdfToon', component='DIFFUSE')
    toon.inputs['Size'].default_value = size
    toon.inputs['Smooth'].default_value = smooth
    nt.links.new(col, toon.inputs['Color'])
    em = N(nt, 'ShaderNodeEmission')
    nt.links.new(rgb_scale(nt, col, amb), em.inputs['Color'])
    em.inputs['Strength'].default_value = 1.0
    add1 = N(nt, 'ShaderNodeAddShader')
    nt.links.new(toon.outputs[0], add1.inputs[0]); nt.links.new(em.outputs[0], add1.inputs[1])
    last = add1.outputs[0]
    if rim_k > 0:
        lw = N(nt, 'ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.35
        ramp = N(nt, 'ShaderNodeValToRGB')
        ramp.color_ramp.interpolation = 'CONSTANT'
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        ramp.color_ramp.elements[1].position = 0.72
        ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
        nt.links.new(lw.outputs['Facing'], ramp.inputs['Fac'])
        rimc = N(nt, 'ShaderNodeMixRGB', blend_type='MULTIPLY')
        rimc.inputs['Fac'].default_value = 1.0
        nt.links.new(ramp.outputs['Color'], rimc.inputs['Color1'])
        rimc.inputs['Color2'].default_value = (*rim, 1)
        em2 = N(nt, 'ShaderNodeEmission'); em2.inputs['Strength'].default_value = rim_k
        nt.links.new(rimc.outputs[0], em2.inputs['Color'])
        add2 = N(nt, 'ShaderNodeAddShader')
        nt.links.new(last, add2.inputs[0]); nt.links.new(em2.outputs[0], add2.inputs[1])
        last = add2.outputs[0]
    if goldmask is not None and not line_only:
        # el oro brilla un poco y tiene un reflejo duro
        gl = N(nt, 'ShaderNodeBsdfToon', component='GLOSSY')
        gl.inputs['Size'].default_value = 0.22; gl.inputs['Smooth'].default_value = 0.05
        gl.inputs['Color'].default_value = (1.0, 0.85, 0.55, 1)
        ge = N(nt, 'ShaderNodeEmission'); ge.inputs['Color'].default_value = (*GOLD, 1); ge.inputs['Strength'].default_value = 0.35
        addg = N(nt, 'ShaderNodeAddShader')
        nt.links.new(gl.outputs[0], addg.inputs[0]); nt.links.new(ge.outputs[0], addg.inputs[1])
        mixg = N(nt, 'ShaderNodeMixShader')
        nt.links.new(goldmask, mixg.inputs['Fac'])
        addb = N(nt, 'ShaderNodeAddShader')
        nt.links.new(last, addb.inputs[0]); nt.links.new(addg.outputs[0], addb.inputs[1])
        nt.links.new(last, mixg.inputs[1]); nt.links.new(addb.outputs[0], mixg.inputs[2])
        last = mixg.outputs[0]
    nt.links.new(last, out.inputs['Surface'])
    return mat

def gold_mat(name='oro'):
    mat = bpy.data.materials.new(name)
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    d = N(nt, 'ShaderNodeBsdfToon', component='DIFFUSE')
    d.inputs['Color'].default_value = (*GOLD, 1); d.inputs['Size'].default_value = 0.7; d.inputs['Smooth'].default_value = 0.05
    g = N(nt, 'ShaderNodeBsdfToon', component='GLOSSY')
    g.inputs['Color'].default_value = (1.0, 0.85, 0.5, 1); g.inputs['Size'].default_value = 0.25; g.inputs['Smooth'].default_value = 0.05
    e = N(nt, 'ShaderNodeEmission'); e.inputs['Color'].default_value = (*GOLD, 1); e.inputs['Strength'].default_value = 0.4
    a1 = N(nt, 'ShaderNodeAddShader'); a2 = N(nt, 'ShaderNodeAddShader')
    nt.links.new(d.outputs[0], a1.inputs[0]); nt.links.new(g.outputs[0], a1.inputs[1])
    nt.links.new(a1.outputs[0], a2.inputs[0]); nt.links.new(e.outputs[0], a2.inputs[1])
    nt.links.new(a2.outputs[0], out.inputs['Surface'])
    return mat

def emit_mat(name, color, strength):
    mat = bpy.data.materials.new(name)
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    e = N(nt, 'ShaderNodeEmission'); e.inputs['Color'].default_value = (*color, 1); e.inputs['Strength'].default_value = strength
    nt.links.new(e.outputs[0], out.inputs['Surface'])
    return mat

M = {}
M['cloak'] = toon_mat('capa', (0.020, 0.032, 0.095), rim=(0.25, 0.7, 1.0), rim_k=0.18, trim=True)
M['cloak_in'] = toon_mat('capa_forro', (0.10, 0.022, 0.17), rim=(0.6, 0.3, 1.0), rim_k=0.15, amb=0.25, trim=True, line_only=True)
M['hood'] = toon_mat('capucha', (0.024, 0.028, 0.085), rim=(0.25, 0.7, 1.0), rim_k=0.22, trim=True)
M['hood_in'] = toon_mat('capucha_dentro', (0.006, 0.004, 0.012), rim_k=0.0, amb=0.1, trim=True, line_only=True)
M['cowl'] = toon_mat('cuello', (0.024, 0.028, 0.085), rim=(0.25, 0.7, 1.0), rim_k=0.2, trim=True)
M['cowl_in'] = toon_mat('cuello_dentro', (0.06, 0.015, 0.10), rim_k=0.1, trim=True, line_only=True)
M['robe'] = toon_mat('tunica', (0.010, 0.011, 0.032), rim=(0.4, 0.4, 1.0), rim_k=0.2, trim=True)
M['robe_in'] = toon_mat('tunica_dentro', (0.006, 0.005, 0.014), rim_k=0.0)
M['sleeve'] = toon_mat('manga', (0.016, 0.016, 0.050), rim=(0.3, 0.6, 1.0), rim_k=0.3, trim=True)
M['sleeve_in'] = toon_mat('manga_forro', (0.12, 0.03, 0.21), rim_k=0.15, amb=0.3, trim=True, line_only=True)
M['sash'] = toon_mat('fajin', (0.10, 0.022, 0.17), rim=(0.7, 0.4, 1.0), rim_k=0.15)
M['glove'] = toon_mat('guante', (0.055, 0.042, 0.050), rim=(0.3, 0.7, 1.0), rim_k=0.3, amb=0.3)
M['face'] = toon_mat('cara', (0.004, 0.004, 0.008), rim_k=0.0, amb=0.0)
def floor_mat():
    mat = bpy.data.materials.new('piso')
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    geo = N(nt, 'ShaderNodeNewGeometry')
    sxyz = N(nt, 'ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sxyz.inputs[0])
    cxy = N(nt, 'ShaderNodeCombineXYZ')
    nt.links.new(sxyz.outputs['X'], cxy.inputs['X']); nt.links.new(sxyz.outputs['Y'], cxy.inputs['Y'])
    ln = N(nt, 'ShaderNodeVectorMath', operation='LENGTH'); nt.links.new(cxy.outputs[0], ln.inputs[0])
    mr = N(nt, 'ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.6; mr.inputs['From Max'].default_value = 5.0
    mr.interpolation_type = 'SMOOTHSTEP'
    nt.links.new(ln.outputs['Value'], mr.inputs['Value'])
    mix = N(nt, 'ShaderNodeMixRGB', blend_type='MIX')
    nt.links.new(mr.outputs[0], mix.inputs['Fac'])
    mix.inputs['Color1'].default_value = (0.020, 0.030, 0.075, 1)
    mix.inputs['Color2'].default_value = (0.003, 0.004, 0.011, 1)
    toon = N(nt, 'ShaderNodeBsdfToon', component='DIFFUSE')
    toon.inputs['Size'].default_value = 0.7; toon.inputs['Smooth'].default_value = 0.1
    nt.links.new(mix.outputs[0], toon.inputs['Color'])
    em = N(nt, 'ShaderNodeEmission'); em.inputs['Strength'].default_value = 0.6
    nt.links.new(mix.outputs[0], em.inputs['Color'])
    add = N(nt, 'ShaderNodeAddShader')
    nt.links.new(toon.outputs[0], add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs['Surface'])
    return mat

M['floor'] = floor_mat()
M['gold'] = gold_mat()
M['eye'] = emit_mat('ojos', (0.55, 0.95, 1.0), 28.0)
M['gem'] = emit_mat('gema', (0.08, 0.45, 1.0), 2.2)
M['gem_core'] = emit_mat('gema_brillo', (0.6, 0.9, 1.0), 4.0)
M['rune'] = emit_mat('runas', (0.25, 0.8, 1.0), 2.6)
M['rune_floor'] = emit_mat('runas_piso', (0.25, 0.75, 1.0), 1.8)
M['spark'] = emit_mat('chispas', (0.5, 0.9, 1.0), 9.0)

# ---------------------------------------------------------------- cuerpo invisible (colisionadores)
def ellipsoid(name, c, r, coll=C_COL, seg=32, rings=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0)
    for v in bm.verts:
        v.co = Vector((c[0] + v.co.x * r[0], c[1] + v.co.y * r[1], c[2] + v.co.z * r[2]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob

def cone_between(name, a, b, ra, rb, coll=C_COL, seg=24):
    a, b = Vector(a), Vector(b)
    d = b - a
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=ra, radius2=rb, depth=d.length)
    rot = d.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    mat = Matrix.Translation((a + b) / 2) @ rot
    bmesh.ops.transform(bm, matrix=mat, verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob

def make_collider(ob, outer=0.008, friction=8.0, single=True):
    ob.modifiers.new('Collision', 'COLLISION')
    ob.collision.thickness_outer = outer
    ob.collision.cloth_friction = friction
    try:
        ob.collision.use_culling = single
    except Exception:
        pass
    ob.hide_render = True
    return ob

HEAD_C = Vector((0, 0.0, 1.66))
body = {}
body['head'] = ellipsoid('c_cabeza', HEAD_C, (0.085, 0.10, 0.115))
body['neck'] = cone_between('c_cuello', (0, 0.01, 1.44), (0, 0.0, 1.60), 0.058, 0.052)
body['shoulders'] = ellipsoid('c_hombros', (0, 0.015, 1.43), (0.24, 0.085, 0.06))
body['chest'] = ellipsoid('c_pecho', (0, 0.01, 1.30), (0.17, 0.12, 0.20))
body['belly'] = ellipsoid('c_panza', (0, 0.01, 1.05), (0.15, 0.11, 0.18))
body['hips'] = ellipsoid('c_cadera', (0, 0.015, 0.90), (0.18, 0.13, 0.13))
for s, nm in ((1, 'd'), (-1, 'i')):
    body['thigh' + nm] = cone_between('c_muslo_' + nm, (s * 0.09, 0.01, 0.86), (s * 0.11, 0.0, 0.47), 0.085, 0.06)
    body['shin' + nm] = cone_between('c_pierna_' + nm, (s * 0.11, 0.0, 0.47), (s * 0.12, 0.0, 0.06), 0.06, 0.045)
    body['knee' + nm] = ellipsoid('c_rodilla_' + nm, (s * 0.11, 0.0, 0.47), (0.06, 0.06, 0.06), seg=16, rings=8)
    body['foot' + nm] = ellipsoid('c_pie_' + nm, (s * 0.125, -0.06, 0.04), (0.05, 0.12, 0.04), seg=16, rings=8)

# brazos: hombro -> codo -> muñeca (antebrazos hacia adelante, manos delante de la cadera)
ARM = {}
for s, nm in ((1, 'd'), (-1, 'i')):
    S = Vector((s * 0.215, 0.02, 1.42))
    E = Vector((s * 0.285, 0.03, 1.14))
    W = Vector((s * 0.235, -0.27, 1.03))
    ARM[nm] = (S, E, W, s)
arms = []
for nm, (S, E, W, s) in ARM.items():
    arms.append(cone_between('c_brazo_' + nm, S, E, 0.05, 0.043))
    arms.append(cone_between('c_antebrazo_' + nm, E, W, 0.042, 0.032))
    arms.append(ellipsoid('c_hombro_' + nm, S, (0.05, 0.05, 0.05), seg=16, rings=8))
    arms.append(ellipsoid('c_codo_' + nm, E, (0.043, 0.043, 0.043), seg=16, rings=8))

for ob in list(body.values()) + arms:
    make_collider(ob)

# piso (se ve y además es colisionador)
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = 'piso'
for c in floor.users_collection:
    c.objects.unlink(floor)
C_FX.objects.link(floor)
floor.data.materials.append(M['floor'])
floor.modifiers.new('Collision', 'COLLISION')
floor.collision.thickness_outer = 0.005
floor.collision.cloth_friction = 20.0

def add_to(cc_names, obs):
    for k in cc_names:
        for ob in obs:
            if ob.name not in CC[k].objects:
                CC[k].objects.link(ob)

torso_names = ['neck', 'shoulders', 'chest', 'belly', 'hips']
legs = [body[k] for k in body if k[:-1] in ('thigh', 'shin', 'knee', 'foot')]
add_to(['robe'], [body[k] for k in torso_names] + legs + [floor])
add_to(['sleeve'], arms)
add_to(['cloak'], list(body.values()) + arms + [floor])
add_to(['cowl'], [body[k] for k in ('head', 'neck', 'shoulders', 'chest')])
add_to(['sash'], legs + [floor])

# ---------------------------------------------------------------- simulación
def cloth(ob, coll_key, *, mass=0.3, tension=15, bending=0.6, shear=5, air=2.0, quality=6, self_col=False,
          dist=0.006, self_dist=0.005, frames=80, pin='pin', friction=None):
    mod = ob.modifiers.new('Cloth', 'CLOTH')
    s = mod.settings
    s.quality = quality
    s.mass = mass
    s.air_damping = air
    s.tension_stiffness = tension
    s.compression_stiffness = tension
    s.shear_stiffness = shear
    s.bending_stiffness = bending
    s.tension_damping = 5
    s.compression_damping = 5
    s.shear_damping = 5
    s.bending_damping = 0.5
    if pin and ob.vertex_groups.get(pin):
        s.vertex_group_mass = pin
        s.pin_stiffness = 1.0
    c = mod.collision_settings
    c.use_collision = True
    c.collection = CC[coll_key]
    c.distance_min = dist
    c.collision_quality = 4
    c.use_self_collision = self_col
    if self_col:
        c.self_distance_min = self_dist
        c.self_friction = 5
    mod.point_cache.frame_start = 1
    mod.point_cache.frame_end = frames
    return mod

def run_sim(obs, frames, label):
    t = time.time()
    scene.frame_start = 1
    scene.frame_end = frames
    bpy.context.view_layer.update()
    scene.frame_set(1)
    for f in range(2, frames + 1):
        scene.frame_set(f)
        if f % 10 == 0:
            log('  %s cuadro %d/%d' % (label, f, frames))
    deps = bpy.context.evaluated_depsgraph_get()
    for ob in obs:
        ev = ob.evaluated_get(deps)
        me = bpy.data.meshes.new_from_object(ev)
        old = ob.data
        for m in list(ob.modifiers):
            ob.modifiers.remove(m)
        ob.data = me
        me.name = ob.name
        bpy.data.meshes.remove(old)
    scene.frame_set(1)
    log('%s listo en %.1fs (%d cuadros)' % (label, time.time() - t, frames))

def frames(n):
    return max(10, int(n * args.simscale))

# ---------------------------------------------------------------- 1. túnica interior
def robe():
    keys = [  # z, rx, ry, yc
        (1.36, 0.180, 0.135, 0.010),
        (1.25, 0.185, 0.140, 0.010),
        (1.13, 0.180, 0.140, 0.008),
        (1.02, 0.165, 0.125, 0.010),
        (0.93, 0.205, 0.160, 0.012),
        (0.60, 0.280, 0.230, 0.010),
        (0.03, 0.420, 0.350, 0.000),
    ]
    rows = resample(catmull(keys, 30), 64, cols=[0, 1])
    NU = 112
    a = np.linspace(0, 2 * np.pi, NU, endpoint=False)
    P = np.zeros((len(rows), NU, 3))
    for i, (z, rx, ry, yc) in enumerate(rows):
        P[i, :, 0] = rx * np.sin(a)
        P[i, :, 1] = yc - ry * np.cos(a)
        P[i, :, 2] = z
    nv = len(rows)
    ob = mesh_obj('tunica', P.reshape(-1, 3), grid_faces(nv, NU, closed_u=True), C_CHAR)
    orient(ob, axis_center(0.01))
    su, sv, utot = arc_lengths(P, closed_u=True)
    sd = np.where(su < utot[:, None] / 2, su, su - utot[:, None])
    bw = 0.05
    A = 0.5 + np.clip(sd / (2 * bw), -3, 3)
    B = sv / (TILE * bw * 2)  # banda de 2*bw de ancho
    set_uv(ob, 'trim', np.stack([A, B], -1).reshape(-1, 2))
    set_uv(ob, 'kind', np.stack([np.ones_like(A), np.zeros_like(A)], -1).reshape(-1, 2))
    z = P[:, 0, 2]
    pin_rows = [i for i in range(nv) if i < 2 or abs(z[i] - 1.02) < 0.035]
    add_pin(ob, [i * NU + j for i in pin_rows for j in range(NU)])
    set_materials(ob, [M['robe'], M['robe_in']])
    smooth(ob)
    return ob

log('construyendo túnica')
ROBE = robe()
cloth(ROBE, 'robe', mass=0.25, tension=20, bending=0.8, air=3.0, frames=frames(70))
run_sim([ROBE], frames(70), 'túnica')
make_collider(ROBE, outer=0.006, friction=10, single=False)
ROBE.hide_render = False
add_to(['sleeve', 'cloak', 'sash'], [ROBE])

# fajín y cinturón (estáticos, sobre la cintura fijada de la túnica)
def band_ring(name, z0, z1, rx, ry, yc, bulge, mat, nu=96, nv=6):
    a = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    verts = []
    for i in range(nv):
        t = i / (nv - 1)
        z = z0 + (z1 - z0) * t
        k = 1 + bulge * math.sin(math.pi * t) + 0.02 * np.sin(5 * a + 2 * t)
        verts += list(zip(rx * k * np.sin(a), yc - ry * k * np.cos(a), np.full(nu, z)))
    faces = grid_faces(nv, nu, closed_u=True)
    ob = mesh_obj(name, verts, faces, C_CHAR)
    orient(ob, axis_center(yc))
    ob.data.materials.append(mat)
    smooth(ob)
    return ob

SASH = band_ring('fajin', 0.985, 1.058, 0.181, 0.144, 0.010, 0.06, M['sash'])
BELT = band_ring('cinturon', 1.056, 1.066, 0.186, 0.149, 0.010, 0.0, M['gold'], nv=3)
KNOT = ellipsoid('nudo', (-0.075, -0.152, 1.015), (0.030, 0.020, 0.026), coll=C_CHAR, seg=20, rings=12)
KNOT.data.materials.append(M['sash'])
smooth(KNOT)
for ob in (SASH, KNOT):
    make_collider(ob, outer=0.004, friction=5, single=False)
    ob.hide_render = False
add_to(['sleeve', 'cloak', 'sash'], [SASH, KNOT])

# ---------------------------------------------------------------- 2. mangas acampanadas + puntas del fajín
def tube_along(name, pts, radii, nu, nv, coll=C_CHAR):
    """Tubo abierto a lo largo de una curva suave; radii = [(fracción de largo, radio), ...]."""
    dense = catmull(pts, 30)
    path = resample(dense, nv)
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    # marco por transporte paralelo
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    Nn = np.cross(T[0], ref); Nn /= np.linalg.norm(Nn)
    frames_ = []
    for i in range(nv):
        if i > 0:
            Nn = Nn - T[i] * np.dot(Nn, T[i]); Nn /= np.linalg.norm(Nn)
        Bn = np.cross(T[i], Nn)
        frames_.append((Nn.copy(), Bn))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    fr = s / s[-1]
    rk = np.array(radii, float)
    r = np.interp(fr, rk[:, 0], rk[:, 1])
    a = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    P = np.zeros((nv, nu, 3))
    for i in range(nv):
        Nn, Bn = frames_[i]
        P[i] = path[i] + r[i] * (np.cos(a)[:, None] * Nn + np.sin(a)[:, None] * Bn)
    ob = mesh_obj(name, P.reshape(-1, 3), grid_faces(nv, nu, closed_u=True), coll)
    # normales hacia afuera del eje
    me = ob.data; me.update()
    tot = 0
    for p in me.polygons:
        i = p.vertices[0] // nu
        tot += p.normal.dot(p.center - Vector(path[min(i, nv - 1)]))
    if tot < 0:
        bm = bmesh.new(); bm.from_mesh(me)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:]); bm.to_mesh(me); bm.free()
    return ob, P, path, s

SLEEVES = []
for nm, (S, E, W, sgn) in ARM.items():
    fwd = (W - E).normalized()
    C = W + fwd * 0.07
    top = S + Vector((-sgn * 0.01, 0, 0.05))
    pts = [tuple(top), tuple(S + (E - S) * 0.5), tuple(E), tuple(E + (W - E) * 0.55), tuple(W), tuple(C)]
    radii = [(0, 0.072), (0.35, 0.078), (0.55, 0.085), (0.80, 0.13), (1.0, 0.19)]
    nv, nu = 44, 44
    ob, P, path, s = tube_along('manga_' + nm, pts, radii, nu, nv)
    su, sv, utot = arc_lengths(P, closed_u=True)
    bw = 0.045
    d = sv[-1][None, :] - sv  # distancia al puño
    A = su / (TILE * bw)
    B = d / bw
    set_uv(ob, 'trim', np.stack([A, B], -1).reshape(-1, 2))
    set_uv(ob, 'kind', np.zeros((nv * nu, 2)))
    add_pin(ob, list(range(0, 3 * nu)))
    set_materials(ob, [M['sleeve'], M['sleeve_in']])
    smooth(ob)
    cloth(ob, 'sleeve', mass=0.2, tension=15, bending=0.5, air=2.0, self_col=True, frames=frames(60))
    SLEEVES.append(ob)

def ribbon(name, top, length, width, angle, nv=26, nu=5):
    verts = []
    dirv = Vector((math.sin(angle), -0.45, -math.cos(angle))).normalized()
    side = Vector((1, 0, 0))
    for i in range(nv):
        for j in range(nu):
            p = Vector(top) + dirv * (length * i / (nv - 1)) + side * (width * (j / (nu - 1) - 0.5))
            verts.append(tuple(p))
    ob = mesh_obj(name, verts, grid_faces(nv, nu), C_CHAR)
    orient(ob, lambda c: (c.x, c.y + 1.0, c.z))  # normales hacia adelante (-y)
    add_pin(ob, list(range(0, 2 * nu)))
    ob.data.materials.append(M['sash'])
    smooth(ob)
    cloth(ob, 'sash', mass=0.08, tension=15, bending=0.08, air=1.0, frames=frames(60))
    return ob

RIBBONS = [ribbon('fajin_punta_1', (-0.092, -0.166, 0.995), 0.50, 0.068, 0.22),
           ribbon('fajin_punta_2', (-0.060, -0.170, 0.992), 0.40, 0.060, -0.12)]
add_to(['sash'], [ROBE, SASH])
log('simulando mangas y fajín')
run_sim(SLEEVES + RIBBONS, frames(60), 'mangas')
for ob in SLEEVES:
    make_collider(ob, outer=0.006, friction=8, single=False)
    ob.hide_render = False
add_to(['cloak'], SLEEVES)

# ---------------------------------------------------------------- 3. capa
def cloak():
    keys = [  # z, rx, ry, yc, media apertura (rad)
        (1.545, 0.105, 0.100, 0.015, 0.20),
        (1.505, 0.200, 0.160, 0.015, 0.26),
        (1.455, 0.310, 0.240, 0.020, 0.36),
        (1.350, 0.480, 0.380, 0.020, 0.55),
        (1.000, 0.640, 0.530, 0.020, 0.80),
        (0.040, 1.050, 0.900, 0.030, 0.88),
    ]
    rows = resample(catmull(keys, 30), 66, cols=[0, 1])
    NU = 156
    nv = len(rows)
    P = np.zeros((nv, NU, 3))
    for i, (z, rx, ry, yc, ao) in enumerate(rows):
        a = np.linspace(ao, 2 * np.pi - ao, NU)
        P[i, :, 0] = rx * np.sin(a)
        P[i, :, 1] = yc - ry * np.cos(a)
        P[i, :, 2] = z
    ob = mesh_obj('capa', P.reshape(-1, 3), grid_faces(nv, NU), C_CHAR)
    orient(ob, axis_center(0.02))
    su, sv, utot = arc_lengths(P)
    bw = 0.055
    dfront = np.minimum(su, utot[:, None] - su)
    dhem = sv[-1][None, :] - sv
    kindv = (dfront < dhem).astype(float)
    A = np.where(kindv > 0.5, dfront / bw, su / (TILE * bw))
    B = np.where(kindv > 0.5, sv / (TILE * bw), dhem / bw)
    set_uv(ob, 'trim', np.stack([A, B], -1).reshape(-1, 2))
    set_uv(ob, 'kind', np.stack([kindv, np.zeros_like(kindv)], -1).reshape(-1, 2))
    pins = [i * NU + j for i in range(2) for j in range(NU)]
    pins += [i * NU + j for i in range(2, 5) for j in (0, 1, NU - 2, NU - 1)]
    add_pin(ob, pins)
    set_materials(ob, [M['cloak'], M['cloak_in']])
    smooth(ob)
    return ob

log('construyendo capa')
CLOAK = cloak()
cloth(CLOAK, 'cloak', mass=0.35, tension=25, bending=1.2, shear=8, air=3.0, quality=7, self_col=True,
      dist=0.007, self_dist=0.006, frames=frames(100))
run_sim([CLOAK], frames(100), 'capa')
make_collider(CLOAK, outer=0.006, friction=10, single=False)
CLOAK.hide_render = False
add_to(['cowl'], [CLOAK] + SLEEVES)

# ---------------------------------------------------------------- capucha (esculpida)
def hood():
    W, R_arc = 0.135, 0.216
    zb, zs = 1.48, 1.62
    nb, na = 16, 34
    rim = []
    for k in range(nb):  # tramo recto de abajo (izquierda, x<0)
        h = k / nb
        rim.append((-(0.112 + 0.023 * math.sin(h * math.pi / 2)), zb + (zs - zb) * h))
    amax = math.acos((R_arc - W) / R_arc)
    for k in range(na + 1):  # arco ojival
        al = amax * k / na
        x = (R_arc - W) - R_arc * math.cos(al)
        rim.append((x, zs + R_arc * math.sin(al)))
    rim = np.array(rim)
    rim[-1, 0] = 0.0
    # profundidad del borde: el ápice avanza (alero que da sombra), los costados se van hacia atrás
    hfrac = (rim[:, 1] - zb) / (rim[-1, 1] - zb)
    ry = -0.085 - 0.105 * hfrac ** 1.3 + 0.035 * (np.abs(rim[:, 0]) / W) ** 2
    O_left = np.stack([rim[:, 0], ry, rim[:, 1]], 1)
    # costura central: del ápice, sube a la punta, baja por la nuca
    seam_pts = [(0, -0.19, 1.82), (0, -0.125, 1.92), (0, -0.05, 2.0), (0, 0.025, 2.055),
                (0, 0.13, 1.94), (0, 0.20, 1.76), (0, 0.19, 1.59), (0, 0.155, 1.47)]
    seam = catmull(seam_pts, 40)
    # correspondencia por fracción de arco
    def arcfrac(Pp):
        d = np.linalg.norm(np.diff(Pp, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(d)])
        return s / s[-1]
    qr = arcfrac(O_left)           # 0 abajo -> 1 ápice
    qs = arcfrac(seam)             # 0 ápice -> 1 nuca
    Sx = np.stack([np.interp(1 - qr, qs, seam[:, k]) for k in range(3)], 1)
    # lado derecho = espejo; el borde completo va de abajo-izq -> ápice -> abajo-der
    O_full = np.vstack([O_left, (O_left[:-1] * [-1, 1, 1])[::-1]])
    S_full = np.vstack([Sx, Sx[:-1][::-1]])
    q_full = np.concatenate([qr, qr[:-1][::-1]])
    nrim = len(O_full)
    NT = 30
    Hc = np.array([0, 0.03, 1.69])
    span = np.linalg.norm(O_full - S_full, axis=1)
    fspan = np.clip(span / 0.06, 0, 1)  # junto al ápice el borde y la costura coinciden: sin abultar
    P = np.zeros((NT, nrim, 3))
    for j in range(NT):
        t = j / (NT - 1)
        base = (1 - t) * O_full + t * S_full
        v = base - Hc
        dist = np.linalg.norm(v, axis=1)
        n = v / dist[:, None]
        w = math.sin(math.pi * t) ** 0.7
        Rt = 0.20 + 0.025 * (1 - q_full)
        r = dist + fspan * (w * np.maximum(0, Rt - dist) + 0.012 * math.sin(math.pi * t))
        p = Hc + n * r[:, None]
        # pliegues suaves en la parte baja (donde la tela se junta sobre el cuello)
        fold = fspan * 0.011 * np.sin(2 * math.pi * (3.2 * t + 2.0 * q_full)) * (1 - q_full) ** 1.6 * math.sin(math.pi * t)
        p = p + n * fold[:, None]
        # se angosta hacia arriba: silueta ojival que termina en punta
        zr = np.clip((p[:, 2] - 1.72) / 0.30, 0, 1)
        p[:, 0] *= 1 - 0.42 * zr ** 1.3
        P[j] = p
    nv, nu = NT, nrim
    ob = mesh_obj('capucha', P.reshape(-1, 3), grid_faces(nv, nu), C_CHAR)
    su, sv, utot = arc_lengths(P)
    bw = 0.035
    A = su / (TILE * bw)
    B = sv / bw  # distancia al borde de la cara
    set_uv(ob, 'trim', np.stack([A, B], -1).reshape(-1, 2))
    set_uv(ob, 'kind', np.zeros((nv * nu, 2)))
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0015)
    bm.to_mesh(me); bm.free(); me.update()
    orient(ob, lambda c: tuple(Hc))
    set_materials(ob, [M['hood'], M['hood_in']])
    smooth(ob)
    return ob

log('construyendo capucha')
HOOD = hood()

# ---------------------------------------------------------------- 4. cuello / capelina
def cowl():
    keys = [  # z, rx, ry, yc
        (1.52, 0.150, 0.138, 0.020),
        (1.51, 0.235, 0.195, 0.020),
        (1.47, 0.340, 0.280, 0.020),
        (1.39, 0.480, 0.390, 0.020),
        (1.29, 0.600, 0.490, 0.020),
    ]
    rows = resample(catmull(keys, 30), 26, cols=[0, 1])
    NU = 132
    a = np.linspace(0, 2 * np.pi, NU, endpoint=False)
    nv = len(rows)
    P = np.zeros((nv, NU, 3))
    for i, (z, rx, ry, yc) in enumerate(rows):
        P[i, :, 0] = rx * np.sin(a)
        P[i, :, 1] = yc - ry * np.cos(a)
        P[i, :, 2] = z
    ob = mesh_obj('cuello', P.reshape(-1, 3), grid_faces(nv, NU, closed_u=True), C_CHAR)
    orient(ob, lambda c: (0, 0.02, c.z - 0.25))
    su, sv, utot = arc_lengths(P, closed_u=True)
    bw = 0.04
    d = sv[-1][None, :] - sv
    set_uv(ob, 'trim', np.stack([su / (TILE * bw), d / bw], -1).reshape(-1, 2))
    set_uv(ob, 'kind', np.zeros((nv * NU, 2)))
    add_pin(ob, list(range(0, 2 * NU)))
    set_materials(ob, [M['cowl'], M['cowl_in']])
    smooth(ob)
    return ob

log('construyendo cuello')
COWL = cowl()
cloth(COWL, 'cowl', mass=0.2, tension=20, bending=0.3, air=3.0, self_col=True, dist=0.006, frames=frames(70))
run_sim([COWL], frames(70), 'cuello')

# ---------------------------------------------------------------- cara, ojos, manos
FACE = ellipsoid('cara', HEAD_C + Vector((0, -0.005, 0)), (0.083, 0.098, 0.113), coll=C_FX)
FACE.data.materials.append(M['face'])
smooth(FACE)
EYES = []
for sx in (-1, 1):
    e = ellipsoid('ojo', (sx * 0.034, -0.093, 1.668), (0.016, 0.006, 0.0062), coll=C_FX, seg=16, rings=8)
    e.rotation_euler = (0, sx * math.radians(-7), 0)
    e.data.materials.append(M['eye'])
    EYES.append(e)
# rotar alrededor de su propio centro
for e in EYES:
    me = e.data
    c = sum((v.co for v in me.vertices), Vector()) / len(me.vertices)
    for v in me.vertices:
        v.co -= c
    e.location = c

def hand(name, W, E, sgn):
    """Mano enguantada estilizada: palma+dedos juntos en una sola pieza curvada, con surcos suaves entre dedos, y pulgar."""
    bm = bmesh.new()
    L, Wd, Th = 0.175, 0.082, 0.030
    res = bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=1.0)
    for v in res['verts']:
        x = (v.co.x * 0.5 + 0.5) * L
        y = v.co.y * Wd / 2
        z = v.co.z * Th / 2
        # muñeca más angosta, dedos un poco más finos
        y *= 0.78 + 0.22 * min(1, x / 0.05) - 0.12 * max(0, (x - 0.10) / 0.075)
        z *= 0.85 + 0.15 * min(1, x / 0.05) - 0.25 * max(0, (x - 0.12) / 0.055)
        # surcos entre dedos en el dorso y la palma
        if x > 0.095:
            k = min(1, (x - 0.095) / 0.02)
            z *= 1 - 0.18 * k * abs(math.cos(math.pi * 2 * (y / (Wd * 0.9)) * 2)) ** 3
        # curvatura de los dedos hacia la palma (-z)
        if x > 0.085:
            ang = (x - 0.085) * 7.0
            rx, rz = x - 0.085, z
            x = 0.085 + rx * math.cos(ang) + rz * math.sin(ang)
            z = -rx * math.sin(ang) + rz * math.cos(ang)
        v.co = Vector((x, y, z))
    res = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=10, radius=1.0)
    thumb_m = Matrix.Translation((0.055, 0.040, -0.010)) @ Matrix.Rotation(math.radians(40), 4, 'Z') @ Matrix.Rotation(math.radians(-20), 4, 'Y')
    for v in res['verts']:
        v.co = thumb_m @ Vector((v.co.x * 0.040, v.co.y * 0.013, v.co.z * 0.012))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    C_CHAR.objects.link(ob)
    # orientación: dedos siguiendo el antebrazo un poco hacia abajo; dorso hacia afuera y arriba; pulgar arriba
    fwd = (W - E).normalized()
    fwd = (fwd + Vector((0, 0, -0.35))).normalized()
    back = Vector((sgn * 1.0, 0, 0.55)).normalized()
    back = (back - fwd * back.dot(fwd)).normalized()
    side = back.cross(fwd)  # y local = z × x
    R = Matrix((fwd, side, back)).transposed()
    ob.matrix_world = Matrix.Translation(W + fwd * 0.045) @ R.to_4x4()
    if side.z < 0:  # que el pulgar quede arriba: espejar en y local
        ob.matrix_world = ob.matrix_world @ Matrix.Diagonal((1, -1, 1, 1))
    ob.data.materials.append(M['glove'])
    smooth(ob)
    sub = ob.modifiers.new('Sub', 'SUBSURF'); sub.levels = 1; sub.render_levels = 1
    return ob

HANDS = [hand('mano_' + nm, W, E, s) for nm, (S, E, W, s) in ARM.items()]

# ---------------------------------------------------------------- medallón (sobre la tela, por rayo)
for ob in list(body.values()) + arms:
    ob.hide_viewport = True
bpy.context.view_layer.update()

def surface_hit(x, z, y0=-1.5):
    deps = bpy.context.evaluated_depsgraph_get()
    hit, loc, nor, idx, obj, mat = scene.ray_cast(deps, Vector((x, y0, z)), Vector((0, 1, 0)))
    if hit:
        return loc, nor, obj
    return Vector((x, -0.15, z)), Vector((0, -1, 0)), None

for e in EYES + [FACE]:
    e.hide_viewport = True
bpy.context.view_layer.update()
mloc, mnor, mobj = surface_hit(0.0, 1.31)
log('medallón sobre', mobj.name if mobj else None, tuple(round(v, 3) for v in mloc))
mnor = (mnor + Vector((0, -1.5, 0))).normalized()
MED_C = mloc + mnor * 0.016

def medallion(c, n):
    bm = bmesh.new()
    def cyl(r1, r2, depth, seg, off):
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r1, radius2=r2, depth=depth)
        bmesh.ops.translate(bm, vec=Vector((0, 0, off)), verts=res['verts'])
    cyl(0.052, 0.048, 0.010, 64, 0.0)
    cyl(0.040, 0.036, 0.008, 64, 0.007)
    # rayos de sol alrededor
    for k in range(12):
        ang = 2 * math.pi * k / 12
        L = 0.022 if k % 2 == 0 else 0.014
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.008, radius2=0.0005, depth=L)
        for v in res['verts']:
            v.co.x *= 0.3
        m = Matrix.Rotation(ang, 4, 'Z') @ Matrix.Translation((0.052 + L / 2 - 0.004, 0, 0)) @ Matrix.Rotation(math.pi / 2, 4, 'Y')
        bmesh.ops.transform(bm, matrix=m, verts=res['verts'])
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.006, radius2=0.006, depth=0.012)
    bmesh.ops.transform(bm, matrix=Matrix.Translation((0, 0.058, 0)) @ Matrix.Rotation(math.pi / 2, 4, 'Y'), verts=res['verts'])
    me = bpy.data.meshes.new('medallon')
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new('medallon', me)
    C_CHAR.objects.link(ob)
    rot = n.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    ob.matrix_world = Matrix.Translation(c) @ rot
    ob.data.materials.append(M['gold'])
    bev = ob.modifiers.new('Bevel', 'BEVEL'); bev.width = 0.0015; bev.segments = 2
    gem = ellipsoid('gema', (0, 0, 0), (0.021, 0.021, 0.012), coll=C_FX, seg=24, rings=12)
    gem.matrix_world = Matrix.Translation(c + n * 0.013) @ rot
    gem.data.materials.append(M['gem'])
    core = ellipsoid('gema_brillo', (0, 0, 0), (0.008, 0.008, 0.004), coll=C_FX, seg=12, rings=6)
    core.matrix_world = Matrix.Translation(c + n * 0.024 + Vector((-0.005, 0, 0.005))) @ rot
    core.data.materials.append(M['gem_core'])
    smooth(gem)
    return ob

MED = medallion(MED_C, mnor)

# cadena: del medallón sube por el pecho hasta perderse bajo el cuello
def chain():
    cu = bpy.data.curves.new('cadena', 'CURVE')
    cu.dimensions = '3D'; cu.bevel_depth = 0.0035; cu.bevel_resolution = 2
    for sx in (-1, 1):
        pts = [MED_C + Vector((sx * 0.02, 0, 0.05))]
        for (x, z) in ((0.045, 1.40), (0.075, 1.46), (0.10, 1.51)):
            l, n_, _ = surface_hit(sx * x, z)
            pts.append(l + Vector((0, -0.006, 0)))
        sp = cu.splines.new('NURBS')
        sp.points.add(len(pts) - 1)
        for k, p in enumerate(pts):
            sp.points[k].co = (p.x, p.y, p.z, 1)
        sp.use_endpoint_u = True
        sp.order_u = 3
    ob = bpy.data.objects.new('cadena', cu)
    C_CHAR.objects.link(ob)
    ob.data.materials.append(M['gold'])
    return ob

MED.hide_viewport = True
chain()
MED.hide_viewport = False
for e in EYES + [FACE]:
    e.hide_viewport = False

# ---------------------------------------------------------------- terminación de la tela: grosor + subdivisión
def finish(ob, thick=0.008, sub=1):
    so = ob.modifiers.new('Grosor', 'SOLIDIFY')
    so.thickness = thick
    so.offset = -1
    so.use_rim = True
    so.material_offset = 1
    so.material_offset_rim = 0
    so.use_even_offset = False
    sb = ob.modifiers.new('Sub', 'SUBSURF')
    sb.levels = 1
    sb.render_levels = sub

for ob in (CLOAK,):
    finish(ob, 0.009, 1)
for ob in SLEEVES + [COWL]:
    finish(ob, 0.007, 1)
finish(HOOD, 0.012, 2)
finish(ROBE, 0.006, 1)
for ob in RIBBONS:
    so = ob.modifiers.new('Grosor', 'SOLIDIFY'); so.thickness = 0.004
    sb = ob.modifiers.new('Sub', 'SUBSURF'); sb.levels = 1; sb.render_levels = 2
for ob in (SASH, BELT):
    sb = ob.modifiers.new('Sub', 'SUBSURF'); sb.levels = 1; sb.render_levels = 1
for ob in list(body.values()) + arms:
    ob.hide_viewport = True
    ob.hide_render = True
for ob in (ROBE, SLEEVES[0], SLEEVES[1], CLOAK, SASH, KNOT):
    for m in list(ob.modifiers):
        if m.type == 'COLLISION':
            ob.modifiers.remove(m)
# el piso se queda con su colisión (no molesta)

# ---------------------------------------------------------------- fondo: círculo de runas, piso, estrellas
def rune_circle(name, R, r_in, mat, n_runes=24, glyph=0.05, extra_rings=(), star=0, coll=C_FX, bevel=0.0035):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'; cu.bevel_depth = bevel; cu.bevel_resolution = 1
    def circle(rad, n=160):
        sp = cu.splines.new('POLY'); sp.points.add(n - 1)
        for k in range(n):
            a = 2 * math.pi * k / n
            sp.points[k].co = (rad * math.cos(a), rad * math.sin(a), 0, 1)
        sp.use_cyclic_u = True
    circle(R); circle(r_in)
    for rr in extra_rings:
        circle(rr)
    rng = random.Random(len(name))
    rm = (R + r_in) / 2
    for k in range(n_runes):
        a = 2 * math.pi * k / n_runes
        g = rng.choice(RUNES)
        up = Vector((math.cos(a), math.sin(a), 0))
        rt = Vector((math.sin(a), -math.cos(a), 0))
        c = up * rm
        for (x0, y0, x1, y1) in g:
            p0 = c + rt * ((x0 - 0.5) * glyph * 0.8) + up * ((y0 - 0.5) * glyph)
            p1 = c + rt * ((x1 - 0.5) * glyph * 0.8) + up * ((y1 - 0.5) * glyph)
            sp = cu.splines.new('POLY'); sp.points.add(1)
            sp.points[0].co = (*p0, 1); sp.points[1].co = (*p1, 1)
    if star:
        # marcas radiales y puntos entre los anillos interiores (sin polígonos estrellados)
        inner = min(extra_rings) if extra_rings else r_in * 0.7
        for k in range(star):
            a = 2 * math.pi * k / star + math.pi / 2
            u = Vector((math.cos(a), math.sin(a), 0))
            L = (r_in - inner) * (0.55 if k % 2 == 0 else 0.3)
            sp = cu.splines.new('POLY'); sp.points.add(1)
            sp.points[0].co = (*(u * (r_in - 0.008)), 1); sp.points[1].co = (*(u * (r_in - 0.008 - L)), 1)
        for rr in (inner + (r_in - inner) * 0.25,):
            n = 96
            for k in range(0, n, 2):
                a0 = 2 * math.pi * k / n; a1 = 2 * math.pi * (k + 1) / n
                sp = cu.splines.new('POLY'); sp.points.add(1)
                sp.points[0].co = (rr * math.cos(a0), rr * math.sin(a0), 0, 1); sp.points[1].co = (rr * math.cos(a1), rr * math.sin(a1), 0, 1)
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    ob.data.materials.append(mat)
    return ob

# cámaras (definen hacia dónde mira el círculo)
AZ = math.radians(28)
def cam(name, target, dist, height, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_fit = 'VERTICAL'
    cd.sensor_height = 36.0
    co = bpy.data.objects.new(name, cd)
    scene.collection.objects.link(co)
    tgt = Vector(target)
    co.location = Vector((tgt.x + dist * math.sin(AZ), tgt.y - dist * math.cos(AZ), height))
    co.rotation_euler = (tgt - co.location).to_track_quat('-Z', 'Y').to_euler()
    return co

CAM_FULL = cam('cam_cuerpo', (0, 0, 1.06), 3.55, 0.95, 50)
CAM_CLOSE = cam('cam_capucha', (0, 0, 1.6), 2.3, 1.52, 85)
CAM_SIDE = cam('cam_costado', (0, 0, 1.0), 4.2, 1.0, 45)
CAM_SIDE.location = Vector((4.2, -0.4, 1.0))
CAM_SIDE.rotation_euler = (Vector((0, 0, 1.0)) - CAM_SIDE.location).to_track_quat('-Z', 'Y').to_euler()

viewdir = Vector((math.sin(AZ), -math.cos(AZ), 0))
HALO = rune_circle('circulo_runas', 0.74, 0.66, M['rune'], n_runes=28, glyph=0.055, extra_rings=(0.40,), star=24)
HALO.location = Vector((0, 0, 1.64)) - viewdir * 0.75
HALO.rotation_euler = viewdir.to_track_quat('Z', 'Y').to_euler()
GROUND = rune_circle('circulo_piso', 1.05, 0.96, M['rune_floor'], n_runes=36, glyph=0.07, extra_rings=(0.70,), star=36, bevel=0.005)
GROUND.location = (0, 0.05, 0.004)

# telón con resplandor y estrellas, perpendicular a la vista
def backdrop():
    bpy.ops.mesh.primitive_plane_add(size=1)
    ob = bpy.context.active_object
    ob.name = 'telon'
    for c in ob.users_collection:
        c.objects.unlink(ob)
    C_FX.objects.link(ob)
    ob.scale = (30, 30, 1)
    head = Vector((0, 0, 1.6))
    dirc = (head - CAM_FULL.location).normalized()
    ob.location = head + dirc * 7.0
    ob.rotation_euler = (-dirc).to_track_quat('Z', 'Y').to_euler()
    mat = bpy.data.materials.new('telon')
    nt = nodes_clear(mat)
    out = N(nt, 'ShaderNodeOutputMaterial')
    tc = N(nt, 'ShaderNodeTexCoord')
    mp = N(nt, 'ShaderNodeMapping'); mp.inputs['Scale'].default_value = (30 / 5.0, 30 / 6.0, 1)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    gr = N(nt, 'ShaderNodeTexGradient', gradient_type='SPHERICAL')
    nt.links.new(mp.outputs[0], gr.inputs['Vector'])
    ramp = N(nt, 'ShaderNodeValToRGB')
    el = ramp.color_ramp.elements
    el[0].position = 0.0; el[0].color = (0.003, 0.004, 0.012, 1)
    el[1].position = 0.92; el[1].color = (0.06, 0.12, 0.34, 1)
    e2 = el.new(0.55); e2.color = (0.012, 0.018, 0.06, 1)
    e3 = el.new(0.995); e3.color = (0.16, 0.30, 0.65, 1)
    nt.links.new(gr.outputs['Fac'], ramp.inputs['Fac'])
    # estrellas
    vo = N(nt, 'ShaderNodeTexVoronoi'); vo.inputs['Scale'].default_value = 140.0
    nt.links.new(tc.outputs['Object'], vo.inputs['Vector'])
    st = N(nt, 'ShaderNodeMapRange'); st.inputs['From Min'].default_value = 0.06; st.inputs['From Max'].default_value = 0.0
    nt.links.new(vo.outputs['Distance'], st.inputs['Value'])
    sep = N(nt, 'ShaderNodeSeparateColor'); nt.links.new(vo.outputs['Color'], sep.inputs[0])
    gt = N(nt, 'ShaderNodeMath', operation='GREATER_THAN'); gt.inputs[1].default_value = 0.78
    nt.links.new(sep.outputs[0], gt.inputs[0])
    sm = N(nt, 'ShaderNodeMath', operation='MULTIPLY')
    nt.links.new(st.outputs[0], sm.inputs[0]); nt.links.new(gt.outputs[0], sm.inputs[1])
    sm2 = N(nt, 'ShaderNodeMath', operation='MULTIPLY'); sm2.inputs[1].default_value = 1.6
    nt.links.new(sm.outputs[0], sm2.inputs[0])
    add = N(nt, 'ShaderNodeMixRGB', blend_type='ADD'); add.inputs['Fac'].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], add.inputs['Color1'])
    comb = N(nt, 'ShaderNodeCombineXYZ')
    for k in range(3):
        nt.links.new(sm2.outputs[0], comb.inputs[k])
    nt.links.new(comb.outputs[0], add.inputs['Color2'])
    em = N(nt, 'ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0
    nt.links.new(add.outputs[0], em.inputs['Color'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob

backdrop()

# chispas flotando
def sparks(n=110):
    bm = bmesh.new()
    rng = random.Random(3)
    k = 0
    while k < n:
        x = rng.uniform(-1.6, 1.6); y = rng.uniform(-0.8, 2.2); z = rng.uniform(0.05, 2.6)
        if abs(x) < 0.55 and -0.6 < y < 0.6:
            continue
        r = rng.uniform(0.003, 0.009)
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
        bmesh.ops.translate(bm, vec=Vector((x, y, z)), verts=res['verts'])
        k += 1
    me = bpy.data.meshes.new('chispas')
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new('chispas', me)
    C_FX.objects.link(ob)
    ob.data.materials.append(M['spark'])
    ob.visible_shadow = False
    return ob

sparks()

# ---------------------------------------------------------------- luces
def light(name, typ, loc, energy, color, size=0.5, target=(0, 0, 1.2)):
    ld = bpy.data.lights.new(name, typ)
    ld.energy = energy
    ld.color = color
    if typ == 'AREA':
        ld.size = size
    elif typ in ('POINT', 'SPOT'):
        ld.shadow_soft_size = size
    elif typ == 'SUN':
        ld.angle = math.radians(2)
    lo = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(lo)
    lo.location = loc
    lo.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return lo

light('luna', 'SUN', (-2.0, -2.2, 3.6), 2.2, (0.72, 0.80, 1.0), target=(0, 0, 1.0))
light('borde_cian', 'AREA', (-1.7, 1.9, 2.3), 420, (0.30, 0.85, 1.0), size=1.2, target=(0, 0, 1.3))
light('borde_violeta', 'AREA', (1.8, 1.7, 1.9), 330, (0.62, 0.38, 1.0), size=1.2, target=(0, 0, 1.2))
light('relleno', 'AREA', (2.2, -2.5, 1.2), 70, (0.35, 0.45, 0.9), size=2.0, target=(0, 0, 1.1))
light('piso_glow', 'POINT', (0, -0.2, 0.15), 25, (0.25, 0.75, 1.0), size=0.6)
light('gema_luz', 'POINT', tuple(MED_C + mnor * 0.06), 0.9, (0.25, 0.6, 1.0), size=0.02)
light('ojos_luz', 'POINT', (0, -0.115, 1.665), 0.25, (0.4, 0.9, 1.0), size=0.02)

world = bpy.data.worlds.new('noche')
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Color'].default_value = (0.004, 0.005, 0.014, 1)
bg.inputs['Strength'].default_value = 1.0

# ---------------------------------------------------------------- render
scene.render.engine = 'CYCLES'
cy = scene.cycles
cy.device = 'CPU'
cy.samples = args.samples
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.03
cy.use_denoising = True
try:
    cy.denoiser = 'OPENIMAGEDENOISE'
except Exception:
    pass
cy.max_bounces = 4
cy.diffuse_bounces = 2
cy.glossy_bounces = 2
cy.transmission_bounces = 0
cy.volume_bounces = 0
cy.transparent_max_bounces = 4
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.blur_glossy = 1.0
scene.render.resolution_x = 1080
scene.render.resolution_y = 1920
scene.render.resolution_percentage = args.scale
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0.0

# Freestyle: contorno solo en el personaje
scene.render.use_freestyle = True
scene.render.line_thickness_mode = 'ABSOLUTE'
scene.render.line_thickness = 1.0
vl = scene.view_layers[0]
vl.use_freestyle = True
fs = vl.freestyle_settings
fs.crease_angle = math.radians(128)
ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new('contorno')
ls.select_by_visibility = True
ls.visibility = 'VISIBLE'
ls.select_by_edge_types = True
ls.select_silhouette = True
ls.select_border = True
ls.select_crease = True
ls.select_external_contour = True
ls.select_by_collection = True
ls.collection = C_CHAR
lst = ls.linestyle or bpy.data.linestyles.new('linea')
ls.linestyle = lst
lst.color = (0.012, 0.010, 0.030)
lst.thickness = 2.0
lst.thickness_position = 'CENTER'
lst.chaining = 'PLAIN'
lst.use_chain_count = False

# compositor: brillo suave (ojos, gema, runas)
scene.use_nodes = True
ctree = scene.node_tree
for n in list(ctree.nodes):
    ctree.nodes.remove(n)
rl = ctree.nodes.new('CompositorNodeRLayers')
gl = ctree.nodes.new('CompositorNodeGlare')
gl.glare_type = 'FOG_GLOW'
gl.quality = 'HIGH'
gl.size = 8
gl.threshold = 0.9
gl.mix = 0.0
comp = ctree.nodes.new('CompositorNodeComposite')
ctree.links.new(rl.outputs['Image'], gl.inputs['Image'])
ctree.links.new(gl.outputs['Image'], comp.inputs['Image'])

log('escena lista; guardando .blend')
try:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'encapuchado.blend'), compress=True)
except Exception as e:
    log('no se pudo guardar el .blend:', e)

VIEWS = {
    'full': (CAM_FULL, 'encapuchado_cuerpo.png', 2.0, 100),
    'close': (CAM_CLOSE, 'encapuchado_capucha.png', 2.6, 100),
    'side': (CAM_SIDE, 'debug_costado.png', 1.5, 50),
}
for v in args.views.split(','):
    v = v.strip()
    if v not in VIEWS:
        continue
    camo, fname, thick, pct = VIEWS[v]
    scene.camera = camo
    lst.thickness = thick * args.scale / 100.0 * pct / 100.0
    scene.render.resolution_percentage = int(args.scale * pct / 100)
    cy.samples = args.samples if v != 'side' else max(16, args.samples // 3)
    scene.render.filepath = os.path.join(OUT, fname)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    log('render %s en %.1fs -> %s' % (v, time.time() - t, scene.render.filepath))

log('fin')
