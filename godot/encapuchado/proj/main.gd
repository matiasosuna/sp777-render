extends Node2D
# Encapuchado en Godot 4.7 (renderer Compatibility), 17/11/2026.
# La ilustración de FLUX recortada va en un solo Polygon2D (malla en grilla, vértices compartidos: sin costuras)
# con pesos a un Skeleton2D. Todo se mueve con funciones periódicas de 8 s: el cuadro 240 es igual al 0 (loop perfecto).
# Las capas (personaje, fondo, anillo, layout.json) las arma prep.py en GitHub Actions.
# Argumento de usuario "debug" (después de --): pinta encima los pesos (color del hueso dominante) y los huesos.

const W := 1080.0
const H := 1920.0
const LOOP := 8.0
const FPS := 30.0
const CELL := 16

var frame := 0
var debug := false
var L: Dictionary
var world: Node2D
var cam: Camera2D
var skel: Skeleton2D
var B := {}          # nombre -> Bone2D
var G := {}          # nombre -> pivote global en reposo
var RL := {}         # nombre -> posición local en reposo
var ring: Sprite2D
var ring_glow: Sprite2D
var glows := []      # [Sprite2D, alfa base, amplitud, fase]
var back_fx: FX
var front_fx: FX
var add_mat: CanvasItemMaterial


class FX extends Node2D:
	# Partículas y estrellas dibujadas a mano: cada una es función del tiempo con período que divide 8 s.
	var items: Array = []
	var t := 0.0
	var dot: Texture2D

	func _draw() -> void:
		for it in items:
			var p: Vector2
			var a: float
			if it.kind == 0:  # partícula que sube
				var f: float = fposmod(t / it.period + it.ph, 1.0)
				p = Vector2(it.x0 + it.sway * sin(TAU * it.k * f + it.psi), it.y0 - it.dist * f)
				a = it.a0 * pow(sin(PI * f), 1.4) * (0.75 + 0.25 * sin(TAU * 3.0 * f + it.psi))
			else:  # estrella que titila
				p = Vector2(it.x0, it.y0)
				a = it.a0 * (0.5 + 0.5 * sin(TAU * it.k * t / 8.0 + it.psi))
			var s: float = it.size
			draw_texture_rect(dot, Rect2(p - Vector2(s, s) * 0.5, Vector2(s, s)), false, Color(it.col, a))


func _ready() -> void:
	debug = "debug" in OS.get_cmdline_user_args()
	var dir := ProjectSettings.globalize_path("res://capas/")
	L = JSON.parse_string(FileAccess.get_file_as_string(dir + "layout.json"))
	add_mat = CanvasItemMaterial.new()
	add_mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD

	world = Node2D.new()
	add_child(world)
	cam = Camera2D.new()
	cam.position = Vector2(W, H) * 0.5
	add_child(cam)
	cam.make_current()

	# fondo
	var fondo := Sprite2D.new()
	fondo.texture = tex(dir + "fondo.jpg")
	fondo.centered = false
	world.add_child(fondo)

	# estrellas que titilan (detrás de todo)
	var rng := RandomNumberGenerator.new()
	rng.seed = 777
	var dot := radial_tex(64, 2.2)
	var stars := FX.new()
	stars.dot = dot
	stars.material = add_mat
	for s in L.estrellas:
		stars.items.append({kind = 1, x0 = float(s[0]), y0 = float(s[1]), k = float(rng.randi_range(1, 3)),
			psi = rng.randf() * TAU, a0 = 0.35 + 0.35 * clampf((float(s[2]) - 170.0) / 85.0, 0.0, 1.0),
			size = rng.randf_range(9.0, 16.0), col = Color(0.85, 0.92, 1.0)})
	world.add_child(stars)
	back_fx = stars

	# círculo de runas (luz sumada) y su brillo
	var an: Dictionary = L.anillo
	ring_glow = Sprite2D.new()
	ring_glow.texture = tex(dir + "anillo_glow.png")
	ring_glow.position = Vector2(an.cx, an.cy)
	ring_glow.material = add_mat
	world.add_child(ring_glow)
	ring = Sprite2D.new()
	ring.texture = tex(dir + "anillo.png")
	ring.position = Vector2(an.cx, an.cy)
	ring.material = add_mat
	world.add_child(ring)

	# partículas de atrás
	var part_back := FX.new()
	part_back.dot = dot
	part_back.material = add_mat
	add_particles(part_back, rng, 34, 0.9)
	world.add_child(part_back)

	# sombra en el piso
	var sombra := Sprite2D.new()
	sombra.texture = radial_tex(128, 1.6)
	sombra.modulate = Color(0.02, 0.02, 0.06, 0.55)
	sombra.position = Vector2(549, 1792)
	sombra.scale = Vector2(5.6, 0.75)
	world.add_child(sombra)

	# esqueleto y malla
	skel = Skeleton2D.new()
	skel.name = "Skel"
	build_rig()
	var poly := build_mesh(dir + "personaje.png")
	world.add_child(poly)
	world.add_child(skel)   # después de la malla: los brillos (hijos de huesos) quedan encima
	bind(poly)
	if debug:
		var dbg := build_debug(poly)
		world.add_child(dbg)
		bind(dbg)

	# brillos: ojos (hijos de la cabeza) y gema (hija del pecho)
	var soft := radial_tex(128, 2.4)
	var core := radial_tex(64, 1.4)
	for o in L.ojos:
		var hg: Vector2 = G["head"]
		var pos: Vector2 = Vector2(o[0], o[1]) - hg
		add_glow(B["head"], soft, pos, 70.0, Color(0.35, 0.9, 1.0), 0.42, 0.22, 0.0)
		add_glow(B["head"], core, pos, 20.0, Color(0.7, 1.0, 1.0), 0.55, 0.25, 0.0)
	var cg: Vector2 = G["chest"]
	var gp: Vector2 = Vector2(L.gema[0], L.gema[1]) - cg
	add_glow(B["chest"], soft, gp, 190.0, Color(0.3, 0.85, 1.0), 0.30, 0.18, 0.9)
	add_glow(B["chest"], core, gp, 56.0, Color(0.65, 1.0, 1.0), 0.45, 0.25, 0.9)

	# partículas de adelante
	front_fx = FX.new()
	front_fx.dot = dot
	front_fx.material = add_mat
	add_particles(front_fx, rng, 22, 1.0)
	world.add_child(front_fx)
	front_fx.set_meta("back", part_back)

	pose(0.0)


func tex(path: String) -> ImageTexture:
	return ImageTexture.create_from_image(Image.load_from_file(path))


func radial_tex(size: int, pw: float) -> ImageTexture:
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var c := (size - 1) * 0.5
	for y in size:
		for x in size:
			var r := clampf(Vector2(x - c, y - c).length() / c, 0.0, 1.0)
			img.set_pixel(x, y, Color(1, 1, 1, pow(1.0 - r, pw)))
	return ImageTexture.create_from_image(img)


func add_particles(fx: FX, rng: RandomNumberGenerator, n: int, sc: float) -> void:
	var periods := [8.0, 4.0, 8.0 / 3.0]
	for i in n:
		var gold := rng.randf() < 0.6
		var per: float = periods[rng.randi_range(0, 2)] if rng.randf() < 0.5 else 8.0
		fx.items.append({kind = 0, x0 = rng.randf_range(110.0, 990.0), y0 = rng.randf_range(1450.0, 1900.0),
			dist = rng.randf_range(450.0, 1000.0) * per / 8.0 + 200.0, period = per, ph = rng.randf(),
			sway = rng.randf_range(6.0, 22.0), k = float(rng.randi_range(1, 2)), psi = rng.randf() * TAU,
			a0 = rng.randf_range(0.35, 0.8), size = rng.randf_range(7.0, 20.0) * sc,
			col = Color(1.0, 0.82, 0.45) if gold else Color(0.5, 0.92, 1.0)})


func add_glow(parent: Node2D, t: Texture2D, pos: Vector2, size: float, col: Color, a0: float, amp: float, ph: float) -> void:
	var s := Sprite2D.new()
	s.texture = t
	s.material = add_mat
	s.position = pos
	s.scale = Vector2.ONE * size / t.get_width()
	s.modulate = col
	parent.add_child(s)
	glows.append([s, a0, amp, ph])


# ---------------- esqueleto ----------------
# [nombre, padre, pivote, cápsulas de influencia [a, b, radio]]
func rig_def() -> Array:
	var d := [
		["hips", "", Vector2(549, 900), [[Vector2(450, 1100), Vector2(450, 1780), 45.0], [Vector2(648, 1100), Vector2(648, 1780), 45.0], [Vector2(549, 900), Vector2(549, 1100), 70.0]]],
		["spine", "hips", Vector2(549, 880), [[Vector2(549, 860), Vector2(549, 775), 85.0]]],
		["chest", "spine", Vector2(549, 760), [[Vector2(549, 760), Vector2(549, 600), 110.0], [Vector2(430, 635), Vector2(668, 635), 40.0]]],
		["head", "chest", Vector2(549, 560), [[Vector2(549, 470), Vector2(549, 280), 115.0]]],
		["hem", "hips", Vector2(549, 1160), [[Vector2(549, 1200), Vector2(549, 1630), 40.0]]],
	]
	# lado izquierdo de la imagen (L) y su espejo (R)
	var side := [
		["arm", "chest", Vector2(345, 635), [[Vector2(335, 660), Vector2(268, 960), 60.0]]],
		["hand", "arm", Vector2(258, 1000), [[Vector2(235, 1025), Vector2(155, 1105), 35.0]]],
		["bell", "arm", Vector2(245, 990), [[Vector2(215, 1070), Vector2(160, 1500), 55.0]]],
		["cloak", "hips", Vector2(300, 1120), [[Vector2(300, 1150), Vector2(240, 1480), 35.0]]],
		["cloak2", "cloak", Vector2(240, 1500), [[Vector2(230, 1520), Vector2(90, 1730), 45.0]]],
	]
	for sfx in ["L", "R"]:
		for b in side:
			var par: String = b[1] if b[1] == "chest" or b[1] == "hips" else b[1] + sfx
			var caps := []
			for c in b[3]:
				caps.append([mx(c[0], sfx), mx(c[1], sfx), c[2]])
			d.append([b[0] + sfx, par, mx(b[2], sfx), caps])
	return d


func mx(v: Vector2, sfx: String) -> Vector2:
	return v if sfx == "L" else Vector2(1098.0 - v.x, v.y)


var RIG: Array

func build_rig() -> void:
	RIG = rig_def()
	for b in RIG:
		var bone := Bone2D.new()
		bone.name = b[0]
		var parent: Node = skel if b[1] == "" else B[b[1]]
		var pg: Vector2 = Vector2.ZERO if b[1] == "" else G[b[1]]
		parent.add_child(bone)
		bone.position = b[2] - pg
		bone.rest = Transform2D(0.0, bone.position)
		bone.set_autocalculate_length_and_angle(false)
		bone.set_length(40.0)
		B[b[0]] = bone
		G[b[0]] = b[2]
		RL[b[0]] = bone.position


func seg_dist(p: Vector2, a: Vector2, b: Vector2) -> float:
	var ab := b - a
	var t := clampf((p - a).dot(ab) / maxf(ab.length_squared(), 1e-6), 0.0, 1.0)
	return p.distance_to(a + ab * t)


var weights := {}
var pts := PackedVector2Array()

func build_mesh(path: String) -> Polygon2D:
	var img := Image.load_from_file(path)
	img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	var iw := img.get_width()
	var ih := img.get_height()
	var bb: Array = L.caja
	var gx0 := maxi(int(bb[0]) - 2 * CELL, 0)
	var gy0 := maxi(int(bb[1]) - 2 * CELL, 0)
	var nx := int(ceil((minf(float(bb[2]) + 2 * CELL, iw) - gx0) / CELL))
	var ny := int(ceil((minf(float(bb[3]) + 2 * CELL, ih) - gy0) / CELL))
	var occ := PackedByteArray()
	occ.resize(nx * ny)
	for j in ny:
		for i in nx:
			var hit := 0
			var y := gy0 + j * CELL
			while y < mini(gy0 + (j + 1) * CELL, ih) and hit == 0:
				var x := gx0 + i * CELL
				while x < mini(gx0 + (i + 1) * CELL, iw):
					if data[(y * iw + x) * 4 + 3] > 3:
						hit = 1
						break
					x += 2
				y += 2
			occ[j * nx + i] = hit
	# dilatar una celda
	var occ2 := occ.duplicate()
	for j in ny:
		for i in nx:
			if occ[j * nx + i] == 0:
				continue
			for dj in [-1, 0, 1]:
				for di in [-1, 0, 1]:
					var ii: int = i + di
					var jj: int = j + dj
					if ii >= 0 and jj >= 0 and ii < nx and jj < ny:
						occ2[jj * nx + ii] = 1
	var idx := {}
	var quads := []
	for j in ny:
		for i in nx:
			if occ2[j * nx + i] == 0:
				continue
			var q := PackedInt32Array()
			for c in [Vector2i(i, j), Vector2i(i + 1, j), Vector2i(i + 1, j + 1), Vector2i(i, j + 1)]:
				if not idx.has(c):
					idx[c] = pts.size()
					pts.append(Vector2(minf(gx0 + c.x * CELL, iw), minf(gy0 + c.y * CELL, ih)))
				q.append(idx[c])
			quads.append(q)
	print("malla: ", pts.size(), " vértices, ", quads.size(), " celdas")

	# pesos: inversa de la distancia a cápsulas (núcleo rígido + transición suave)
	var nv := pts.size()
	var flat := PackedFloat32Array()
	flat.resize(RIG.size() * nv)
	for vi in nv:
		var p := pts[vi]
		var ws := []
		var tot := 0.0
		for b in RIG:
			var d := 1e9
			for c in b[3]:
				d = minf(d, maxf(seg_dist(p, c[0], c[1]) - c[2], 0.0))
			var w := 1.0 / pow(d + 20.0, 3.0)
			ws.append(w)
			tot += w
		for bi in RIG.size():
			flat[bi * nv + vi] = ws[bi] / tot
	for bi in RIG.size():
		weights[RIG[bi][0]] = flat.slice(bi * nv, (bi + 1) * nv)

	var poly := Polygon2D.new()
	poly.texture = ImageTexture.create_from_image(img)
	poly.polygon = pts
	poly.uv = pts
	poly.polygons = quads
	return poly


func bind(poly: Polygon2D) -> void:
	poly.skeleton = poly.get_path_to(skel)
	for b in RIG:
		poly.add_bone(skel.get_path_to(B[b[0]]), weights[b[0]])


func build_debug(src: Polygon2D) -> Polygon2D:
	var dbg := Polygon2D.new()
	dbg.polygon = src.polygon
	dbg.polygons = src.polygons
	var cols := PackedColorArray()
	for vi in pts.size():
		var best := 0
		for bi in RIG.size():
			if weights[RIG[bi][0]][vi] > weights[RIG[best][0]][vi]:
				best = bi
		var c := Color.from_hsv(fposmod(best * 0.618, 1.0), 0.85, 1.0, 0.45)
		cols.append(c)
	dbg.vertex_colors = cols
	var lines := Node2D.new()
	lines.draw.connect(_draw_dbg.bind(lines))
	dbg.add_child(lines)
	return dbg


func _draw_dbg(lines: Node2D) -> void:
	for b in RIG:
		for c in b[3]:
			lines.draw_line(c[0], c[1], Color.WHITE, 3.0)
		lines.draw_circle(b[2], 7.0, Color.RED)


# ---------------- animación ----------------
func bump(t: float) -> float:
	# 0 en t=0 y t=8, 1 en t=4, con meseta suave (ofrenda: sube, sostiene, baja)
	var s := 0.5 - 0.5 * cos(TAU * t / LOOP)
	return s * s * (3.0 - 2.0 * s)


func setb(n: String, dp: Vector2, rot: float, sc := Vector2.ONE) -> void:
	var b: Bone2D = B[n]
	b.position = RL[n] + dp
	b.rotation = rot
	b.scale = sc


func pose(t: float) -> void:
	var w := TAU / LOOP
	var br := 0.5 - 0.5 * cos(2.0 * w * t)          # dos respiraciones por loop
	var e := bump(t)
	var e2 := bump(t - 0.35)
	var e3 := bump(t - 0.7)
	var D := PI / 180.0
	setb("spine", Vector2(0, -1.5 * br), 0.5 * D * sin(w * t + 0.4))
	setb("chest", Vector2(0, -3.0 * br), 0.0, Vector2(1.0 + 0.006 * br, 1.0 + 0.004 * br))
	setb("head", Vector2(0, -1.0 * br), 2.2 * D * sin(w * t + 0.3) + 0.6 * D * sin(2.0 * w * t + 1.0))
	var arm := 7.0 * D * e
	setb("armL", Vector2.ZERO, arm)
	setb("armR", Vector2.ZERO, -arm)
	setb("handL", Vector2.ZERO, 6.0 * D * e2)
	setb("handR", Vector2.ZERO, -6.0 * D * e2)
	# la manga campana cuelga: compensa el giro del brazo y llega tarde
	var bell := 7.0 * D * (-0.6 * e + 0.5 * (e3 - e)) + 1.0 * D * sin(w * t - 1.2)
	setb("bellL", Vector2.ZERO, bell)
	setb("bellR", Vector2.ZERO, -bell + 0.4 * D * sin(w * t - 0.5))
	setb("cloakL", Vector2.ZERO, 0.8 * D * sin(w * t - 1.0) + 0.4 * D * sin(2.0 * w * t - 0.3))
	setb("cloak2L", Vector2.ZERO, 1.8 * D * sin(w * t - 1.7) + 0.6 * D * sin(2.0 * w * t - 1.0))
	setb("cloakR", Vector2.ZERO, -0.8 * D * sin(w * t - 1.3) - 0.4 * D * sin(2.0 * w * t - 0.6))
	setb("cloak2R", Vector2.ZERO, -1.8 * D * sin(w * t - 2.0) - 0.6 * D * sin(2.0 * w * t - 1.3))
	setb("hem", Vector2.ZERO, 0.7 * D * sin(w * t - 2.0) + 0.3 * D * sin(2.0 * w * t - 0.6))

	# cámara: acercamiento suave y vuelta (periódico)
	var z := 0.5 - 0.5 * cos(w * t)
	cam.zoom = Vector2.ONE * (1.0 + 0.035 * z)
	cam.position = Vector2(W * 0.5, H * 0.5 - 25.0 * z)

	# anillo: gira 360/k grados por loop; pulso de brillo
	var k: float = float(L.anillo.k)
	var pul := 0.5 + 0.5 * sin(2.0 * w * t - 1.0)
	ring.rotation = TAU / k * (t / LOOP)
	ring_glow.rotation = ring.rotation
	ring.modulate = Color(1, 1, 1, 0.9 + 0.1 * pul)
	ring_glow.modulate = Color(1, 1, 1, 0.18 + 0.3 * pul)

	for g in glows:
		var p: float = 0.5 + 0.5 * sin(2.0 * w * t + float(g[3]))
		g[0].modulate.a = g[1] + g[2] * p

	for fx in [back_fx, front_fx, front_fx.get_meta("back")]:
		fx.t = t
		fx.queue_redraw()


func _process(_d: float) -> void:
	pose(frame / FPS)
	frame += 1
