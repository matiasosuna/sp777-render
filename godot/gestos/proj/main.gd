extends Node2D
# Encapuchado con gestos (fase 2, Godot 4.7 Compatibility). Lee linea.json: escenas (fondo, cuerpo de pie/sentado,
# encuadre, efectos) y acciones con tiempos; la biblioteca de gestos (gestos.gd) da la pose de cada instante y el
# títere por capas (rig.gd) la aplica. Se renderiza con Movie Maker en GitHub Actions (sp777-render, godot.yml).
# Argumentos de usuario (después de --): debug (dibuja huesos), t=SEGUNDOS (arranca en ese instante).

const W := 1080.0
const H := 1920.0
const FPS := 30.0
const RigScript := preload("res://rig.gd")
const GestosScript := preload("res://gestos.gd")

var frame := 0
var debug := false
var dir := ""
var L: Dictionary
var T: Dictionary
var gest
var rig
var cam: Camera2D
var fondos := {}
var frentes := {}
var perso: Node2D
var anillo: Sprite2D
var anillo_glow: Sprite2D
var sombra: Sprite2D
var cuadro: Node2D
var velas := []
var sol: Sprite2D
var part: Node2D
var fundido: ColorRect
var glows := []
var luz_pecho: Sprite2D
var add_mat: CanvasItemMaterial
var dot: ImageTexture
var esc_actual := -1
var cuadrado_max := 0.0
var estado := {}


class FX extends Node2D:
	var items: Array = []
	var t := 0.0
	var dot: Texture2D

	func _draw() -> void:
		for it in items:
			var f: float = fposmod(t / it.period + it.ph, 1.0)
			var p := Vector2(it.x0 + it.sway * sin(TAU * it.k * f + it.psi), it.y0 - it.dist * f)
			var a: float = it.a0 * pow(sin(PI * f), 1.4) * (0.75 + 0.25 * sin(TAU * 3.0 * f + it.psi))
			var s: float = it.size
			draw_texture_rect(dot, Rect2(p - Vector2(s, s) * 0.5, Vector2(s, s)), false, Color(it.col, a))


class Cuadro extends Node2D:
	# cuadrado de luz (coords del personaje)
	var esquinas: Array = []
	var prog := 0.0
	var alfa := 0.0
	var punto := Vector2.ZERO
	var chispa := 0.0
	var dot: Texture2D

	func _draw() -> void:
		if alfa <= 0.01:
			return
		for i in 4:
			var f := clampf(prog - i, 0.0, 1.0)
			if f <= 0.0:
				break
			var a: Vector2 = esquinas[i]
			var b: Vector2 = a.lerp(esquinas[(i + 1) % 4], f)
			draw_line(a, b, Color(1.0, 0.8, 0.4, 0.16 * alfa), 34.0)
			draw_line(a, b, Color(1.0, 0.85, 0.5, 0.35 * alfa), 14.0)
			draw_line(a, b, Color(1.0, 0.97, 0.85, 0.95 * alfa), 4.5)
			draw_texture_rect(dot, Rect2(a - Vector2(30, 30), Vector2(60, 60)), false, Color(1.0, 0.9, 0.6, 0.5 * alfa))
		if chispa > 0.01:
			draw_texture_rect(dot, Rect2(punto - Vector2(70, 70), Vector2(140, 140)), false, Color(1.0, 0.85, 0.5, 0.55 * chispa * alfa))
			draw_texture_rect(dot, Rect2(punto - Vector2(22, 22), Vector2(44, 44)), false, Color(1.0, 1.0, 0.95, 0.9 * chispa * alfa))


func _ready() -> void:
	var ua := OS.get_cmdline_user_args()
	debug = "debug" in ua
	for a in ua:
		if a.begins_with("t="):
			frame = int(round(float(a.substr(2)) * FPS))
	dir = ProjectSettings.globalize_path("res://capas/")
	L = JSON.parse_string(FileAccess.get_file_as_string(dir + "layout.json"))
	T = JSON.parse_string(FileAccess.get_file_as_string(ProjectSettings.globalize_path("res://linea.json")))
	gest = GestosScript.new()
	add_mat = CanvasItemMaterial.new()
	add_mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	dot = radial_tex(64, 1.5)

	cam = Camera2D.new()
	cam.position = Vector2(W, H) * 0.5
	add_child(cam)
	cam.make_current()

	# fondos
	var fn := Node2D.new()
	add_child(fn)
	for e in T.escenas:
		var nom: String = e.fondo
		if not fondos.has(nom):
			var s := Sprite2D.new()
			var ruta := dir + nom + ".jpg"
			s.texture = tex(ruta if FileAccess.file_exists(ruta) else dir + "noche.jpg")
			s.centered = false
			s.visible = false
			fn.add_child(s)
			fondos[nom] = s

	# sol (detrás del personaje)
	sol = Sprite2D.new()
	sol.texture = radial_tex(128, 2.4)
	sol.material = add_mat
	var so: Array = T.get("sol", [540, 1000, 500])
	sol.position = Vector2(so[0], so[1])
	sol.scale = Vector2.ONE * float(so[2]) / 128.0
	sol.modulate = Color(1.0, 0.8, 0.55, 0.0)
	add_child(sol)

	# personaje (anillo detrás, sombra, títere, cuadrado de luz)
	perso = Node2D.new()
	add_child(perso)
	var an: Dictionary = L.anillo
	anillo_glow = Sprite2D.new()
	anillo_glow.texture = tex(dir + "anillo_glow.png")
	anillo_glow.position = Vector2(an.cx, an.cy)
	anillo_glow.material = add_mat
	perso.add_child(anillo_glow)
	anillo = Sprite2D.new()
	anillo.texture = tex(dir + "anillo.png")
	anillo.position = Vector2(an.cx, an.cy)
	anillo.material = add_mat
	perso.add_child(anillo)
	sombra = Sprite2D.new()
	sombra.texture = radial_tex(128, 1.6)
	sombra.modulate = Color(0.02, 0.02, 0.06, 0.5)
	sombra.position = Vector2(549, 1792)
	sombra.scale = Vector2(5.6, 0.75)
	perso.add_child(sombra)
	rig = RigScript.new()
	perso.add_child(rig)
	rig.construir(dir, L.piezas, T.get("rig", {}))
	# brillos de ojos y gema (hijos de cabeza y pecho)
	var soft := radial_tex(128, 2.4)
	var core := radial_tex(64, 1.4)
	for o in L.ojos:
		var hg: Vector2 = rig.G["head"]
		var pos: Vector2 = Vector2(o[0], o[1]) - hg
		add_glow(rig.B["head"], soft, pos, 70.0, Color(0.35, 0.9, 1.0), 0.42, 0.22, 0.0)
		add_glow(rig.B["head"], core, pos, 20.0, Color(0.7, 1.0, 1.0), 0.55, 0.25, 0.0)
	var cg: Vector2 = rig.G["chest"]
	var gp: Vector2 = Vector2(L.gema[0], L.gema[1]) - cg
	add_glow(rig.B["chest"], soft, gp, 190.0, Color(0.3, 0.85, 1.0), 0.30, 0.18, 0.9)
	add_glow(rig.B["chest"], core, gp, 56.0, Color(0.65, 1.0, 1.0), 0.45, 0.25, 0.9)
	luz_pecho = Sprite2D.new()
	luz_pecho.texture = soft
	luz_pecho.material = add_mat
	luz_pecho.position = gp
	luz_pecho.modulate = Color(1.0, 0.85, 0.55, 0.0)
	luz_pecho.z_index = 1
	rig.B["chest"].add_child(luz_pecho)
	cuadro = Cuadro.new()
	cuadro.esquinas = gest.esquinas()
	cuadro.dot = dot
	cuadro.material = add_mat
	cuadro.z_index = 10
	perso.add_child(cuadro)

	# primeros planos: <pieza>_frente.png entre el cuerpo y los brazos (z 0, después del personaje),
	# <pieza>_alto.png delante de todo (z 20)
	for e in T.escenas:
		if e.has("frente"):
			var nom: String = e.frente
			if frentes.has(nom):
				continue
			frentes[nom] = []
			for suf in ["_frente", "_alto"]:
				var ruta: String = dir + nom + String(suf) + ".png"
				if FileAccess.file_exists(ruta):
					var s := Sprite2D.new()
					s.texture = tex(ruta)
					s.centered = false
					s.visible = false
					s.z_index = 0 if suf == "_frente" else 20
					add_child(s)
					frentes[nom].append(s)

	# velas y sol (luz sumada)
	for v in T.get("velas", []):
		var s := Sprite2D.new()
		s.texture = soft
		s.material = add_mat
		s.position = Vector2(v[0], v[1])
		s.scale = Vector2.ONE * float(v[2]) / 128.0
		s.modulate = Color(1.0, 0.7, 0.35, 0.0)
		s.z_index = 21
		add_child(s)
		velas.append(s)

	# partículas
	part = FX.new()
	part.dot = dot
	part.material = add_mat
	part.z_index = 21
	var rng := RandomNumberGenerator.new()
	rng.seed = 777
	var per := [8.0, 6.0, 5.0]
	for i in 40:
		var gold := rng.randf() < 0.65
		part.items.append({x0 = rng.randf_range(60.0, 1020.0), y0 = rng.randf_range(1300.0, 2000.0),
			dist = rng.randf_range(700.0, 1300.0), period = per[rng.randi_range(0, 2)], ph = rng.randf(),
			sway = rng.randf_range(6.0, 22.0), k = float(rng.randi_range(1, 2)), psi = rng.randf() * TAU,
			a0 = rng.randf_range(0.5, 0.95), size = rng.randf_range(10.0, 26.0),
			col = Color(1.0, 0.82, 0.45) if gold else Color(0.5, 0.92, 1.0)})
	add_child(part)

	if debug:
		var dn := Node2D.new()
		dn.z_index = 50
		dn.draw.connect(_draw_dbg.bind(dn))
		rig.add_child(dn)
		dn.set_meta("dbg", true)
		rig.dbg = dn

	var cl := CanvasLayer.new()
	cl.layer = 5
	add_child(cl)
	fundido = ColorRect.new()
	fundido.color = Color(0, 0, 0, 1)
	fundido.size = Vector2(W, H)
	cl.add_child(fundido)
	paso(frame / FPS)


func _draw_dbg(n: Node2D) -> void:
	for b in rig.RIG:
		var bn: Bone2D = rig.B[b[0]]
		var p: Vector2 = rig.skel.to_local(bn.global_position)
		if b[1] != "":
			var q: Vector2 = rig.skel.to_local(rig.B[b[1]].global_position)
			n.draw_line(q, p, Color.WHITE, 3.0)
		n.draw_circle(p, 8.0, Color.RED)
	for s in ["L", "R"]:
		n.draw_circle(rig.punta(s), 9.0, Color.YELLOW)
		n.draw_circle(Vector2(estado["mu" + s]), 7.0, Color.CYAN)


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


func add_glow(parent: Node2D, t: Texture2D, pos: Vector2, size: float, col: Color, a0: float, amp: float, ph: float) -> void:
	var s := Sprite2D.new()
	s.texture = t
	s.material = add_mat
	s.position = pos
	s.scale = Vector2.ONE * size / t.get_width()
	s.modulate = col
	s.z_index = 4
	parent.add_child(s)
	glows.append([s, a0, amp, ph])


func escena(t: float) -> int:
	var es: Array = T.escenas
	for i in es.size():
		if t < float(es[i].a):
			return i
	return es.size() - 1


func paso(t: float) -> void:
	var es: Array = T.escenas
	var i := escena(t)
	var e: Dictionary = es[i]
	var de: float = e.de
	var a: float = e.a
	var u := clampf((t - de) / maxf(a - de, 0.01), 0.0, 1.0)
	if i != esc_actual:
		esc_actual = i
		cuadrado_max = 0.0
		for k in fondos:
			fondos[k].visible = (k == e.fondo)
		for k in frentes:
			for f in frentes[k]:
				f.visible = e.has("frente") and k == e.frente
		var sen := String(e.get("cuerpo", "pie")) == "sentado"
		rig.sentado(sen)
		sombra.position = Vector2(549, 1330 if sen else 1792)
	var efx: Array = e.get("efectos", [])

	# encuadre del personaje (+ vaivén del bote)
	var esc: float = e.esc
	var ancla := Vector2(e.ancla[0], e.ancla[1])
	var en := Vector2(e.en[0], e.en[1])
	var bob := Vector2.ZERO
	var rot := 0.0
	if "bote" in efx:
		bob = Vector2(0, 7.0 * sin(TAU * t / 3.4))
		rot = deg_to_rad(0.7) * sin(TAU * t / 3.4 + 0.8)
	perso.scale = Vector2.ONE * esc
	perso.rotation = rot
	perso.position = en - (ancla * esc).rotated(rot) + bob
	if e.has("frente") and frentes.has(e.frente):
		for f in frentes[e.frente]:
			f.rotation = rot
			f.position = en + bob - en.rotated(rot)
	var ti: Array = e.get("tinte", [1, 1, 1])
	perso.modulate = Color(ti[0], ti[1], ti[2])
	sombra.visible = not ("bote" in efx) and not e.has("frente")

	# cámara: acercamiento lento en cada escena
	var z: Array = e.get("zoom", [1.0, 1.04])
	var zz := lerpf(float(z[0]), float(z[1]), u * u * (3.0 - 2.0 * u))
	cam.zoom = Vector2.ONE * zz

	# gestos
	var base: Dictionary = gest.POSES[e.get("base", "de_pie")]
	# solo las acciones de esta escena (al cortar de escena el cuerpo arranca quieto en la base nueva)
	var acc := []
	for ac in T.acciones:
		if float(ac[1]) >= de - 0.01 and float(ac[1]) < a:
			acc.append(ac)
	estado = gest.estado(t, acc, base)
	rig.aplicar(estado, t)
	if rig.dbg != null:
		rig.dbg.queue_redraw()

	# efectos
	var w := TAU / 4.0
	var hay_anillo := "anillo" in efx
	anillo.visible = hay_anillo
	anillo_glow.visible = hay_anillo
	var pul := 0.5 + 0.5 * sin(w * t - 1.0)
	anillo.rotation = t * deg_to_rad(4.0)
	anillo_glow.rotation = anillo.rotation
	anillo.modulate = Color(1, 1, 1, 0.9 + 0.1 * pul)
	anillo_glow.modulate = Color(1, 1, 1, 0.18 + 0.3 * pul)
	for g in glows:
		var p: float = 0.5 + 0.5 * sin(2.0 * w * t + float(g[3]))
		g[0].modulate.a = g[1] + g[2] * p
	var lp: float = estado["luz_pecho"]
	luz_pecho.modulate.a = 0.75 * lp
	luz_pecho.scale = Vector2.ONE * (2.2 + 1.6 * lp)
	cuadrado_max = maxf(cuadrado_max, float(estado["cuadrado"]))
	cuadro.prog = cuadrado_max
	cuadro.alfa = float(estado["cuadrado_a"]) if "cuadrado" in efx else 0.0
	if cuadro.alfa < 0.02:
		cuadro.alfa = 0.0
	cuadro.punto = estado["apunta"]
	cuadro.chispa = estado["chispa"]
	cuadro.queue_redraw()
	var hay_velas := "velas" in efx
	for k in velas.size():
		var v: Sprite2D = velas[k]
		v.visible = hay_velas
		v.modulate.a = 0.55 + 0.12 * sin(11.0 * t + k * 1.7) + 0.07 * sin(23.0 * t + k)
	sol.visible = "sol" in efx
	sol.modulate.a = 0.45 + 0.08 * sin(TAU * t / 5.0)
	part.visible = "particulas" in efx
	part.t = t
	part.queue_redraw()

	# fundidos: negro al empezar, al terminar y en cada cambio de escena
	var fa := 0.0
	var bordes := [0.0, float(es[es.size() - 1].a)]
	for k in range(es.size() - 1):
		bordes.append(float(es[k].a))
	for b in bordes:
		var d := absf(t - float(b)) / 0.35
		if d < 1.0:
			fa = maxf(fa, 1.0 - d * d * (3.0 - 2.0 * d))
	fundido.color.a = fa


func _process(_d: float) -> void:
	paso(frame / FPS)
	frame += 1
