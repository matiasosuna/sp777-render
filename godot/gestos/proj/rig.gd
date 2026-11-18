extends Node2D
# Títere por capas del encapuchado (fase 2). Coordenadas del personaje = píxeles de la ilustración a 1080x1920.
# Capas (de atrás para adelante): piernas sentado (opcional) · cuerpo · brazos · esclavina de los hombros · manos.
# Cada capa es un Polygon2D en grilla con pesos SOLO a sus huesos (los brazos no arrastran el cuerpo).
# Los brazos se mueven con IK de dos huesos (hombro-codo, codo-muñeca) hacia la muñeca que pide el gesto;
# la mano gira aparte (o apunta a un punto) y la manga campana cuelga con gravedad.

const CELL := 16

var dir := ""
var skel: Skeleton2D
var B := {}       # nombre -> Bone2D
var G := {}       # nombre -> pivote en reposo (coords del personaje)
var RL := {}      # nombre -> posición local en reposo
var RIG: Array
var CAPS := {}    # nombre de hueso -> cápsulas [a, b, radio]
var cuerpo_pie: Polygon2D
var cuerpo_sentado: Polygon2D
var piernas: Sprite2D
var brazo := {}   # "L"/"R" -> Polygon2D
var manto: Polygon2D
var mano := {}    # "L"/"R" -> Sprite2D
var pluma: Sprite2D
var arm := {}     # "L"/"R" -> datos de IK
var dbg: Node2D
var t := 0.0


func rig_def() -> Array:
	# [nombre, padre, pivote]
	var d := [
		["hips", "", Vector2(549, 900)],
		["spine", "hips", Vector2(549, 880)],
		["chest", "spine", Vector2(549, 760)],
		["head", "chest", Vector2(549, 560)],
		["hem", "hips", Vector2(549, 1160)],
	]
	var side := [
		["hombro", "chest", Vector2(360, 640)],
		["brazo", "hombro", Vector2(335, 665)],
		["ante", "brazo", Vector2(285, 815)],
		["mano", "ante", Vector2(245, 985)],
		["camp", "ante", Vector2(240, 1000)],
		["capa", "hips", Vector2(300, 1120)],
		["capa2", "capa", Vector2(240, 1500)],
	]
	for s in ["L", "R"]:
		for b in side:
			var par: String = b[1] if b[1] == "chest" or b[1] == "hips" else b[1] + s
			d.append([b[0] + s, par, mx(b[2], s)])
	return d


func caps_def() -> void:
	CAPS = {
		"hips": [[Vector2(450, 1100), Vector2(450, 1780), 45.0], [Vector2(648, 1100), Vector2(648, 1780), 45.0], [Vector2(549, 900), Vector2(549, 1100), 70.0]],
		"spine": [[Vector2(549, 860), Vector2(549, 775), 85.0]],
		"chest": [[Vector2(549, 760), Vector2(549, 600), 110.0], [Vector2(450, 640), Vector2(648, 640), 40.0]],
		"head": [[Vector2(549, 470), Vector2(549, 280), 115.0]],
		"hem": [[Vector2(549, 1200), Vector2(549, 1630), 40.0]],
	}
	var side := {
		"hombro": [[Vector2(300, 700), Vector2(400, 660), 45.0]],
		"brazo": [[Vector2(335, 665), Vector2(285, 815), 55.0]],
		"ante": [[Vector2(285, 815), Vector2(250, 975), 50.0]],
		"mano": [[Vector2(235, 1020), Vector2(150, 1110), 35.0]],
		"camp": [[Vector2(215, 1070), Vector2(160, 1500), 55.0]],
		"capa": [[Vector2(300, 1150), Vector2(240, 1480), 35.0]],
		"capa2": [[Vector2(230, 1520), Vector2(90, 1730), 45.0]],
	}
	for s in ["L", "R"]:
		for k in side:
			var cs := []
			for c in side[k]:
				cs.append([mx(c[0], s), mx(c[1], s), c[2]])
			CAPS[k + s] = cs


func mx(v: Vector2, s: String) -> Vector2:
	return v if s == "L" else Vector2(1098.0 - v.x, v.y)


func construir(capas_dir: String, piezas: Dictionary, conf: Dictionary) -> void:
	dir = capas_dir
	RIG = rig_def()
	caps_def()
	skel = Skeleton2D.new()
	for b in RIG:
		var bone := Bone2D.new()
		bone.name = b[0]
		var parent: Node = skel if b[1] == "" else B[b[1]]
		var pg: Vector2 = Vector2.ZERO if b[1] == "" else G[b[1]]
		parent.add_child(bone)
		var p: Vector2 = b[2]
		bone.position = p - pg
		bone.rest = Transform2D(0.0, bone.position)
		bone.set_autocalculate_length_and_angle(false)
		bone.set_length(30.0)
		B[b[0]] = bone
		G[b[0]] = p
		RL[b[0]] = bone.position

	var cuerpo_huesos := ["hips", "spine", "chest", "head", "hem", "hombroL", "hombroR", "capaL", "capa2L", "capaR", "capa2R"]
	# piernas sentado (pieza de FLUX), detrás del cuerpo
	if piezas.has("sentado") and conf.has("sentado"):
		var c: Dictionary = conf["sentado"]
		piernas = Sprite2D.new()
		piernas.texture = tex(dir + "sentado.png")
		piernas.centered = false
		piernas.scale = Vector2.ONE * float(c["esc"])
		piernas.position = Vector2(c["pos"][0], c["pos"][1])
		piernas.visible = false
		add_child(piernas)
	cuerpo_pie = malla(dir + "cuerpo.png", cuerpo_huesos)
	add_child(cuerpo_pie)
	if FileAccess.file_exists(dir + "cuerpo_sentado.png"):
		cuerpo_sentado = malla(dir + "cuerpo_sentado.png", cuerpo_huesos)
		cuerpo_sentado.visible = false
		add_child(cuerpo_sentado)
	for s in ["L", "R"]:
		var p := malla(dir + "brazo" + s + ".png", ["brazo" + s, "ante" + s, "camp" + s])
		p.z_index = 1
		add_child(p)
		brazo[s] = p
	manto = malla(dir + "manto.png", cuerpo_huesos)
	manto.z_index = 2
	add_child(manto)
	add_child(skel)
	for p in [cuerpo_pie, cuerpo_sentado, brazo["L"], brazo["R"], manto]:
		if p != null:
			ligar(p)
	# manos: sprites rígidos, hijos del hueso de la muñeca
	for s in ["L", "R"]:
		var sp := Sprite2D.new()
		sp.texture = tex(dir + "mano" + s + ".png")
		sp.centered = false
		sp.position = -G["mano" + s]
		sp.z_index = 3
		B["mano" + s].add_child(sp)
		mano[s] = sp
	if piezas.has("mano_pluma") and conf.has("pluma"):
		var c: Dictionary = conf["pluma"]
		pluma = Sprite2D.new()
		pluma.texture = tex(dir + "mano_pluma.png")
		pluma.offset = -Vector2(c["agarre"][0], c["agarre"][1])   # punto de la pieza que va en la muñeca
		pluma.centered = false
		pluma.scale = Vector2.ONE * float(c["esc"])
		pluma.rotation = deg_to_rad(float(c["rot"]))
		pluma.z_index = 3
		pluma.modulate.a = 0.0
		B["mano" + String(c["lado"])].add_child(pluma)
	# datos para IK
	for s in ["L", "R"]:
		var S0: Vector2 = G["brazo" + s]
		var E0: Vector2 = G["ante" + s]
		var W0: Vector2 = G["mano" + s]
		var F0: Vector2 = mx(Vector2(140, 1115), s)
		arm[s] = {
			"L1": S0.distance_to(E0), "L2": E0.distance_to(W0),
			"a_up": (E0 - S0).angle(), "a_fo": (W0 - E0).angle(), "a_ha": (F0 - W0).angle(),
			"hand_len": W0.distance_to(F0), "W0": W0,
			"bend": signf(wrapf((E0 - S0).angle() - (W0 - S0).angle(), -PI, PI)),
		}
		if arm[s]["bend"] == 0.0:
			arm[s]["bend"] = 1.0 if s == "L" else -1.0


func tex(path: String) -> ImageTexture:
	return ImageTexture.create_from_image(Image.load_from_file(path))


func seg_dist(p: Vector2, a: Vector2, b: Vector2) -> float:
	var ab := b - a
	var u := clampf((p - a).dot(ab) / maxf(ab.length_squared(), 1e-6), 0.0, 1.0)
	return p.distance_to(a + ab * u)


func malla(path: String, huesos: Array) -> Polygon2D:
	var img := Image.load_from_file(path)
	img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	var iw := img.get_width()
	var ih := img.get_height()
	var nx := int(ceil(float(iw) / CELL))
	var ny := int(ceil(float(ih) / CELL))
	var occ := PackedByteArray()
	occ.resize(nx * ny)
	for j in ny:
		for i in nx:
			var hit := 0
			var y := j * CELL
			while y < mini((j + 1) * CELL, ih) and hit == 0:
				var x := i * CELL
				while x < mini((i + 1) * CELL, iw):
					if data[(y * iw + x) * 4 + 3] > 3:
						hit = 1
						break
					x += 2
				y += 2
			occ[j * nx + i] = hit
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
	var pts := PackedVector2Array()
	for j in ny:
		for i in nx:
			if occ2[j * nx + i] == 0:
				continue
			var q := PackedInt32Array()
			for c in [Vector2i(i, j), Vector2i(i + 1, j), Vector2i(i + 1, j + 1), Vector2i(i, j + 1)]:
				if not idx.has(c):
					idx[c] = pts.size()
					pts.append(Vector2(minf(c.x * CELL, iw), minf(c.y * CELL, ih)))
				q.append(idx[c])
			quads.append(q)
	var nv := pts.size()
	var nb := huesos.size()
	var flat := PackedFloat32Array()
	flat.resize(nb * nv)
	for vi in nv:
		var p := pts[vi]
		var tot := 0.0
		for bi in nb:
			var dmin := 1e9
			for c in CAPS[huesos[bi]]:
				dmin = minf(dmin, maxf(seg_dist(p, c[0], c[1]) - c[2], 0.0))
			var w := 1.0 / pow(dmin + 20.0, 3.0)
			flat[bi * nv + vi] = w
			tot += w
		for bi in nb:
			flat[bi * nv + vi] /= tot
	var poly := Polygon2D.new()
	poly.texture = ImageTexture.create_from_image(img)
	poly.polygon = pts
	poly.uv = pts
	poly.polygons = quads
	var w := {}
	for bi in nb:
		w[huesos[bi]] = flat.slice(bi * nv, (bi + 1) * nv)
	poly.set_meta("pesos", w)
	print("malla ", path.get_file(), ": ", nv, " vértices")
	return poly


func ligar(poly: Polygon2D) -> void:
	poly.skeleton = poly.get_path_to(skel)
	var w: Dictionary = poly.get_meta("pesos")
	for h in w:
		poly.add_bone(skel.get_path_to(B[h]), w[h])


func setb(n: String, dp: Vector2, rot: float, sc := Vector2.ONE) -> void:
	var b: Bone2D = B[n]
	b.position = RL[n] + dp
	b.rotation = rot
	b.scale = sc


func sentado(si: bool) -> void:
	if cuerpo_sentado == null or piernas == null:
		return
	cuerpo_sentado.visible = si
	piernas.visible = si
	cuerpo_pie.visible = not si


# punta de los dedos (coords del personaje) con la pose actual
func punta(s: String) -> Vector2:
	var hb: Bone2D = B["mano" + s]
	var F0: Vector2 = mx(Vector2(140, 1115), s)
	return skel.to_local(hb.global_transform * (F0 - G["mano" + s]))


func aplicar(e: Dictionary, tt: float) -> void:
	t = tt
	var D := PI / 180.0
	var pe: float = e["pecho"]
	var esc: float = e["pecho_esc"]
	setb("spine", Vector2(0, -pe * 0.35), float(e["torso"]) * D)
	setb("chest", Vector2(0, -pe * 0.65), 0.0, Vector2(1.0 + esc, 1.0 + esc * 0.6))
	setb("head", Vector2(0, float(e["cabeza_y"]) - pe * 0.1), float(e["cabeza"]) * D)
	setb("hombroL", Vector2(0, -float(e["hombros"])), 0.0)
	setb("hombroR", Vector2(0, -float(e["hombros"])), 0.0)
	var w := TAU / 6.0
	var viento: float = e["viento"]
	setb("hem", Vector2.ZERO, viento * (0.7 * D * sin(w * t - 2.0)))
	setb("capaL", Vector2.ZERO, viento * (0.8 * D * sin(w * t - 1.0)))
	setb("capa2L", Vector2.ZERO, viento * (1.8 * D * sin(w * t - 1.7)))
	setb("capaR", Vector2.ZERO, viento * (-0.8 * D * sin(w * t - 1.3)))
	setb("capa2R", Vector2.ZERO, viento * (-1.8 * D * sin(w * t - 2.0)))
	for s in ["L", "R"]:
		var dh: Vector2 = Vector2(0, -float(e["hombros"]) - pe * 0.65)
		ik(s, Vector2(e["mu" + s]) + dh, float(e["mano" + s]), float(e["apunta_w"]), Vector2(e["apunta"]))
		var atras: float = e["atras" + s]
		brazo[s].z_index = -1 if atras > 0.5 else 1
		mano[s].z_index = -1 if atras > 0.5 else 3
	var pl: float = e["pluma"]
	if pluma != null:
		pluma.modulate.a = clampf(pl, 0.0, 1.0)
		mano[String(pluma.get_parent().name).substr(4)].modulate.a = 1.0 - clampf(pl, 0.0, 1.0)


func ik(s: String, target: Vector2, hand_deg: float, aim_w: float, aim: Vector2) -> void:
	var a: Dictionary = arm[s]
	var up: Bone2D = B["brazo" + s]
	var fo: Bone2D = B["ante" + s]
	var ha: Bone2D = B["mano" + s]
	var ca: Bone2D = B["camp" + s]
	var L1: float = a["L1"]
	var L2: float = a["L2"]
	var S := skel.to_local(up.global_position)
	var v := target - S
	var d := clampf(v.length(), absf(L1 - L2) + 1.0, L1 + L2 - 0.5)
	var a1 := v.angle()
	var c := clampf((L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d), -1.0, 1.0)
	var th1 := a1 + float(a["bend"]) * acos(c)
	var E := S + Vector2.from_angle(th1) * L1
	var Wp := S + v.normalized() * d
	var th2 := (Wp - E).angle()
	var parent_g: float = (skel.global_transform.affine_inverse() * B["hombro" + s].global_transform).get_rotation()
	var up_g := th1 - float(a["a_up"])
	var fo_g := th2 - float(a["a_fo"])
	up.rotation = wrapf(up_g - parent_g, -PI, PI)
	fo.rotation = wrapf(fo_g - up_g, -PI, PI)
	var hand_g := fo_g + deg_to_rad(hand_deg)
	if aim_w > 0.0:
		var want := (aim - Wp).angle() - float(a["a_ha"])
		hand_g = lerp_angle(hand_g, want, clampf(aim_w, 0.0, 1.0))
	ha.rotation = wrapf(hand_g - fo_g, -PI, PI)
	# la campana cuelga: compensa parte del giro del antebrazo
	var sw := 1.2 * PI / 180.0 * sin(TAU * t / 3.0 + (0.0 if s == "L" else 1.3))
	ca.rotation = -0.55 * wrapf(fo_g, -PI, PI) + sw
