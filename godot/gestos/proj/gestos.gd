extends RefCounted
# Biblioteca de gestos del encapuchado (fase 2). Reutilizable: cada video es una lista de acciones con tiempos
# (linea.json: [["write", 0, 4.5], ["breathe_in", 4.5, 6.5], ...]) sobre una pose base por escena.
#
# Canales (todos se interpolan): pecho/hombros (px hacia arriba), pecho_esc (escala extra del pecho), cabeza (grados),
# cabeza_y (px), torso (grados), muL/muR (muñeca objetivo, coords del personaje), manoL/manoR (giro de la mano, grados),
# apunta (punto) y apunta_w (0..1: la mano apunta los dedos ahí), pluma (0..1: mano con pluma), atrasL/atrasR (brazo
# detrás del cuerpo), viento (0..1), y canales de efectos: luz_pecho, cuadrado (0..4 lados dibujados), cuadrado_a, chispa.
#
# Pose = diccionario parcial de canales. Acción = claves [u, pose, curva] con u de 0 a 1 sobre la duración de la acción
# (o sobre "ciclo" segundos si se repite), o una función ("fn") que arma la pose en cada instante.
# Entre acciones se mezcla durante "entrada" segundos desde donde quedó la anterior.

const MU_L := Vector2(245, 985)
const MU_R := Vector2(853, 985)

var DEF := {
	"pecho": 0.0, "hombros": 0.0, "pecho_esc": 0.0, "cabeza": 0.0, "cabeza_y": 0.0, "torso": 0.0,
	"muL": MU_L, "muR": MU_R, "manoL": 0.0, "manoR": 0.0,
	"apunta": Vector2(549, 745), "apunta_w": 0.0, "pluma": 0.0, "atrasL": 0.0, "atrasR": 0.0, "viento": 1.0,
	"luz_pecho": 0.0, "cuadrado": 0.0, "cuadrado_a": 0.0, "chispa": 0.0,
}

var POSES := {
	# bases de escena
	"de_pie": {},
	"escritorio": {"muL": Vector2(430, 945), "muR": Vector2(668, 950), "manoL": 18.0, "manoR": -22.0, "cabeza_y": 6.0, "viento": 0.0},
	"sentado": {"muL": Vector2(285, 985), "muR": Vector2(813, 985), "manoL": 30.0, "manoR": -30.0, "viento": 0.3},
	# poses de gesto
	"escribir_a": {"muL": Vector2(425, 940), "manoL": 22.0, "pluma": 1.0, "cabeza_y": 9.0, "cabeza": 2.0},
	"escribir_b": {"muL": Vector2(447, 952), "manoL": 14.0, "pluma": 1.0, "cabeza_y": 9.0, "cabeza": 1.0},
	"escribir_c": {"muL": Vector2(466, 940), "manoL": 20.0, "pluma": 1.0, "cabeza_y": 9.0, "cabeza": 0.0},
	"mira_arriba": {"muL": Vector2(432, 940), "manoL": 18.0, "pluma": 1.0, "cabeza_y": -3.0, "cabeza": -3.0},
	"inhalar": {"pecho": 16.0, "hombros": 20.0, "pecho_esc": 0.035, "cabeza_y": -5.0, "luz_pecho": 1.0},
	"exhalar": {"pecho": -3.0, "hombros": -4.0, "pecho_esc": -0.005, "cabeza_y": 2.0, "luz_pecho": 0.1},
	"brazos_arriba": {"muL": Vector2(250, 290), "muR": Vector2(848, 290), "manoL": -25.0, "manoR": 25.0,
		"hombros": 14.0, "pecho": 8.0, "pecho_esc": 0.02, "cabeza_y": -6.0},
	"brazos_medio": {"muL": Vector2(150, 600), "muR": Vector2(948, 600), "manoL": -15.0, "manoR": 15.0, "hombros": 8.0, "pecho": 5.0},
	"medita_in": {"pecho": 7.0, "hombros": 8.0, "pecho_esc": 0.015, "luz_pecho": 0.8},
	"invitar": {"muL": Vector2(245, 925), "muR": Vector2(853, 925), "manoL": 40.0, "manoR": -40.0, "cabeza": 3.0, "cabeza_y": -2.0,
		"luz_pecho": 0.6, "hombros": 4.0},
}

var ACCIONES := {
	"write": {"ciclo": 0.9, "claves": [[0.0, "escribir_a", "sine"], [0.33, "escribir_b", "sine"], [0.66, "escribir_c", "sine"], [1.0, "escribir_a", "sine"]]},
	"look_up": {"claves": [[0.0, "escribir_a", "inout"], [0.6, "mira_arriba", "inout"], [1.0, "mira_arriba", "inout"]]},
	"breathe_in": {"claves": [[0.0, "de_pie", "sine"], [1.0, "inhalar", "sine"]], "entrada": 0.3},
	"breathe_out": {"claves": [[0.0, "inhalar", "sine"], [1.0, "exhalar", "sine"]], "entrada": 0.1},
	"square": {"fn": "fn_cuadrado", "entrada": 0.6},
	"raise_arms": {"claves": [[0.0, "de_pie", "inout"], [0.4, "brazos_arriba", "out"], [0.8, "brazos_arriba", "inout"], [1.0, "brazos_medio", "inout"]]},
	"meditate": {"claves": [[0.0, "sentado", "sine"], [0.5, "medita_in", "sine"], [1.0, "sentado", "sine"]]},
	"invite": {"claves": [[0.0, "sentado", "inout"], [0.55, "invitar", "out"], [1.0, "invitar", "inout"]], "entrada": 0.2},
}

# cuadrado de luz delante del pecho: cada fase (inhala, retiene, exhala, retiene) dibuja un lado
const CENTRO := Vector2(549, 760)
const MEDIO := 200.0


func esquinas() -> Array:
	return [CENTRO + Vector2(-MEDIO, MEDIO), CENTRO + Vector2(-MEDIO, -MEDIO), CENTRO + Vector2(MEDIO, -MEDIO), CENTRO + Vector2(MEDIO, MEDIO)]


func fn_cuadrado(u: float, dur: float) -> Dictionary:
	var lado := u * 4.0
	var i := mini(int(floor(lado)), 3)
	var f := suave(clampf(lado - i, 0.0, 1.0), "inout")
	var es := esquinas()
	var p: Vector2 = es[i].lerp(es[(i + 1) % 4], f)
	# respiración acompaña: sube en el lado 1, queda arriba en el 2, baja en el 3, queda abajo en el 4
	var r := 0.0
	if i == 0:
		r = f
	elif i == 1:
		r = 1.0
	elif i == 2:
		r = 1.0 - f
	return {
		"apunta": p, "apunta_w": 1.0,
		"muL": p + Vector2(-110, 80), "muR": p + Vector2(110, 80),
		"cuadrado": i + f, "cuadrado_a": 1.0, "chispa": 1.0,
		"pecho": 9.0 * r, "hombros": 10.0 * r, "pecho_esc": 0.02 * r, "luz_pecho": 0.5 * r,
	}


func suave(u: float, curva: String) -> float:
	u = clampf(u, 0.0, 1.0)
	match curva:
		"lin":
			return u
		"in":
			return u * u
		"out":
			return 1.0 - (1.0 - u) * (1.0 - u)
		"sine":
			return 0.5 - 0.5 * cos(PI * u)
	return u * u * (3.0 - 2.0 * u)


func mezclar(a: Dictionary, b: Dictionary, w: float) -> Dictionary:
	var r := {}
	for k in a:
		r[k] = lerp(a[k], b[k], w)
	return r


func completa(base: Dictionary, parcial: Dictionary) -> Dictionary:
	var r := DEF.duplicate()
	for k in base:
		r[k] = base[k]
	for k in parcial:
		r[k] = parcial[k]
	return r


func pose(nombre, base: Dictionary) -> Dictionary:
	if nombre is Dictionary:
		return completa(base, nombre)
	# las poses de escena (de_pie, escritorio, sentado) se toman como "volver a la base"
	if nombre in ["de_pie", "escritorio", "sentado"]:
		return completa(base, {})
	return completa(base, POSES[nombre])


func evaluar(acc: Dictionary, t: float, t0: float, t1: float, base: Dictionary) -> Dictionary:
	var dur := maxf(t1 - t0, 0.001)
	if acc.has("fn"):
		var uf := clampf((t - t0) / dur, 0.0, 1.0)
		return completa(base, call(acc["fn"], uf, dur))
	var u: float
	if acc.has("ciclo"):
		u = fposmod((t - t0) / float(acc["ciclo"]), 1.0)
	else:
		u = clampf((t - t0) / dur, 0.0, 1.0)
	var cl: Array = acc["claves"]
	for i in range(cl.size() - 1):
		var k0: Array = cl[i]
		var k1: Array = cl[i + 1]
		if u <= float(k1[0]) or i == cl.size() - 2:
			var f := (u - float(k0[0])) / maxf(float(k1[0]) - float(k0[0]), 1e-4)
			return mezclar(pose(k0[1], base), pose(k1[1], base), suave(f, String(k1[2])))
	return pose(cl[0][1], base)


# estado de todos los canales en el instante t
func estado(t: float, linea: Array, base: Dictionary) -> Dictionary:
	var activa := -1
	var ultima := -1
	for i in linea.size():
		var li: Array = linea[i]
		if t >= float(li[1]) and t < float(li[2]):
			activa = i
		if float(li[2]) <= t:
			ultima = i
	var quieta := completa(base, {})
	if activa < 0:
		if ultima >= 0:
			var li: Array = linea[ultima]
			var fin := evaluar(ACCIONES[li[0]], float(li[2]), float(li[1]), float(li[2]), base)
			var w := suave((t - float(li[2])) / 0.6, "inout")
			return mezclar(fin, quieta, w)
		return quieta
	var a: Array = linea[activa]
	var acc: Dictionary = ACCIONES[a[0]]
	var ahora := evaluar(acc, t, float(a[1]), float(a[2]), base)
	var ent: float = acc.get("entrada", 0.5)
	var dt := t - float(a[1])
	if dt < ent:
		var antes := quieta
		if activa > 0:
			var p: Array = linea[activa - 1]
			if float(a[1]) - float(p[2]) < 0.6:
				antes = evaluar(ACCIONES[p[0]], float(p[2]), float(p[1]), float(p[2]), base)
				if float(a[1]) > float(p[2]):
					antes = mezclar(antes, quieta, suave((float(a[1]) - float(p[2])) / 0.6, "inout"))
		return mezclar(antes, ahora, suave(dt / ent, "inout"))
	return ahora
