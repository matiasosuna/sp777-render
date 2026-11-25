from __future__ import annotations
"""Chequea los prompts de imágenes de videos de estudio (practicas/img/<id>.json) contra su guion (practicas/<id>.md).
Uso (desde spiritual-practices): python3 validar.py id…   → una línea por id: «id OK» o «id MAL: motivo»; sale 1 si alguno está mal.
Reglas: estilo 0 o 2 (Matías descartó 1 y 3 el 24/11/2026); claves = intro (+ historia si el guion tiene Historia) + cada
«escena:» del guion, mismo texto; prompts en inglés, de 8 a 90 palabras; nada sensual; personas con cara visible."""
import json, pathlib, re, sys

PROHIBIDAS = r"\b(nude|naked|bare|shirtless|topless|sensual|sexy|lingerie|bikini|underwear|kiss|kissing|embrac\w*|lover|romantic|seductive|cleavage|bathing|bath|undress\w*|skin-tight|erotic)\b"
CABEZA = r"\b(face|faces|head|whole body|full figure|wide shot|from a distance|seen from behind|back view)\b"
PERSONA = r"\b(man|woman|men|women|person|people|monk|healer|teacher|figure|boy|girl|elder|student|students|patient|someone)\b"


def escenas(md: str) -> tuple[list[str], bool]:
    cuerpo = md.split("\n## ")[0]
    return re.findall(r"^escena:\s*(.+?)\s*$", cuerpo, re.M), bool(re.search(r"^\*\*\d+\. Historia\*\*", cuerpo, re.M))


def revisar(i: str, suave: bool = False) -> str | None:
    md_p, j_p = pathlib.Path(f"practicas/{i}.md"), pathlib.Path(f"practicas/img/{i}.json")
    if not md_p.exists(): return "falta el guion"
    if not j_p.exists(): return "faltan los prompts"
    try: d = json.loads(j_p.read_text())
    except Exception as e: return f"json roto ({e})"
    if d.get("estilo") not in (0, 2): return f"estilo {d.get('estilo')!r} (solo 0 o 2)"
    imgs = d.get("imgs")
    if not isinstance(imgs, dict) or not imgs: return "sin imgs"
    esc, hist = escenas(md_p.read_text())
    if not esc: return "el guion no tiene escenas"
    if len(set(esc)) != len(esc): return "el guion repite un nombre de escena"
    esperadas = ["intro"] + (["historia"] if hist else []) + esc
    if set(imgs) != set(esperadas):
        return f"claves distintas a las escenas: sobran {sorted(set(imgs) - set(esperadas))}, faltan {sorted(set(esperadas) - set(imgs))}"
    for k, p in imgs.items():
        if not isinstance(p, str): return f"{k}: prompt no es texto"
        n = len(p.split())
        if not 8 <= n <= 90: return f"{k}: {n} palabras (8 a 90)"
        if re.search(PROHIBIDAS, p, re.I): return f"{k}: palabra prohibida «{re.search(PROHIBIDAS, p, re.I)[0]}»"
        if re.search(r"[áéíóúñ¿¡]", p): return f"{k}: parece en castellano"
        if not suave and re.search(PERSONA, p, re.I) and not re.search(r"\bno people\b", p, re.I) and not re.search(CABEZA, p, re.I):
            return f"{k}: hay una persona pero no pide la cara o el cuerpo entero visible"
    return None


if __name__ == "__main__":
    mal, suave = 0, "--suave" in sys.argv  # --suave: no exige la cara (prompts viejos ya revisados en imagen)
    for i in [a for a in sys.argv[1:] if a != "--suave"]:
        m = revisar(i, suave); mal += bool(m)
        print(f"{i} {'OK' if not m else 'MAL: ' + m}")
    sys.exit(1 if mal else 0)
