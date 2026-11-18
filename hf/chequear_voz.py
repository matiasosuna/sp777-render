#!/usr/bin/env python3
"""Revisa el timeline de una voz: ninguna palabra corta debe durar de más (Chatterbox repetía «Here's how it goes.»).
Mismo criterio que videos/modal_voz.py `decir`: dura más de max(1.0 s, 0.13 s por letra) → mal.
Uso: chequear_voz.py timeline.json chequeo.json   (sale 1 si hay palabras largas). No imprime el guion: solo frase y duración."""
import json, re, sys

tl = json.load(open(sys.argv[1]))
largas = []
for i, w in enumerate(tl["palabras"]):
    n = len(re.sub(r"[^a-z']", "", w["texto"].lower()))
    d = w["fin"] - w["ini"]
    if n and d > max(1.0, 0.13 * n): largas.append({"frase": w["frase"], "n": i, "letras": n, "dura": round(d, 2), "ini": round(w["ini"], 2)})
r = {"duracion": round(tl["duracion"], 1), "palabras": len(tl["palabras"]), "largas": largas}
json.dump(r, open(sys.argv[2], "w"))
print(f"voz {r['duracion']} s, {r['palabras']} palabras, {len(largas)} largas", *(f"frase {x['frase']} ({x['letras']} letras) {x['dura']} s en {x['ini']} s" for x in largas), sep="\n  ")
sys.exit(1 if largas else 0)
