"""Video de estudio con imágenes fijas (22/11/2026): una imagen por escena del guion + voz + subtítulos. Sin animación.
Uso: python img/armar.py <practica.md> <timeline.json> <voz.wav> <imgs/> <salida.mp4>
Cortes: «intro» al empezar Nombre/Para qué/Pasos, «historia» en la Historia, y cada «escena:» de la práctica."""
import json, re, subprocess, sys, pathlib

md_p, tl_p, voz, imgs, out = sys.argv[1:6]
md = open(md_p).read(); tl = json.load(open(tl_p)); D = pathlib.Path(imgs)
claves = json.load(open(D / "claves.json")); archivo = {k: next(D.glob(f"{i}.*[gG]")) for i, k in enumerate(claves)}
titulo = (re.search(r"^titulo:\s*(.+)$", md, re.M) or [None, ""])[1].strip()

# escenas: «escena: X» antes de la línea i-n (misma numeración que las frases del timeline)
esc, i, n, pend = [], -1, 0, None
for l in md.splitlines():
    if l.startswith("## "): break
    s = l.strip()
    if re.match(r"\*\*(\d)\.", s): i, n = i + 1, 0; continue
    m = re.match(r"escena:\s*(.+)$", s)
    if m and i >= 0: pend = m.group(1).strip(); continue
    if i >= 0 and s and not s.startswith("#") and not re.match(r"[a-z]+:", s):
        if pend: esc.append(((i, n), pend)); pend = None
        n += 1

pal = tl["palabras"]; fr = lambda w: tuple(int(x) for x in w["frase"].split("-"))
cortes = []
for p in tl["partes"]:
    if p["tag"] != "practica": cortes.append((p["ini"], "historia" if p["tag"] == "historia" and "historia" in archivo else "intro"))
for (pos, nombre) in esc:
    w = next((w for w in pal if fr(w) >= pos), None)
    if w: cortes.append((max(0, w["ini"] - 0.3), nombre))
cortes.sort()
seg = []
for t, k in cortes:
    if k not in archivo: continue
    if seg and seg[-1][1] == k: continue
    seg.append([t, k])
seg[0][0] = 0.0
dur = tl["duracion"]
lista = []
for j, (t, k) in enumerate(seg):
    fin = seg[j + 1][0] if j + 1 < len(seg) else dur
    lista.append(f"file '{archivo[k].resolve()}'\nduration {max(0.1, fin - t):.3f}")
lista.append(f"file '{archivo[seg[-1][1]].resolve()}'")
pathlib.Path("lista.txt").write_text("\n".join(lista) + "\n")

# subtítulos: una línea del guion por vez; arriba el nombre de la escena
ts = lambda x: f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"
lim = lambda s: s.replace("{", "(").replace("}", ")").replace("\n", " ")
ev, frases = [], {}
for w in pal: frases.setdefault(w["frase"], []).append(w)
for ws in frases.values():
    ev.append(f"Dialogue: 0,{ts(ws[0]['ini'])},{ts(ws[-1]['fin'] + 0.25)},Sub,,0,0,0,,{lim(' '.join(w['texto'] for w in ws))}")
for j, (t, k) in enumerate(seg):
    fin = seg[j + 1][0] if j + 1 < len(seg) else dur
    ev.append(f"Dialogue: 0,{ts(t)},{ts(fin)},Tit,,0,0,0,,{lim(titulo if k in ('intro', 'historia') else k).upper()}")
pathlib.Path("subs.ass").write_text("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,DejaVu Sans,58,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,1,0,0,0,100,100,0,0,3,14,0,2,70,70,170,1
Style: Tit,DejaVu Sans,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,1,0,0,0,100,100,2,0,3,12,0,8,70,70,110,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(ev) + "\n")

subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "lista.txt", "-i", voz,
                "-vf", "scale=-2:1920:flags=lanczos,crop=1080:1920,fps=10,ass=subs.ass,format=yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage", "-crf", "24",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
                "-shortest", "-movflags", "+faststart", out], check=True)
print(f"{len(seg)} imágenes, {dur:.0f} s")
