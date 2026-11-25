"""Ciclo de la nube (25/11/2026): todo el repertorio sin la Mac. Lo corre .github/workflows/noche.yml cada 30 min.

Carpetas (las arma el workflow): sp/ = spiritual-practices (solo lectura: guiones, prompts, nube/aprobados.txt, nube/rehacer.txt),
est/ = rama `estado` de este repo (lo único que se escribe): imagenes.json, voces.json, telegram.json, nube.json, hojas/<id>.jpg,
pendientes.txt (hojas para revisar); los json van en est/practicas/ (mismo camino que en spiritual-practices). La sesión de Claude en la nube revisa las hojas, anota en sp/practicas/nube/ y copia los
json de estado a practicas/ de spiritual-practices.

Pasos de cada vuelta:
 1. rehacer.txt («id [semilla]»): saca el id de imágenes y de lo que está en curso → vuelve a hacer imágenes (con su semilla).
 2. runs de img.yml terminados → hoja a est/hojas, voz a voces.json, id a «revisar» (o a «fallos»).
 3. runs de subir.yml terminados → telegram.json.
 4. aprobados.txt ∩ revisar → subir.yml (los mp4 de sus runs).
 5. guiones con prompts y sin imágenes → FLUX en Modal (cuenta con crédito gratis, tope 28 USD c/u) → Telegram → imagenes.json.
 6. con imágenes y sin video → img.yml (tandas de 25).
"""
import json, os, pathlib, re, subprocess, sys, tarfile, time

SP, EST = pathlib.Path("sp").resolve(), pathlib.Path("est").resolve()
MOTOR = pathlib.Path("motor").resolve()
REPO = os.environ.get("GITHUB_REPOSITORY", "matiasosuna/sp777-render")
RAMA_SP = os.environ.get("RAMA_SP", "puente/spiritual-practices-777")
MAX_IMG = int(os.environ.get("MAX_IMG", "30"))
MAX_TANDAS = int(os.environ.get("MAX_TANDAS", "3"))
TANDA = 25
CUENTAS = [("hermestarot444", 28), ("familia", 28), ("osuna-devs", 28)]
log = lambda *a: print(*a, flush=True)


def leer(p, vacio):
    return json.loads(p.read_text()) if p.exists() else vacio


def escribir(p, d):
    p.write_text(json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def sh(*a, env=None, check=True):
    r = subprocess.run(a, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and r.returncode: raise RuntimeError(f"{' '.join(a[:4])}…: {r.stderr.strip()[-400:]}")
    return r.stdout


def lineas(p):
    return [l.split() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")] if p.exists() else []


J = EST / "practicas"; J.mkdir(parents=True, exist_ok=True)
img, voces, tg = leer(J / "imagenes.json", {}), leer(J / "voces.json", {}), leer(J / "telegram.json", {})
nube = leer(EST / "nube.json", {})
for k in ("render", "revisar", "fallos", "subir"): nube.setdefault(k, {})
nube.setdefault("rehechos", []); nube.setdefault("cola", [])
cola = list(dict.fromkeys(nube["cola"]))


def guardar():  # siempre, aunque la vuelta se corte: lo lanzado queda anotado y no se repite
    nube["cola"] = list(dict.fromkeys(cola))
    escribir(J / "imagenes.json", img); escribir(J / "voces.json", voces); escribir(J / "telegram.json", tg)
    escribir(EST / "nube.json", nube)
    aprob_ = {l[0] for l in lineas(SP / "practicas/nube/aprobados.txt")}
    (EST / "pendientes.txt").write_text("".join(f"{i} {r}\n" for i, r in sorted(nube["revisar"].items()) if i not in aprob_))


import atexit  # noqa: E402
atexit.register(guardar)
(EST / "hojas").mkdir(parents=True, exist_ok=True)

# 1. rehacer
for l in lineas(SP / "practicas/nube/rehacer.txt"):
    clave = " ".join(l)
    if clave in nube["rehechos"]: continue
    i = l[0]
    for k in ("render", "revisar", "fallos"): nube[k].pop(i, None)
    img.pop(i, None); cola = [x for x in cola if x != i]
    nube["rehechos"].append(clave); log(f"rehacer {clave}")

# 2. runs de img.yml terminados
def estado_run(r):
    return json.loads(sh("gh", "run", "view", str(r), "-R", REPO, "--json", "status,conclusion"))

for r in sorted(set(nube["render"].values())):
    if estado_run(r)["status"] != "completed": continue
    ids = [i for i, x in nube["render"].items() if x == r]
    for i in ids:
        d = pathlib.Path(f"bajado/{r}/{i}")
        sh("gh", "run", "download", str(r), "-R", REPO, "-n", f"img-{i}", "-D", str(d), check=False)
        res = leer(d / "res.json", {})
        hoja = d / "salida" / f"{i}-hoja.jpg"
        if res.get("info") and hoja.exists():
            (EST / "hojas" / f"{i}.jpg").write_bytes(hoja.read_bytes())
            if res.get("voz", {}).get(i): voces[i] = res["voz"][i]
            nube["revisar"][i] = r
        else:
            nube["fallos"][i] = r
        del nube["render"][i]
    log(f"img run {r}: {len(ids)} recogidos")

# 3. runs de subir.yml terminados
for r, ids in list(nube["subir"].items()):
    if estado_run(r)["status"] != "completed": continue
    d = pathlib.Path(f"bajado/subir-{r}")
    sh("gh", "run", "download", str(r), "-R", REPO, "-n", "subida", "-D", str(d), check=False)
    sub = leer(d / "subida.json", {})
    tg.update(sub)
    for i in sub: nube["revisar"].pop(i, None)
    del nube["subir"][r]; log(f"subir run {r}: {len(sub)} en Telegram")

# 4. aprobados → subir.yml
aprob = {l[0] for l in lineas(SP / "practicas/nube/aprobados.txt")}
subiendo = {i for ids in nube["subir"].values() for i in ids}
listos = sorted(i for i in aprob if i in nube["revisar"] and i not in subiendo)


def lanzar(workflow, campos, titulo):
    args = ["gh", "workflow", "run", workflow, "-R", REPO]
    for k, v in campos.items(): args += ["-f", f"{k}={v}"]
    sh(*args)
    for _ in range(12):
        time.sleep(5)
        runs = json.loads(sh("gh", "run", "list", "-R", REPO, "-w", workflow, "-L", "20", "--json", "databaseId,displayTitle"))
        for x in runs:
            if x["displayTitle"] == titulo: return str(x["databaseId"])
    raise RuntimeError(f"no encontré el run de {workflow}")


if listos:
    runs = " ".join(sorted({str(nube["revisar"][i]) for i in listos}))
    r = lanzar("subir.yml", {"runs": runs, "ids": " ".join(listos), "mapas": "estado"}, f"subir {runs}")
    nube["subir"][r] = listos; log(f"subir.yml {r}: {len(listos)} aprobados")

# 5. imágenes en Modal
def gasto(p):
    n = p.upper().replace("-", "_")
    env = {"MODAL_TOKEN_ID": os.environ.get(f"MODAL_{n}_ID", ""), "MODAL_TOKEN_SECRET": os.environ.get(f"MODAL_{n}_SECRET", "")}
    if not env["MODAL_TOKEN_ID"]: return None, env
    out = sh("modal", "billing", "summary", env=env, check=False)
    c = re.search(r"Credits:\s*(-?[\d.]+)", out); b = re.search(r"Billed Cost:\s*\$?([\d.]+)", out)
    return ((-float(c[1]), float(b[1])) if c and b else None), env


def cuenta():
    for p, tope in CUENTAS:
        g, env = gasto(p)
        if g and g[0] < tope and g[1] == 0: return p, g[0], env
    return None


sys.path.insert(0, str(MOTOR / "img"))
import validar  # noqa: E402

os.chdir(SP)
en_curso = set(nube["render"]) | set(nube["revisar"]) | set(nube["fallos"]) | set(cola)
cand = sorted(p.stem for p in pathlib.Path("practicas/img").glob("*.json")
              if pathlib.Path(f"practicas/{p.stem}.md").exists() and p.stem not in img and p.stem not in en_curso
              and (p.stem not in tg or any(c.split()[0] == p.stem for c in nube["rehechos"])))
hechas = 0
if cand: escribir(pathlib.Path("practicas/imagenes.json"), img)
for i in cand:
    if hechas >= MAX_IMG: break
    m = validar.revisar(i, suave=True)
    if m: log(f"{i}: prompts mal ({m})"); continue
    c = cuenta()
    if not c: log("todas las cuentas de Modal llegaron al tope: corto"); break
    p, g, env = c
    out = pathlib.Path(f"/tmp/imgs/{i}"); out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "run", str(MOTOR / "img/flux.py"), "--prompts", f"practicas/img/{i}.json", "--outdir", str(out)],
                       capture_output=True, text=True, env={**os.environ, **env})
    n = len(json.loads(pathlib.Path(f"practicas/img/{i}.json").read_text())["imgs"])
    pngs = sorted(out.glob("*.png"))
    if r.returncode or len(pngs) != n: log(f"{i}: Modal hizo {len(pngs)} de {n} ({r.stderr.strip()[-200:]})"); continue
    from PIL import Image
    for f in pngs: Image.open(f).convert("RGB").save(f.with_suffix(".jpg"), quality=88); f.unlink()
    with tarfile.open(f"/tmp/imgs/{i}.tgz", "w:gz") as t:
        for f in sorted(out.iterdir()): t.add(f, arcname=f"./{f.name}")
    sh("python", "practicas/telegram.py", "subir-imgs", f"/tmp/imgs/{i}.tgz", i)
    img = json.loads(pathlib.Path("practicas/imagenes.json").read_text())
    cola.append(i); hechas += 1; log(f"{i}: imágenes en {p} (crédito usado USD {g})")
os.chdir(SP.parent)

# 6. videos: con imágenes y sin video → img.yml
cola = [i for i in dict.fromkeys(cola) if i in img and i not in nube["render"] and i not in nube["revisar"]]
sha = sh("git", "-C", str(SP), "rev-parse", "HEAD").strip()  # img.yml lee el guion de ese commit
for k in range(MAX_TANDAS):
    t = cola[:TANDA]
    if not t: break
    ids = " ".join(t)
    r = lanzar("img.yml", {"ref": sha, "ids": ids, "mapas": "estado"}, f"img {ids}")
    for i in t: nube["render"][i] = r
    cola = cola[TANDA:]; log(f"img.yml {r}: {len(t)} ids")

log(f"fin: render {len(nube['render'])}, revisar {len(nube['revisar'])}, fallos {len(nube['fallos'])}, cola {len(cola)}, subiendo {sum(map(len, nube['subir'].values()))}")
