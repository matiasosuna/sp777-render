"""Imágenes de un video de estudio (22/11/2026): FLUX.1-schnell en Modal (perfil familia, L40S, ~2,5 s por imagen).
Uso: modal run img/flux.py --prompts practicas/img/<id>.json --outdir imgs
El json trae {"estilo": 0-3, "imgs": {clave: escena}}; sale imgs/<n>.png en el mismo orden + imgs/claves.json."""
import modal, json, pathlib, time

app = modal.App("sp777-img")
vol = modal.Volume.from_name("flux-cache", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.5.1", "diffusers==0.32.2", "transformers==4.47.1", "accelerate==1.2.1",
                    "sentencepiece==0.2.0", "protobuf==5.29.2", "huggingface_hub[hf_transfer]==0.27.1")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"}))

ESTILOS = [  # uno por video, rotando (Matías eligió estos cuatro)
    "Antique hand-painted watercolor and ink book illustration, early 1900s, cream paper, sepia and muted red, fine linework.",
    "Flat 2D Persian miniature painting, hand-painted gouache, not a photograph, intricate patterns, lapis blue and gold.",
    "Modern minimal flat vector illustration, not a photograph, soft pastel palette, simple shapes, clean light background.",
    "Soft modern watercolor, loose brushstrokes, gentle light, airy pale background.",
]
COMUN = (" Everyone fully clothed in modest traditional clothing, calm dignified posture, non-sensual, natural hands, "
         "correct anatomy, two hands per person. No text, no letters, no calligraphy, no writing, no seal, no stamp, no signature, no border.")


@app.function(image=img, gpu="L40S", volumes={"/cache": vol}, timeout=900)
def generar(prompts: list) -> list:
    import torch, io
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained("unsloth/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    out = []
    for i, p in enumerate(prompts):
        im = pipe(p, width=768, height=1344, num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256,
                  generator=torch.Generator("cuda").manual_seed(11 + i)).images[0]
        b = io.BytesIO(); im.save(b, "PNG"); out.append(b.getvalue())
    return out


@app.local_entrypoint()
def main(prompts: str, outdir: str):
    d = json.load(open(prompts)); est = ESTILOS[int(d.get("estilo", 0)) % len(ESTILOS)]
    claves = list(d["imgs"]); o = pathlib.Path(outdir); o.mkdir(parents=True, exist_ok=True)
    t = time.time()
    for i, b in enumerate(generar.remote([f"{est} {d['imgs'][k]}{COMUN}" for k in claves])):
        (o / f"{i}.png").write_bytes(b)
    (o / "claves.json").write_text(json.dumps(claves))
    print(f"{len(claves)} imágenes en {round(time.time() - t)} s")
