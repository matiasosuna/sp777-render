#!/bin/sh
# Imágenes de videos de estudio, desde la Mac (la GPU es de Modal, perfil familia): por id, FLUX → jpg → Telegram
# (practicas/imagenes.json). No corre si el gasto del mes de Modal llega a USD 27 (los 30 del plan son gratis).
# Uso (desde spiritual-practices): boveda correr -p radar -- boveda correr -p spiritual-777 -- sh <rr>/img/lanzar.sh id…
set -e; M=$(cd "$(dirname "$0")" && pwd); TMP=${TMPDIR:-/tmp}/sp777-img
for ID in "$@"; do
  G=$(MODAL_PROFILE=familia modal billing summary 2>/dev/null | awk '/Ephemeral Apps/{print $3}')
  [ -n "$G" ] || { echo "no pude leer el gasto de Modal: corto"; exit 1; }
  awk "BEGIN{exit !($G >= 27)}" && { echo "gasto del mes USD $G: tope alcanzado, corto"; exit 1; }
  rm -rf "$TMP/$ID"; mkdir -p "$TMP/$ID"
  MODAL_PROFILE=familia modal run "$M/flux.py" --prompts "practicas/img/$ID.json" --outdir "$TMP/$ID" 2>&1 | tail -1
  for f in "$TMP/$ID"/*.png; do sips -s format jpeg -s formatOptions 88 "$f" --out "${f%.png}.jpg" >/dev/null && rm "$f"; done
  tar czf "$TMP/$ID.tgz" -C "$TMP/$ID" .
  TELEGRAM_CANAL_REPERTORIO=$TELEGRAM_CANAL_REPERTORIO python3 practicas/telegram.py subir-imgs "$TMP/$ID.tgz" "$ID"
  echo "$ID listo (gasto del mes USD $G)"
done
