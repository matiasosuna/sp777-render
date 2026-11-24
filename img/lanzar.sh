#!/bin/sh
# Imágenes de videos de estudio, desde la Mac (la GPU es de Modal): por id, FLUX → jpg → Telegram (practicas/imagenes.json).
# Reparte entre cuentas de Modal y usa cada una hasta su tope (gasto del mes en «Ephemeral Apps»; los USD 30 de cada una son gratis):
#   hermestarot444, familia y osuna-devs hasta 28 c/u; tarot no se usa (queda para Tarot y Myths).
#   Nunca kdp (suspendida). Cambiar topes con TOPES="perfil:tope …".
# Uso (desde spiritual-practices): boveda correr -p radar -- boveda correr -p spiritual-777 -- sh <rr>/img/lanzar.sh id…
set -e; M=$(cd "$(dirname "$0")" && pwd); TMP=${TMPDIR:-/tmp}/sp777-img
TOPES=${TOPES:-"hermestarot444:28 familia:28 osuna-devs:28"}
elegir() {  # primer perfil con gasto < tope; imprime «perfil gasto»
  for pt in $TOPES; do P=${pt%%:*}; T=${pt##*:}
    G=$(MODAL_PROFILE=$P modal billing summary 2>/dev/null | awk '/Ephemeral Apps/{print $3}')
    [ -n "$G" ] || continue
    awk "BEGIN{exit !($G < $T)}" && { echo "$P $G"; return 0; }
  done; return 1; }
for ID in "$@"; do
  PG=$(elegir) || { echo "todas las cuentas llegaron a su tope: corto"; exit 1; }
  P=${PG% *}; G=${PG#* }
  rm -rf "$TMP/$ID"; mkdir -p "$TMP/$ID"
  MODAL_PROFILE=$P modal run "$M/flux.py" --prompts "practicas/img/$ID.json" --outdir "$TMP/$ID" 2>&1 | tail -1
  for f in "$TMP/$ID"/*.png; do sips -s format jpeg -s formatOptions 88 "$f" --out "${f%.png}.jpg" >/dev/null && rm "$f"; done
  tar czf "$TMP/$ID.tgz" -C "$TMP/$ID" .
  TELEGRAM_CANAL_REPERTORIO=$TELEGRAM_CANAL_REPERTORIO python3 practicas/telegram.py subir-imgs "$TMP/$ID.tgz" "$ID"
  echo "$ID listo en $P (gasto del mes USD $G)"
done
