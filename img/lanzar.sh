#!/bin/sh
# Imágenes de videos de estudio, desde la Mac (la GPU es de Modal): por id, FLUX → jpg → Telegram (practicas/imagenes.json).
# Reparte entre cuentas de Modal: usa cada una mientras el crédito gastado del mes («Credits» de modal billing summary;
# los USD 30 de cada una son gratis) esté debajo de su tope y no tenga nada cobrado («Billed Cost» 0). Si todas llegaron, corta.
#   hermestarot444, familia y osuna-devs hasta 28 c/u, en ese orden; tarot no se usa (queda para Tarot y Myths).
#   Nunca kdp (suspendida). Cambiar topes con TOPES="perfil:tope …".
# Sale con 3 si todas las cuentas llegaron al tope, con 1 si falló un id (no sube nada de ese id).
# Uso (desde spiritual-practices): boveda correr -p radar -- boveda correr -p spiritual-777 -- sh <rr>/img/lanzar.sh id…
#      sh lanzar.sh --elegir    (solo dice qué cuenta usaría)
M=$(cd "$(dirname "$0")" && pwd); TMP=${TMPDIR:-/tmp}/sp777-img
TOPES=${TOPES:-"hermestarot444:28 familia:28 osuna-devs:28"}
gasto() {  # «crédito_usado cobrado» de un perfil, o nada si no se pudo leer
  MODAL_PROFILE=$1 modal billing summary 2>/dev/null |
    awk '/Credits:/{c=-$2} /Billed Cost:/{b=$3; gsub(/\$/,"",b)} END{if (c != "" && b != "") print c, b}'; }
elegir() {  # primer perfil con crédito usado < tope y cobrado 0; imprime «perfil gasto»
  for pt in $TOPES; do P=${pt%%:*}; T=${pt##*:}
    GB=$(gasto "$P"); [ -n "$GB" ] || continue
    G=${GB% *}; B=${GB#* }
    awk "BEGIN{exit !($G < $T && $B == 0)}" && { echo "$P $G"; return 0; }
  done; return 1; }
if [ "${1:-}" = "--elegir" ]; then elegir; exit $?; fi
for ID in "$@"; do
  J="practicas/img/$ID.json"
  N=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));assert int(d['estilo']) in (0,2),'estilo';print(len(d['imgs']))" "$J" 2>/dev/null) ||
    { echo "$ID: $J inválido o con estilo descartado"; exit 1; }
  PG=$(elegir) || { echo "todas las cuentas llegaron a su tope: corto"; exit 3; }
  P=${PG% *}; G=${PG#* }
  rm -rf "$TMP/$ID"; mkdir -p "$TMP/$ID"
  MODAL_PROFILE=$P modal run "$M/flux.py" --prompts "$J" --outdir "$TMP/$ID" > "$TMP/$ID.log" 2>&1
  H=$(ls "$TMP/$ID"/*.png 2>/dev/null | wc -l | tr -d ' ')
  [ "$H" = "$N" ] || { echo "$ID: Modal hizo $H de $N imágenes en $P ($(tail -1 "$TMP/$ID.log"))"; exit 1; }
  for f in "$TMP/$ID"/*.png; do sips -s format jpeg -s formatOptions 88 "$f" --out "${f%.png}.jpg" >/dev/null && rm "$f" || exit 1; done
  tar czf "$TMP/$ID.tgz" -C "$TMP/$ID" . || exit 1
  python3 practicas/telegram.py subir-imgs "$TMP/$ID.tgz" "$ID" || { echo "$ID: no se pudo subir a Telegram"; exit 1; }
  echo "$ID listo en $P (crédito usado del mes USD $G)"
done
