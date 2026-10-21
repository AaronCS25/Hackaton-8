#!/usr/bin/env bash
# Descarga el modelo una vez y ejecuta los bloques indicados en orden.
# Un bloque con fallos no detiene a los siguientes. El código de salida indica si hubo fallos.
# Uso
#   bash scripts/correr_bloques.sh <n_gpus> <bloque> [<bloque> ...]
set -u
GPUS="$1"
shift
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.2-3b')"
estado=0
for b in "$@"; do
  uv run python scripts/cola.py --bloque "$b" --gpus "$GPUS" || estado=1
done
exit $estado
