#!/usr/bin/env bash
# Elige el learning rate de cada método en validación y ejecuta la cola principal (LoRA factorial y métodos).
# Uso
#   bash scripts/lanzar_principal.sh <n_gpus>
set -eu
GPUS="$1"
LRS="$(uv run python scripts/elegir_lr.py)"
echo "learning rates elegidos ${LRS}"
uv run python scripts/cola.py --bloque lora,metodos --gpus "$GPUS" --lrs "$LRS"
