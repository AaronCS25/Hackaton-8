"""Ejecuta una sola corrida a partir de un JSON de configuración.

Uso
    uv run python scripts/correr.py results/cola/<run_id>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.config import ExperimentConfig  # noqa: E402
from router.experimento import correr  # noqa: E402

if __name__ == "__main__":
    cfg = ExperimentConfig.desde_dict(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    res = correr(cfg)
    resumen = {k: round(res["test"][k], 4) for k in ("score", "json_valido", "exact_match", "acc_slots")}
    print(cfg.run_id, "test", resumen, "params", res["params_entrenables"])
