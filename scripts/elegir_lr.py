"""Elige el learning rate de cada método según el score en validación del bloque barrido_lr.

También asigna el prefijo barrido_ a los resultados de ese bloque, de modo que el bloque metodos
vuelva a correr la configuración elegida con la evaluación sobre el adapter fusionado.
Imprime un JSON {metodo: lr} apto para scripts/cola.py --lrs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "results" / "runs"
PREDS = RAIZ / "results" / "preds"


def main() -> None:
    mejores: dict[str, tuple[float, float]] = {}
    for ruta in sorted(RUNS.glob("*.json")):
        r = json.loads(ruta.read_text(encoding="utf-8"))
        cfg = r["config"]
        if cfg["bloque"] != "barrido_lr":
            continue
        if not cfg["run_id"].startswith("barrido_"):
            nuevo = f"barrido_{cfg['run_id']}"
            r["config"]["run_id"] = nuevo
            (RUNS / f"{nuevo}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            ruta.unlink()
            for split in ("val", "test"):
                p = PREDS / f"{cfg['run_id']}_{split}.jsonl"
                if p.exists():
                    p.rename(PREDS / f"{nuevo}_{split}.jsonl")
        score = r["val"]["score"]
        if cfg["metodo"] not in mejores or score > mejores[cfg["metodo"]][0]:
            mejores[cfg["metodo"]] = (score, cfg["lr"])
    for m, (s, lr) in sorted(mejores.items()):
        print(f"{m}: lr {lr:g} con score val {s:.4f}", file=sys.stderr)
    print(json.dumps({m: lr for m, (_, lr) in mejores.items()}))


if __name__ == "__main__":
    main()
