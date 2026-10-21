"""Cola de corridas con un worker por GPU.

Cada worker toma la siguiente corrida pendiente y la ejecuta en un proceso nuevo con una sola GPU
visible, lo que aísla la medición de VRAM y tiempo. La cola es reanudable porque las corridas con
resultado guardado se omiten.

Uso
    uv run python scripts/cola.py --bloque lora --gpus 4
    uv run python scripts/cola.py --bloque lora,metodos --gpus 2 --lrs '{"mica": 1e-4, "frod": 1e-4}'
    uv run python scripts/cola.py --bloque validacion --r 48 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from router import grilla  # noqa: E402

DIR_COLA = RAIZ / "results" / "cola"
DIR_LOCKS = RAIZ / "results" / "locks"
DIR_LOGS = RAIZ / "results" / "logs"
DIR_RUNS = RAIZ / "results" / "runs"
DIR_FALLOS = RAIZ / "results" / "fallos"

# FRoD es el método más lento, por lo que se programa primero para equilibrar la carga entre GPUs.
ORDEN = {"frod": 0, "lora": 1, "shira": 2, "mica": 3, "unilora": 4, "zeroshot": 5, "fewshot": 6}


def tomar(run_id: str) -> bool:
    try:
        fd = os.open(DIR_LOCKS / f"{run_id}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def worker(gpu: int, run_ids: list[str], log) -> None:
    for run_id in run_ids:
        if (DIR_RUNS / f"{run_id}.json").exists() or not tomar(run_id):
            continue
        entorno = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        t0 = time.time()
        log(f"gpu{gpu} inicia {run_id}")
        with open(DIR_LOGS / f"{run_id}.log", "w", encoding="utf-8") as f:
            proc = subprocess.run(
                [sys.executable, str(RAIZ / "scripts" / "correr.py"), str(DIR_COLA / f"{run_id}.json")],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=entorno,
            )
        minutos = (time.time() - t0) / 60
        if proc.returncode == 0:
            log(f"gpu{gpu} termina {run_id} en {minutos:.1f} min")
        else:
            DIR_FALLOS.mkdir(parents=True, exist_ok=True)
            (DIR_FALLOS / f"{run_id}.txt").write_text(f"returncode {proc.returncode}\n", encoding="utf-8")
            log(f"gpu{gpu} FALLA {run_id} (ver results/logs/{run_id}.log)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bloque", required=True)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--lrs", type=json.loads, default=None)
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--solo", nargs="*", default=None, help="subconjunto de run_ids")
    a = p.parse_args()

    kw = {k: v for k, v in (("lrs", a.lrs), ("r", a.r), ("lr", a.lr)) if v is not None}
    # Varios bloques separados por coma comparten la misma cola para equilibrar la carga.
    configs = [c for b in a.bloque.split(",") for c in grilla.bloque(b, **kw)]
    vistos: set[str] = set()
    configs = [c for c in configs if not (c.run_id in vistos or vistos.add(c.run_id))]
    if a.solo:
        configs = [c for c in configs if c.run_id in a.solo]
    configs.sort(key=lambda c: (ORDEN.get(c.metodo, 9), c.replica))
    for d in (DIR_COLA, DIR_LOCKS, DIR_LOGS, DIR_RUNS):
        d.mkdir(parents=True, exist_ok=True)
    for c in configs:
        (DIR_COLA / f"{c.run_id}.json").write_text(json.dumps(c.a_dict(), indent=2), encoding="utf-8")
        lock = DIR_LOCKS / f"{c.run_id}.lock"
        if lock.exists() and not (DIR_RUNS / f"{c.run_id}.json").exists():
            lock.unlink()  # lock de una ejecución anterior interrumpida

    pendientes = [c.run_id for c in configs if not (DIR_RUNS / f"{c.run_id}.json").exists()]
    cerrojo = threading.Lock()

    def log(msg: str) -> None:
        with cerrojo:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    log(f"bloque {a.bloque} con {len(configs)} corridas, {len(pendientes)} pendientes, {a.gpus} GPUs")
    hilos = [threading.Thread(target=worker, args=(g, pendientes, log)) for g in range(a.gpus)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    fallos = [r for r in pendientes if not (DIR_RUNS / f"{r}.json").exists()]
    log(f"fin del bloque {a.bloque}, fallos {fallos}")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
