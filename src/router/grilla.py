"""Definición de los bloques del diseño experimental.

Cada réplica usa una semilla distinta para toda la ronda, de modo que la réplica 1 corre completa con
SEMILLAS[1] y la réplica 2 con SEMILLAS[2]. La selección de learning rate y de r* se hace sobre el split
de validación, y el split de test se reserva para el reporte final.
"""

from __future__ import annotations

from router.config import ExperimentConfig

SEMILLAS = {1: 1234, 2: 5678}

# Learning rate elegido en validación para cada método. Se completa tras el bloque barrido_lr.
LR_ELEGIDO: dict[str, float] = {}

LORA_R = (2, 8, 32, 128)
LORA_LR = (1e-4, 2e-4, 5e-4)
CAPACIDADES = {
    "mica": ("r", (8, 32, 128)),
    "frod": ("sparse_rate", (0.005, 0.01, 0.02)),
    "unilora": ("theta_d_length", (2**12, 2**15, 2**18)),
    "shira": ("r", (8, 32, 128)),
}
BARRIDO_LR = {
    "mica": ("r", 32, (1e-4, 3e-4)),
    "frod": ("sparse_rate", 0.01, (1e-4, 3e-4)),
    "unilora": ("theta_d_length", 2**15, (2e-3, 8e-3)),
    "shira": ("r", 32, (1e-4, 3e-4)),
}


def _id(metodo: str, campo: str | None, valor, lr: float | None, replica: int) -> str:
    partes = [metodo]
    if campo:
        partes.append(f"{campo}{valor}")
    if lr is not None:
        partes.append(f"lr{lr:g}")
    partes.append(f"s{replica}")
    return "_".join(partes)


def _cfg(bloque: str, metodo: str, campo: str | None, valor, lr: float, replica: int) -> ExperimentConfig:
    kw = {campo: valor} if campo else {}
    if metodo == "frod":
        # FRoD necesita más memoria de activaciones. Se mantiene el batch efectivo de 16.
        kw.update(micro_batch=4, grad_accum=4)
    return ExperimentConfig(
        run_id=_id(metodo, campo, valor, lr, replica),
        bloque=bloque,
        metodo=metodo,
        seed=SEMILLAS[replica],
        replica=replica,
        lr=lr,
        **kw,
    )


def bloque(nombre: str, **kw) -> list[ExperimentConfig]:
    if nombre == "base":
        return [
            ExperimentConfig(run_id="zeroshot", bloque="base", metodo="zeroshot"),
            ExperimentConfig(run_id="fewshot", bloque="base", metodo="fewshot"),
        ]
    if nombre == "piloto":
        return [_cfg("piloto", "lora", "r", r, 2e-4, 1) for r in (1, 256)]
    if nombre == "lora":
        return [
            _cfg("lora", "lora", "r", r, lr, rep)
            for rep in (1, 2)
            for r in LORA_R
            for lr in LORA_LR
        ]
    if nombre == "barrido_lr":
        configs = [
            _cfg("barrido_lr", m, campo, valor, lr, 1)
            for m, (campo, valor, lrs) in BARRIDO_LR.items()
            for lr in lrs
        ]
        # Ids propios para que el bloque metodos vuelva a correr la configuración elegida con la
        # evaluación sobre el adapter fusionado.
        for c in configs:
            c.run_id = f"barrido_{c.run_id}"
        return configs
    if nombre == "metodos":
        lrs = kw.get("lrs") or LR_ELEGIDO
        return [
            _cfg("metodos", m, campo, valor, lrs[m], rep)
            for rep in (1, 2)
            for m, (campo, valores) in CAPACIDADES.items()
            if m in lrs
            for valor in valores
        ]
    if nombre == "validacion":
        return [_cfg("validacion", "lora", "r", kw["r"], kw["lr"], rep) for rep in (1, 2)]
    raise ValueError(f"bloque desconocido {nombre}")
