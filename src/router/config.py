"""Configuración de un experimento y construcción del adapter PEFT correspondiente."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
MODELO_BASE = "ibm-granite/granite-4.2-3b"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
METODOS = ("zeroshot", "fewshot", "lora", "mica", "frod", "unilora", "shira")


@dataclasses.dataclass
class ExperimentConfig:
    """Hiperparámetros de una corrida. Cada corrida se identifica por run_id."""

    run_id: str
    bloque: str
    metodo: str
    seed: int = 1234
    replica: int = 1
    lr: float = 2e-4
    # Capacidad del adapter. Solo se usa el campo que corresponde al método.
    r: int | None = None
    alpha_ratio: float = 2.0
    sparse_rate: float | None = None
    theta_d_length: int | None = None
    # Entrenamiento
    epochs: int = 2
    # El batch efectivo es micro_batch x grad_accum = 16. Ambos se ajustan por entorno según la VRAM.
    micro_batch: int = dataclasses.field(default_factory=lambda: int(os.environ.get("ROUTER_MICRO_BATCH", "8")))
    grad_accum: int = dataclasses.field(default_factory=lambda: int(os.environ.get("ROUTER_GRAD_ACCUM", "2")))
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_len: int = 1024
    # En GPUs de 80 GB se desactiva con ROUTER_GRADIENT_CHECKPOINTING=0 para entrenar más rápido.
    gradient_checkpointing: bool = dataclasses.field(
        default_factory=lambda: os.environ.get("ROUTER_GRADIENT_CHECKPOINTING", "1") == "1"
    )
    # Evaluación
    eval_batch: int = 32
    max_new_tokens: int = 160
    n_latencia: int = 16
    n_fewshot: int = 5

    @property
    def entrena(self) -> bool:
        return self.metodo not in ("zeroshot", "fewshot")

    def a_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def desde_dict(cls, d: dict) -> "ExperimentConfig":
        campos = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in campos})


def construir_peft_config(cfg: ExperimentConfig):
    """Devuelve el objeto de configuración PEFT del método. Las semillas internas siguen a cfg.seed."""
    import peft

    if cfg.metodo in ("lora", "mica"):
        return peft.LoraConfig(
            r=cfg.r,
            lora_alpha=int(cfg.alpha_ratio * cfg.r),
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
            init_lora_weights="mica" if cfg.metodo == "mica" else True,
            task_type="CAUSAL_LM",
        )
    if cfg.metodo == "frod":
        return peft.FrodConfig(
            sparse_rate=cfg.sparse_rate,
            target_modules=TARGET_MODULES,
            projection_prng_key=cfg.seed,
            task_type="CAUSAL_LM",
        )
    if cfg.metodo == "unilora":
        return peft.UniLoraConfig(
            r=cfg.r or 4,
            theta_d_length=cfg.theta_d_length,
            proj_seed=cfg.seed,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
    if cfg.metodo == "shira":
        return peft.ShiraConfig(
            r=cfg.r,
            mask_type="random",
            random_seed=cfg.seed,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
    raise ValueError(f"metodo sin adapter {cfg.metodo}")


def guardar_json(ruta: Path, obj: dict) -> None:
    """Escritura atómica para que una corrida interrumpida no deje archivos a medias."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(ruta)
