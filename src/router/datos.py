"""Carga de splits y tokenización para entrenamiento y generación."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from router import esquema, prompts
from router.config import RAIZ

# ROUTER_DATOS permite apuntar a otro directorio, por ejemplo en pruebas de humo.
DIR_PROCESADO = Path(os.environ.get("ROUTER_DATOS", RAIZ / "data" / "processed"))


def cargar_split(nombre: str) -> list[dict]:
    ruta = DIR_PROCESADO / f"{nombre}.jsonl"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def elegir_few_shot(train: list[dict], k: int, seed: int = 0) -> list[tuple[str, str]]:
    """Toma k ejemplos de train con intenciones distintas para el baseline few-shot."""
    rng = random.Random(seed)
    por_intencion: dict[str, list[dict]] = {}
    for f in train:
        por_intencion.setdefault(f["salida"]["intencion"], []).append(f)
    intenciones = sorted(por_intencion)
    rng.shuffle(intenciones)
    elegidos = [rng.choice(por_intencion[i]) for i in intenciones[:k]]
    return [(f["texto"], esquema.a_texto(f["salida"])) for f in elegidos]


def texto_prompt(tokenizer, texto: str, few_shot=None) -> str:
    return tokenizer.apply_chat_template(
        prompts.mensajes(texto, few_shot), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


class DatasetSFT(torch.utils.data.Dataset):
    """Cada ítem contiene input_ids y labels. La pérdida se calcula solo sobre el JSON objetivo y el token de fin."""

    def __init__(self, filas: list[dict], tokenizer, max_len: int):
        self.items = []
        for f in filas:
            prompt_ids = tokenizer(texto_prompt(tokenizer, f["texto"]), add_special_tokens=False)["input_ids"]
            objetivo = esquema.a_texto(f["salida"]) + tokenizer.eos_token
            target_ids = tokenizer(objetivo, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + target_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def n_tokens(self) -> int:
        return sum(len(ids) for ids, _ in self.items)


def collate(batch, pad_id: int):
    """Padding a la izquierda. Así los tokens con pérdida quedan al final de cada fila y basta con
    calcular logits sobre las últimas posiciones (ver experimento.perdida_cola)."""
    largo = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), largo), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), largo), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), largo), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, largo - len(ids) :] = torch.tensor(ids)
        labels[i, largo - len(lab) :] = torch.tensor(lab)
        mask[i, largo - len(ids) :] = 1
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask, "position_ids": position_ids}
