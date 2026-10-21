"""Ejecución de una corrida completa, que incluye entrenamiento, evaluación y medición de recursos."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from router import datos, esquema, metricas
from router.config import MODELO_BASE, RAIZ, ExperimentConfig, construir_peft_config, guardar_json

DIR_RESULTADOS = Path(os.environ.get("ROUTER_RESULTADOS", RAIZ / "results"))
DIR_RUNS = DIR_RESULTADOS / "runs"
DIR_PREDS = DIR_RESULTADOS / "preds"


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sincronizar() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MedidorEnergia:
    """Muestrea la potencia de la GPU visible en un hilo aparte e integra la energía en Wh."""

    def __init__(self, intervalo: float = 0.5):
        self.intervalo = intervalo
        self.muestras: list[float] = []
        self._activo = False

    def _bucle(self) -> None:
        while self._activo:
            try:
                self.muestras.append(torch.cuda.power_draw(0) / 1000)
            except Exception:  # NVML no disponible
                return
            time.sleep(self.intervalo)

    def __enter__(self):
        import threading

        self._activo = True
        self._t0 = time.perf_counter()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._activo = False
        self._hilo.join()
        self.segundos = time.perf_counter() - self._t0
        self.potencia_media_w = float(np.mean(self.muestras)) if self.muestras else None
        self.energia_wh = self.potencia_media_w * self.segundos / 3600 if self.muestras else None


def cargar_modelo_y_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    return modelo, tokenizer


def contar_parametros(modelo) -> tuple[int, int]:
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    return entrenables, total


def perdida_cola(modelo, batch: dict) -> torch.Tensor:
    """Cross-entropy calculada solo sobre la cola de cada fila.

    Con padding a la izquierda, todos los tokens con pérdida están dentro de las últimas k posiciones.
    Proyectar únicamente esas posiciones al vocabulario evita materializar logits de tamaño
    seq_len x vocab, que dominan la memoria con prompts largos.
    """
    labels = batch["labels"]
    con_perdida = (labels != -100).any(dim=0).nonzero()
    k = int(labels.shape[1] - con_perdida[0]) + 1
    salida = modelo(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        logits_to_keep=k,
        use_cache=False,
    )
    logits = salida.logits[:, :-1, :].float()
    objetivo = labels[:, labels.shape[1] - k + 1 :]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), objetivo.reshape(-1), ignore_index=-100
    )


def entrenar(modelo, tokenizer, cfg: ExperimentConfig, train: list[dict]) -> dict:
    ds = datos.DatasetSFT(train, tokenizer, cfg.max_len)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.micro_batch,
        shuffle=True,
        generator=gen,
        collate_fn=lambda b: datos.collate(b, tokenizer.pad_token_id),
    )
    pasos_por_epoca = math.ceil(len(loader) / cfg.grad_accum)
    pasos_totales = pasos_por_epoca * cfg.epochs
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    optimizador = torch.optim.AdamW(parametros, lr=cfg.lr, weight_decay=cfg.weight_decay)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(optimizador, int(cfg.warmup_ratio * pasos_totales), pasos_totales)

    modelo.train()
    perdidas, tokens = [], 0
    torch.cuda.reset_peak_memory_stats()
    sincronizar()
    t0 = time.perf_counter()
    paso = 0
    for _ in range(cfg.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(modelo.device) for k, v in batch.items()}
            tokens += int(batch["attention_mask"].sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perdida = perdida_cola(modelo, batch)
            (perdida / cfg.grad_accum).backward()
            perdidas.append(float(perdida.detach()))
            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(parametros, 1.0)
                optimizador.step()
                scheduler.step()
                optimizador.zero_grad(set_to_none=True)
                paso += 1
    sincronizar()
    segundos = time.perf_counter() - t0
    return {
        "train_segundos": segundos,
        "train_tokens": tokens,
        "train_tokens_por_seg": tokens / segundos,
        "pasos": paso,
        "vram_pico_train_gb": torch.cuda.max_memory_allocated() / 2**30,
        "vram_reservada_train_gb": torch.cuda.max_memory_reserved() / 2**30,
        "loss_inicial": float(np.mean(perdidas[:5])),
        "loss_final": float(np.mean(perdidas[-10:])),
        "curva_loss": perdidas,
    }


@torch.no_grad()
def generar(modelo, tokenizer, textos: list[str], cfg: ExperimentConfig, few_shot=None, batch: int | None = None):
    """Generación greedy en lotes con padding a la izquierda. Devuelve textos y segundos totales."""
    tokenizer.padding_side = "left"
    batch = batch or cfg.eval_batch
    salidas = []
    sincronizar()
    t0 = time.perf_counter()
    for i in range(0, len(textos), batch):
        prompts_lote = [datos.texto_prompt(tokenizer, t, few_shot) for t in textos[i : i + batch]]
        enc = tokenizer(prompts_lote, return_tensors="pt", padding=True, add_special_tokens=False).to(modelo.device)
        out = modelo.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        nuevos = out[:, enc["input_ids"].shape[1] :]
        salidas += tokenizer.batch_decode(nuevos, skip_special_tokens=True)
    sincronizar()
    return salidas, time.perf_counter() - t0


def evaluar(modelo, tokenizer, filas: list[dict], cfg: ExperimentConfig, few_shot=None) -> tuple[dict, list[dict]]:
    textos = [f["texto"] for f in filas]
    salidas, segundos = generar(modelo, tokenizer, textos, cfg, few_shot)
    parseadas = [esquema.parsear_salida(s) for s in salidas]
    res = metricas.calcular(parseadas, [f["salida"] for f in filas])
    res["gen_segundos_lote"] = segundos
    res["gen_ms_por_ejemplo_lote"] = 1000 * segundos / len(filas)
    preds = [{"id": f["id"], "respuesta": s, "parseada": p} for f, s, p in zip(filas, salidas, parseadas)]
    return res, preds


def medir_latencia(modelo, tokenizer, filas: list[dict], cfg: ExperimentConfig, few_shot=None, n=None) -> float:
    """Latencia media en ms por solicitud individual (batch 1), como en un servicio en línea."""
    textos = [f["texto"] for f in filas[: n or cfg.n_latencia]]
    generar(modelo, tokenizer, textos[:1], cfg, few_shot, batch=1)
    _, segundos = generar(modelo, tokenizer, textos, cfg, few_shot, batch=1)
    return 1000 * segundos / len(textos)


def correr(cfg: ExperimentConfig) -> dict:
    """Ejecuta una corrida y guarda results/runs/<run_id>.json. Si ya existe, la omite."""
    destino = DIR_RUNS / f"{cfg.run_id}.json"
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))
    import peft
    import transformers

    fijar_semilla(cfg.seed)
    train, val, test = datos.cargar_split("train"), datos.cargar_split("val"), datos.cargar_split("test")
    t_inicio = time.perf_counter()
    modelo, tokenizer = cargar_modelo_y_tokenizer()
    vram_modelo = torch.cuda.memory_allocated() / 2**30

    res: dict = {
        "config": cfg.a_dict(),
        "entorno": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "python": platform.python_version(),
        },
        "vram_modelo_gb": vram_modelo,
    }

    few_shot = datos.elegir_few_shot(train, cfg.n_fewshot) if cfg.metodo == "fewshot" else None
    if cfg.entrena:
        if cfg.gradient_checkpointing:
            modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            modelo.enable_input_require_grads()
        sincronizar()
        t0 = time.perf_counter()
        modelo = peft.get_peft_model(modelo, construir_peft_config(cfg))
        sincronizar()
        res["init_adapter_segundos"] = time.perf_counter() - t0
        res["params_entrenables"], res["params_totales"] = contar_parametros(modelo)
        with MedidorEnergia() as energia:
            res["entrenamiento"] = entrenar(modelo, tokenizer, cfg, train)
        res["entrenamiento"]["potencia_media_w"] = energia.potencia_media_w
        res["entrenamiento"]["energia_wh"] = energia.energia_wh
        modelo.eval()
        modelo.base_model.model.gradient_checkpointing_disable()
        modelo.base_model.model.config.use_cache = True
        # Latencia con el adapter sin fusionar, en pocos ejemplos porque algunos métodos
        # reconstruyen la actualización de pesos en cada forward.
        res["latencia_ms_sin_merge"] = medir_latencia(modelo, tokenizer, test, cfg, n=4)
        # La evaluación se hace con el adapter fusionado, que es la forma de despliegue.
        sincronizar()
        t0 = time.perf_counter()
        modelo = modelo.merge_and_unload()
        sincronizar()
        res["merge_segundos"] = time.perf_counter() - t0
        res["eval_merged"] = True
    else:
        modelo.eval()
        res["params_entrenables"], res["params_totales"] = 0, contar_parametros(modelo)[1]

    torch.cuda.reset_peak_memory_stats()
    res["val"], preds_val = evaluar(modelo, tokenizer, val, cfg, few_shot)
    with MedidorEnergia() as energia:
        res["test"], preds_test = evaluar(modelo, tokenizer, test, cfg, few_shot)
    res["test"]["energia_wh"] = energia.energia_wh
    res["vram_pico_inferencia_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["latencia_ms"] = medir_latencia(modelo, tokenizer, test, cfg, few_shot)
    res["total_segundos"] = time.perf_counter() - t_inicio

    for nombre, preds in (("val", preds_val), ("test", preds_test)):
        ruta = DIR_PREDS / f"{cfg.run_id}_{nombre}.jsonl"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds), encoding="utf-8")
    guardar_json(destino, res)
    return res
