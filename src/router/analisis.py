"""Funciones de análisis de resultados para el notebook 03."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from router.config import RAIZ

NOMBRES = {
    "zeroshot": "Zero-shot",
    "fewshot": "Few-shot (5)",
    "lora": "LoRA",
    "mica": "MiCA",
    "frod": "FRoD",
    "unilora": "Uni-LoRA",
    "shira": "SHiRA",
}


def capacidad(cfg: dict) -> str:
    for campo in ("r", "sparse_rate", "theta_d_length"):
        if cfg.get(campo) is not None:
            return f"{campo}={cfg[campo]:g}"
    return ""


def cargar_corridas(dir_runs: Path | None = None) -> pd.DataFrame:
    """Una fila por corrida con configuración, métricas de val y test, recursos y tiempos."""
    filas = []
    for ruta in sorted((dir_runs or RAIZ / "results" / "runs").glob("*.json")):
        r = json.loads(ruta.read_text(encoding="utf-8"))
        c, e = r["config"], r.get("entrenamiento", {})
        fila = {
            "run_id": c["run_id"],
            "bloque": c["bloque"],
            "metodo": c["metodo"],
            "nombre": NOMBRES.get(c["metodo"], c["metodo"]),
            "capacidad": capacidad(c),
            "r": c.get("r"),
            "sparse_rate": c.get("sparse_rate"),
            "theta_d_length": c.get("theta_d_length"),
            "lr": c["lr"] if r["params_entrenables"] else np.nan,
            "replica": c["replica"],
            "seed": c["seed"],
            "params": r["params_entrenables"],
            "log10_params": np.log10(r["params_entrenables"]) if r["params_entrenables"] else np.nan,
            "init_min": r.get("init_adapter_segundos", 0) / 60,
            "train_min": e.get("train_segundos", 0) / 60,
            "total_min": r["total_segundos"] / 60,
            "tokens_por_seg": e.get("train_tokens_por_seg"),
            "vram_train_gb": e.get("vram_pico_train_gb"),
            "vram_inferencia_gb": r.get("vram_pico_inferencia_gb"),
            "energia_train_wh": e.get("energia_wh"),
            "potencia_train_w": e.get("potencia_media_w"),
            "energia_test_wh": r["test"].get("energia_wh"),
            "latencia_ms": r["latencia_ms"],
            "latencia_sin_merge_ms": r.get("latencia_ms_sin_merge"),
            "eval_merged": r.get("eval_merged", False),
            "loss_final": e.get("loss_final"),
            "gpu": r["entorno"]["gpu"],
        }
        for split in ("val", "test"):
            for k, v in r[split].items():
                if isinstance(v, (int, float)) and k not in ("n",):
                    fila[f"{split}_{k}"] = v
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_por_config(df: pd.DataFrame, metrica: str = "test_score") -> pd.DataFrame:
    """Media y desviación estándar entre réplicas para cada configuración."""
    claves = ["metodo", "nombre", "capacidad", "lr", "params", "log10_params"]
    agg = df.groupby(claves, dropna=False).agg(
        media=(metrica, "mean"),
        std=(metrica, "std"),
        n=(metrica, "size"),
        train_min=("train_min", "mean"),
        total_min=("total_min", "mean"),
        vram_train_gb=("vram_train_gb", "mean"),
        energia_train_wh=("energia_train_wh", "mean"),
        latencia_ms=("latencia_ms", "mean"),
    )
    return agg.reset_index().sort_values(["metodo", "params"])


def saturacion(x: np.ndarray, techo: float, amplitud: float, pendiente: float, centro: float) -> np.ndarray:
    """Curva logística en x = log10(parámetros). El score se aproxima a techo cuando x crece."""
    return techo - amplitud / (1 + np.exp(pendiente * (x - centro)))


def ajustar_saturacion(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ajusta la curva logística y devuelve los parámetros y el R²."""
    p0 = [y.max(), y.max() - y.min() + 1e-3, 2.0, float(np.median(x))]
    limites = ([0, 0, 0.1, x.min() - 3], [1.0, 1.0, 20, x.max() + 3])
    params, _ = curve_fit(saturacion, x, y, p0=p0, bounds=limites, maxfev=20000)
    pred = saturacion(x, *params)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return params, float(r2)


def punto_de_codo(params: np.ndarray, tolerancia: float = 0.005) -> float:
    """Menor log10(parámetros) desde el cual multiplicar por 10 los parámetros mejora el score
    en menos de `tolerancia`. A partir de ese punto, invertir más capacidad no es rentable."""
    xs = np.linspace(params[3] - 6, params[3] + 6, 4000)
    ganancia = saturacion(xs + 1, *params) - saturacion(xs, *params)
    candidatos = xs[(ganancia < tolerancia) & (xs > params[3])]
    return float(candidatos.min()) if len(candidatos) else float("nan")


def ajustar_superficie(df: pd.DataFrame, metrica: str = "val_score"):
    """Regresión cuadrática del score sobre x1 = log2(r) y x2 = log10(lr) con interacción.

    Devuelve el modelo de statsmodels, el punto estacionario (r*, lr*) y si ese punto es un máximo.
    """
    import statsmodels.formula.api as smf

    d = df.assign(x1=np.log2(df["r"].astype(float)), x2=np.log10(df["lr"].astype(float)), y=df[metrica])
    modelo = smf.ols("y ~ x1 + x2 + I(x1**2) + I(x2**2) + x1:x2", data=d).fit()
    b = modelo.params
    hess = np.array([[2 * b["I(x1 ** 2)"], b["x1:x2"]], [b["x1:x2"], 2 * b["I(x2 ** 2)"]]])
    grad0 = -np.array([b["x1"], b["x2"]])
    x1s, x2s = np.linalg.solve(hess, grad0)
    es_maximo = bool(np.all(np.linalg.eigvalsh(hess) < 0))
    return modelo, (float(2**x1s), float(10**x2s)), es_maximo


def frontera_pareto(df: pd.DataFrame, costo: str, valor: str = "media") -> pd.DataFrame:
    """Configuraciones no dominadas, es decir, sin otra de menor costo y mayor o igual valor."""
    orden = df.sort_values([costo, valor], ascending=[True, False])
    mejor, filas = -np.inf, []
    for _, f in orden.iterrows():
        if f[valor] > mejor:
            filas.append(f)
            mejor = f[valor]
    return pd.DataFrame(filas)
