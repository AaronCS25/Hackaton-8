"""Métricas de evaluación del router."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from router import esquema

INVALIDO = "__invalido__"
CAMPOS_CLASE = {
    "intencion": esquema.INTENCIONES,
    "agente": esquema.AGENTES,
    "prioridad": esquema.PRIORIDADES,
}


def _slot_igual(campo: str, pred, real) -> bool:
    if real is None or pred is None:
        return pred is None and real is None
    if campo in ("n_personas", "monto"):
        try:
            return abs(float(pred) - float(real)) < 1e-6
        except (TypeError, ValueError):
            return False
    return esquema.normalizar_texto(pred) == esquema.normalizar_texto(real)


def calcular(predicciones: list[dict | None], referencias: list[dict]) -> dict:
    """predicciones contiene el JSON parseado de cada respuesta o None si no fue parseable."""
    n = len(referencias)
    validas = [esquema.salida_valida(p) for p in predicciones]
    res = {"n": n, "json_valido": float(np.mean(validas))}

    f1s = []
    for campo, clases in CAMPOS_CLASE.items():
        y_true = [r[campo] for r in referencias]
        y_pred = [p.get(campo, INVALIDO) if isinstance(p, dict) else INVALIDO for p in predicciones]
        y_pred = [y if y in clases else INVALIDO for y in y_pred]
        f1 = f1_score(y_true, y_pred, labels=list(clases), average="macro", zero_division=0)
        res[f"f1_{campo}"] = float(f1)
        res[f"acc_{campo}"] = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
        f1s.append(f1)
    res["score"] = float(np.mean(f1s))

    y_true = [r["requiere_aprobacion_humana"] for r in referencias]
    y_pred = [p.get("requiere_aprobacion_humana") if isinstance(p, dict) else None for p in predicciones]
    y_pred = [y if isinstance(y, bool) else (not t) for y, t in zip(y_pred, y_true)]
    res["f1_aprobacion"] = float(f1_score(y_true, y_pred, zero_division=0))

    aciertos_slots = []
    for campo in esquema.SLOTS:
        ok = []
        for p, r in zip(predicciones, referencias):
            slots = p.get("slots") if isinstance(p, dict) else None
            pred = slots.get(campo) if isinstance(slots, dict) else "__ausente__"
            ok.append(_slot_igual(campo, pred, r["slots"][campo]))
        res[f"acc_slot_{campo}"] = float(np.mean(ok))
        aciertos_slots.append(ok)
    res["acc_slots"] = float(np.mean(aciertos_slots))

    exacto = []
    for p, r, v in zip(predicciones, referencias, validas):
        if not v:
            exacto.append(False)
            continue
        iguales = all(p[c] == r[c] for c in ("intencion", "agente", "prioridad", "requiere_aprobacion_humana"))
        iguales = iguales and all(_slot_igual(c, p["slots"][c], r["slots"][c]) for c in esquema.SLOTS)
        exacto.append(iguales)
    res["exact_match"] = float(np.mean(exacto))
    return res
