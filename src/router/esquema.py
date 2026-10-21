"""Esquema de salida del router y reglas deterministas de etiquetado.

La anotación del dataset define la intención, los atributos factuales y los slots de cada mensaje.
Las etiquetas agente, prioridad y requiere_aprobacion_humana se derivan de esos datos con las
funciones de este módulo, de modo que todo el dataset sigue exactamente las mismas reglas.
"""

from __future__ import annotations

import json
import re
import unicodedata

INTENCIONES = (
    "cotizar",
    "reservar",
    "modificar",
    "cancelar",
    "reclamo",
    "pago_reembolso",
    "emergencia_en_viaje",
    "consulta_info",
)
AGENTES = ("logistica", "finanzas", "seguridad", "clima_info", "humano")
PRIORIDADES = ("P1", "P2", "P3", "P4")
PLAZOS = ("en_curso", "menos_24h", "hasta_72h", "mas_72h", "sin_fecha")
SLOTS = ("destino", "fecha", "n_personas", "monto")

# Atributos factuales que se anotan en cada mensaje.
ATRIBUTOS = {
    "riesgo_personas": bool,
    "plazo": str,
    "dinero_en_disputa": bool,
    "mueve_dinero": bool,
    "pide_humano_o_legal": bool,
    "datos_personales": bool,
    "excepcion": bool,
    "tema_pagos": bool,
}

CAMPOS_SALIDA = ("intencion", "agente", "prioridad", "requiere_aprobacion_humana", "slots")


def derivar_agente(intencion: str, a: dict) -> str:
    if a["riesgo_personas"]:
        return "seguridad"
    if a["pide_humano_o_legal"] or intencion == "reclamo":
        return "humano"
    if intencion == "pago_reembolso":
        return "finanzas"
    if intencion == "consulta_info":
        return "finanzas" if a["tema_pagos"] else "clima_info"
    return "logistica"


def derivar_prioridad(intencion: str, a: dict) -> str:
    if a["riesgo_personas"] or intencion == "emergencia_en_viaje":
        return "P1"
    if a["plazo"] in ("en_curso", "menos_24h", "hasta_72h") or a["dinero_en_disputa"] or a["pide_humano_o_legal"]:
        return "P2"
    if a["plazo"] == "mas_72h":
        return "P3"
    return "P4"


def derivar_aprobacion(intencion: str, a: dict) -> bool:
    return bool(
        intencion == "cancelar"
        or a["mueve_dinero"]
        or a["datos_personales"]
        or a["excepcion"]
        or a["riesgo_personas"]
    )


def construir_salida(intencion: str, atributos: dict, slots: dict) -> dict:
    """Arma el JSON objetivo con el orden de campos fijo."""
    return {
        "intencion": intencion,
        "agente": derivar_agente(intencion, atributos),
        "prioridad": derivar_prioridad(intencion, atributos),
        "requiere_aprobacion_humana": derivar_aprobacion(intencion, atributos),
        "slots": {k: slots.get(k) for k in SLOTS},
    }


def a_texto(salida: dict) -> str:
    """Serializa la salida en una sola línea, tal como debe generarla el modelo."""
    return json.dumps(salida, ensure_ascii=False, separators=(", ", ": "))


def validar_atributos(intencion: str, a: dict, slots: dict) -> list[str]:
    errores = []
    if intencion not in INTENCIONES:
        errores.append(f"intencion invalida {intencion!r}")
    for nombre, tipo in ATRIBUTOS.items():
        if nombre not in a:
            errores.append(f"falta atributo {nombre}")
        elif not isinstance(a[nombre], tipo):
            errores.append(f"tipo invalido en {nombre}")
    if a.get("plazo") not in PLAZOS:
        errores.append(f"plazo invalido {a.get('plazo')!r}")
    if set(slots) != set(SLOTS):
        errores.append(f"slots con claves {sorted(slots)}")
    if slots.get("n_personas") is not None and not isinstance(slots["n_personas"], int):
        errores.append("n_personas debe ser int o null")
    if slots.get("monto") is not None and not isinstance(slots["monto"], (int, float)):
        errores.append("monto debe ser numerico o null")
    for k in ("destino", "fecha"):
        if slots.get(k) is not None and not isinstance(slots[k], str):
            errores.append(f"{k} debe ser str o null")
    # Coherencia mínima entre intención y atributos.
    if intencion == "emergencia_en_viaje" and a.get("plazo") != "en_curso":
        errores.append("emergencia_en_viaje requiere plazo en_curso")
    if a.get("riesgo_personas") and intencion != "emergencia_en_viaje":
        errores.append("riesgo_personas solo aplica a emergencia_en_viaje")
    if a.get("tema_pagos") and intencion != "consulta_info":
        errores.append("tema_pagos solo aplica a consulta_info")
    return errores


def normalizar_texto(s: str | None) -> str | None:
    """Minúsculas, sin tildes ni espacios extra. Se usa para comparar slots de texto."""
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip(" .,")


def parsear_salida(texto: str) -> dict | None:
    """Extrae el primer objeto JSON de la respuesta del modelo. Devuelve None si no es válido."""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()
    inicio = texto.find("{")
    if inicio < 0:
        return None
    profundidad = 0
    for i, c in enumerate(texto[inicio:], start=inicio):
        if c == "{":
            profundidad += 1
        elif c == "}":
            profundidad -= 1
            if profundidad == 0:
                try:
                    obj = json.loads(texto[inicio : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def salida_valida(obj: dict | None) -> bool:
    """Verifica que la salida cumpla el esquema completo con valores permitidos."""
    if not isinstance(obj, dict) or set(obj) != set(CAMPOS_SALIDA):
        return False
    if obj["intencion"] not in INTENCIONES or obj["agente"] not in AGENTES or obj["prioridad"] not in PRIORIDADES:
        return False
    if not isinstance(obj["requiere_aprobacion_humana"], bool):
        return False
    return isinstance(obj["slots"], dict) and set(obj["slots"]) == set(SLOTS)
