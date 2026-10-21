"""Valida archivos JSONL del dataset crudo y reporta la distribución de etiquetas derivadas.

Uso
    uv run python scripts/validar_dataset.py data/raw/lote_01.jsonl [más archivos]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router import esquema  # noqa: E402

CAMPOS = {"id", "texto", "intencion", "atributos", "slots", "origen", "dificultad"}


def validar_archivo(ruta: Path) -> tuple[list[dict], list[str]]:
    filas, errores = [], []
    for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        if not linea.strip():
            continue
        try:
            fila = json.loads(linea)
        except json.JSONDecodeError as e:
            errores.append(f"{ruta.name}:{n} JSON invalido ({e})")
            continue
        faltan = CAMPOS - set(fila)
        if faltan:
            errores.append(f"{ruta.name}:{n} faltan campos {sorted(faltan)}")
            continue
        if not isinstance(fila["texto"], str) or len(fila["texto"].strip()) < 5:
            errores.append(f"{ruta.name}:{n} texto vacio o muy corto")
        for e in esquema.validar_atributos(fila["intencion"], fila["atributos"], fila["slots"]):
            errores.append(f"{ruta.name}:{n} ({fila['id']}) {e}")
        filas.append(fila)
    return filas, errores


def main(rutas: list[str]) -> int:
    todas, errores = [], []
    for r in rutas:
        filas, err = validar_archivo(Path(r))
        todas += filas
        errores += err
    ids = collections.Counter(f["id"] for f in todas)
    errores += [f"id duplicado {i}" for i, c in ids.items() if c > 1]
    textos = collections.Counter(f["texto"].strip().lower() for f in todas)
    errores += [f"texto duplicado {t[:60]!r}" for t, c in textos.items() if c > 1]

    print(f"filas {len(todas)} | errores {len(errores)}")
    for e in errores[:50]:
        print("  ERROR", e)
    validas = [f for f in todas if not esquema.validar_atributos(f["intencion"], f["atributos"], f["slots"])]
    salidas = [esquema.construir_salida(f["intencion"], f["atributos"], f["slots"]) for f in validas]
    for campo in ("intencion", "agente", "prioridad", "requiere_aprobacion_humana"):
        conteo = collections.Counter(str(s[campo]) for s in salidas)
        print(f"{campo}: {dict(sorted(conteo.items()))}")
    nulos = {k: sum(s["slots"][k] is None for s in salidas) for k in esquema.SLOTS}
    print(f"slots nulos: {nulos}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
