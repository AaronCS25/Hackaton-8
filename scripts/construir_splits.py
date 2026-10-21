"""Construye los splits train, val y test a partir de data/raw.

Se derivan las etiquetas con las reglas de esquema.py, se eliminan casi duplicados por similitud de
5-gramas de caracteres y se estratifica por intención dentro de cada origen (lote o gold).

Uso
    uv run python scripts/construir_splits.py
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from router import esquema  # noqa: E402

SEMILLA = 2026
UMBRAL_DUPLICADO = 0.7
# Cantidad por split según origen. Los ejemplos gold pesan más en test porque son los casos difíciles.
CUOTAS = {"gold": {"test": 100, "val": 20}, "lote": {"test": 100, "val": 80}}


def ngramas(texto: str, n: int = 5) -> set[str]:
    t = esquema.normalizar_texto(texto)
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cargar() -> list[dict]:
    filas = []
    for ruta in sorted((RAIZ / "data" / "raw").glob("*.jsonl")):
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                f = json.loads(linea)
                if not esquema.validar_atributos(f["intencion"], f["atributos"], f["slots"]):
                    filas.append(f)
    return filas


def quitar_casi_duplicados(filas: list[dict]) -> tuple[list[dict], int]:
    vistos, salida, quitados = [], [], 0
    for f in filas:
        g = ngramas(f["texto"])
        if any(jaccard(g, v) >= UMBRAL_DUPLICADO for v in vistos):
            quitados += 1
            continue
        vistos.append(g)
        salida.append(f)
    return salida, quitados


def estratificar(filas: list[dict], cuotas: dict[str, int], rng: random.Random) -> dict[str, list[dict]]:
    """Reparte cada intención en proporción a su peso para cumplir las cuotas de test y val."""
    por_intencion = collections.defaultdict(list)
    for f in filas:
        por_intencion[f["intencion"]].append(f)
    splits = {"test": [], "val": [], "train": []}
    total = len(filas)
    for intencion in sorted(por_intencion):
        grupo = por_intencion[intencion]
        rng.shuffle(grupo)
        k_test = round(cuotas["test"] * len(grupo) / total)
        k_val = round(cuotas["val"] * len(grupo) / total)
        splits["test"] += grupo[:k_test]
        splits["val"] += grupo[k_test : k_test + k_val]
        splits["train"] += grupo[k_test + k_val :]
    return splits


def main() -> None:
    rng = random.Random(SEMILLA)
    filas = cargar()
    rng.shuffle(filas)
    # Los gold se procesan primero para que, ante un casi duplicado, se conserve la versión gold.
    filas.sort(key=lambda f: f["origen"] != "gold")
    filas, quitados = quitar_casi_duplicados(filas)

    final = {"train": [], "val": [], "test": []}
    for origen, cuotas in CUOTAS.items():
        subset = [f for f in filas if f["origen"] == origen]
        for nombre, lista in estratificar(subset, cuotas, rng).items():
            final[nombre] += lista

    destino = RAIZ / "data" / "processed"
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, lista in final.items():
        rng.shuffle(lista)
        registros = [
            {
                "id": f["id"],
                "texto": f["texto"],
                "salida": esquema.construir_salida(f["intencion"], f["atributos"], f["slots"]),
                "atributos": f["atributos"],
                "origen": f["origen"],
                "dificultad": f["dificultad"],
            }
            for f in lista
        ]
        (destino / f"{nombre}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
        )
        origenes = collections.Counter(f["origen"] for f in lista)
        print(f"{nombre}: {len(lista)} filas {dict(origenes)}")
    print(f"casi duplicados eliminados: {quitados}")


if __name__ == "__main__":
    main()
