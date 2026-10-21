"""Estilo común de los gráficos. Cada método conserva el mismo color en todas las figuras."""

from __future__ import annotations

import matplotlib.pyplot as plt

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"

# Paleta categórica validada para daltonismo, asignada en orden fijo.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_METODO = {
    "LoRA": SERIES[0],
    "FRoD": SERIES[1],
    "MiCA": SERIES[2],
    "Uni-LoRA": SERIES[3],
    "SHiRA": SERIES[4],
}
MARCADOR_METODO = {"LoRA": "o", "FRoD": "s", "MiCA": "^", "Uni-LoRA": "D", "SHiRA": "v"}


def aplicar_estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": EJE,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.titlecolor": TINTA,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRILLA,
            "grid.linewidth": 0.6,
            "xtick.color": TINTA_TENUE,
            "ytick.color": TINTA_TENUE,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "font.family": "sans-serif",
            "figure.dpi": 110,
        }
    )
