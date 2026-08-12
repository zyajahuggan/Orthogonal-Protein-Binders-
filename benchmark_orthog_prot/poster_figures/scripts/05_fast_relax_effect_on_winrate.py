"""Poster version of py_rosetta/scripts/analysis/win_rate_delta.py's diverging bar
chart: change in best-metric win rate (fast_relax_riam - riam) per spm/dataset. Reads
the already-computed CSV instead of recomputing (win_rate_delta.py must be run first).

New file -- does not modify or overwrite win_rate_delta.py or its outputs.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "py_rosetta" / "results" / "win_rate_comparison" / "riam_vs_fast_relax_riam_win_rate_delta.csv"

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 25,
    "axes.titleweight": "bold",
    "axes.labelsize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

INCREASE_COLOR = "#2a78d6"
DECREASE_COLOR = "#e34948"

MODEL_LABELS = {"af3": "AF3", "boltz": "Boltz-2", "chai": "Chai", "esmfold2": "ESMFold2"}
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

df = pd.read_csv(CSV_PATH)

labels = [f"{MODEL_LABELS[row.spm]}\n{DATASET_LABELS[row.dataset]}" for row in df.itertuples()]
deltas = df["fast_relax_riam_vs_riam_delta"].tolist()
bar_colors = [INCREASE_COLOR if d >= 0 else DECREASE_COLOR for d in deltas]

fig, ax = plt.subplots(figsize=(13, 8))
bars = ax.bar(labels, deltas, color=bar_colors)
ax.axhline(0, color="#898781", linewidth=1.5)

for bar, delta in zip(bars, deltas):
    offset = 0.015 if delta >= 0 else -0.015
    va = "bottom" if delta >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, delta + offset, f"{delta:+.0%}",
            ha="center", va=va, fontsize=17)

ax.margins(y=0.2)
ax.set_ylabel("Change in Best-Metric Win Rate\n(Fast Relax RIAM − RIAM)")
ax.set_title("Effect of Fast Relax on Best-Metric Win Rate per SPM/Dataset")

increase_patch = plt.Rectangle((0, 0), 1, 1, color=INCREASE_COLOR, label="Win rate increased")
decrease_patch = plt.Rectangle((0, 0), 1, 1, color=DECREASE_COLOR, label="Win rate decreased")
ax.legend(handles=[increase_patch, decrease_patch])

plt.tight_layout()
plt.savefig(OUT_DIR / "fast_relax_effect_on_winrate.png", dpi=300)
plt.close()

print("saved fast_relax_effect_on_winrate.png")
