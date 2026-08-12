"""Poster version of py_rosetta/scripts/analysis/fast_relax_vs_native_win_rate.py:
best-metric win rate, fast_relax_riam vs. the native spm that gave it its input
structures. Reads the already-computed CSV instead of recomputing (win_rate_delta.py
must be run first).

New file -- does not modify or overwrite fast_relax_vs_native_win_rate.py or its
outputs.
"""

from pathlib import Path
import numpy as np
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

NATIVE_COLOR = "#2a78d6"
FAST_RELAX_COLOR = "#4a3aa7"

MODEL_LABELS = {"af3": "AF3", "boltz": "Boltz-2", "chai": "Chai", "esmfold2": "ESMFold2"}
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

df = pd.read_csv(CSV_PATH)

group_labels = [f"{MODEL_LABELS[row.spm]}\n{DATASET_LABELS[row.dataset]}" for row in df.itertuples()]
x = np.arange(len(group_labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(13, 8))
ax.bar(x - bar_width / 2, df["native_win_rate"], width=bar_width, color=NATIVE_COLOR, label="Native SPM")
ax.bar(x + bar_width / 2, df["fast_relax_riam_win_rate"], width=bar_width, color=FAST_RELAX_COLOR,
       label="Fast Relax RIAM")

for xi, native_rate, fr_rate in zip(x, df["native_win_rate"], df["fast_relax_riam_win_rate"]):
    ax.text(xi - bar_width / 2, native_rate + 0.02, f"{native_rate:.0%}", ha="center", va="bottom", fontsize=14)
    ax.text(xi + bar_width / 2, fr_rate + 0.02, f"{fr_rate:.0%}", ha="center", va="bottom", fontsize=14)

ax.axhline(1 / 6, linestyle="--", color="gray", linewidth=1.5, label="Random (1/6)")
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Best-Metric Win Rate")
ax.set_title("Best-Metric Win Rate: Fast Relax RIAM vs. Native SPM")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fast_relax_vs_native_winrate.png", dpi=300)
plt.close()

print(df[["spm", "dataset", "native_win_rate", "fast_relax_riam_win_rate", "fast_relax_riam_vs_native_delta"]]
      .to_string(index=False))
