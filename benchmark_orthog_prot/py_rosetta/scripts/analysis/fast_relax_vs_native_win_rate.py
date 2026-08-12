"""Two-way version of native_vs_riam_vs_fast_relax_riam_win_rate.png -- drops the riam bars
so each spm/dataset group only compares fast_relax_riam against the native spm metrics that
gave it its input structures. Reads the already-computed win rates straight from
win_rate_delta.py's output CSV instead of recomputing them.

New file -- does not modify or overwrite win_rate_delta.py or its outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pairing_utils import BENCHMARK_DIR

NATIVE_COLOR = "#2a78d6"
FAST_RELAX_COLOR = "#4a3aa7"

comparison_dir = BENCHMARK_DIR / "results" / "win_rate_comparison"
df = pd.read_csv(comparison_dir / "riam_vs_fast_relax_riam_win_rate_delta.csv")

group_labels = [f"{row.spm}\n{row.dataset}" for row in df.itertuples()]
x = np.arange(len(group_labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_width / 2, df["native_win_rate"], width=bar_width, color=NATIVE_COLOR, label="native")
ax.bar(x + bar_width / 2, df["fast_relax_riam_win_rate"], width=bar_width, color=FAST_RELAX_COLOR, label="fast_relax_riam")

for xi, native_rate, fr_rate in zip(x, df["native_win_rate"], df["fast_relax_riam_win_rate"]):
    ax.text(xi - bar_width / 2, native_rate + 0.02, f"{native_rate:.0%}", ha="center", va="bottom", fontsize=8)
    ax.text(xi + bar_width / 2, fr_rate + 0.02, f"{fr_rate:.0%}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Best-metric win rate")
ax.set_title("Best-metric win rate: fast_relax_riam vs. the native spm that gave it its input structures")
ax.legend()
plt.tight_layout()
plt.savefig(comparison_dir / "fast_relax_riam_vs_native_win_rate.png", dpi=300)
plt.close()

print(df[["spm", "dataset", "native_win_rate", "fast_relax_riam_win_rate", "fast_relax_riam_vs_native_delta"]]
      .to_string(index=False))
