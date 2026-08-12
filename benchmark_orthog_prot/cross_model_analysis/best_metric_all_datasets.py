"""Combines the per-dataset best_metric_comparison.csv files (produced by
best_metric_comparison.py) into a single table and a single grouped-bar chart,
so all 4 DL models' best metric can be compared side by side across both
datasets at once.

Reads only the already-computed CSVs -- no recomputation, nothing here
touches best_metric_comparison.py or its per-dataset outputs.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}
DATASET_ORDER = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

# combine the two per-dataset CSVs into one long table
frames = []
for dataset in DATASET_ORDER:
    df = pd.read_csv(RESULTS_DIR / dataset / "best_metric_comparison.csv")
    df.insert(0, "dataset", DATASET_LABELS[dataset])
    frames.append(df)
combined = pd.concat(frames, ignore_index=True)
combined.to_csv(RESULTS_DIR / "best_metric_all_datasets.csv", index=False)

# grouped bar chart: one group per dataset, one bar per model within the group
fig, ax = plt.subplots(figsize=(9, 6))
n_models = len(MODEL_ORDER)
bar_width = 0.8 / n_models
group_positions = np.arange(len(DATASET_ORDER))

for i, model in enumerate(MODEL_ORDER):
    offset = (i - (n_models - 1) / 2) * bar_width
    heights, labels = [], []
    for dataset in DATASET_ORDER:
        row = combined[(combined["dataset"] == DATASET_LABELS[dataset]) & (combined["model"] == model)].iloc[0]
        heights.append(row["auc"])
        labels.append(f"{row['best_metric']}\n{row['auc']:.3f}")
    bars = ax.bar(group_positions + offset, heights, width=bar_width * 0.9,
                   color=MODEL_COLORS[model], label=model)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                label, ha="center", va="bottom", fontsize=6.5)

ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
ax.set_ylabel("AUC")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_title("Best metric per model, by dataset")
ax.legend(title="Model", loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "best_metric_all_datasets.png", dpi=300)
plt.close()

print(combined.to_string(index=False))
