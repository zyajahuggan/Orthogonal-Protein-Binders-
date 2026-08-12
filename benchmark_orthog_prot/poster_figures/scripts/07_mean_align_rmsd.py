"""Poster version of py_rosetta/scripts/analysis/rmsd_comparison.py: mean align RMSD
per model, split by cognate vs. non-cognate, faceted by dataset. Reads the same
fast_relax_riam align-RMSD CSVs directly -- no recomputation.

New file -- does not modify or overwrite rmsd_comparison.py or its outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RMSD_DIR = Path(__file__).resolve().parent.parent.parent / "py_rosetta" / "metrics" / "rmsd"
OUT_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 23,
    "axes.titleweight": "bold",
    "figure.titlesize": 27,
    "axes.labelsize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

MODEL_ORDER = ["af3", "boltz", "chai", "esmfold2"]
MODEL_LABELS = {"af3": "AF3", "boltz": "Boltz-2", "chai": "Chai", "esmfold2": "ESMFold2"}

DATASETS = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

COGNATE_COLOR = "#2a78d6"
NON_COGNATE_COLOR = "#e34948"

summary_rows = []
for dataset in DATASETS:
    csv_path = RMSD_DIR / f"fast_relax_riam_{dataset}_align_rmsd.csv"
    df = pd.read_csv(csv_path)
    for model in MODEL_ORDER:
        for cognate_status, status_label in [(1, "cognate"), (0, "non_cognate")]:
            vals = df.loc[(df["model"] == model) & (df["cognate_status"] == cognate_status), "rmsd"]
            summary_rows.append({
                "dataset": dataset, "model": model, "cognate_status": status_label,
                "mean_rmsd": vals.mean(), "std_rmsd": vals.std(), "n": len(vals),
            })
summary = pd.DataFrame(summary_rows)

x = np.arange(len(MODEL_ORDER))
width = 0.35
fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 8), sharey=True)
for ax, dataset in zip(axes, DATASETS):
    sub = summary[summary["dataset"] == dataset]
    for offset, (status_label, color) in zip(
        [-width / 2, width / 2],
        [("cognate", COGNATE_COLOR), ("non_cognate", NON_COGNATE_COLOR)],
    ):
        rows = sub[sub["cognate_status"] == status_label].set_index("model").loc[MODEL_ORDER]
        ax.bar(x + offset, rows["mean_rmsd"], width, yerr=rows["std_rmsd"], capsize=4,
               label=status_label.replace("_", "-").title(), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title(DATASET_LABELS[dataset])

axes[0].set_ylabel("Mean Align RMSD (Å)")
axes[0].legend()
fig.suptitle("Mean Align RMSD by Model, Dataset, and Cognate Status", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "mean_align_rmsd_by_model_dataset_cognate.png", dpi=300)
plt.close()

print(summary.to_string(index=False))
