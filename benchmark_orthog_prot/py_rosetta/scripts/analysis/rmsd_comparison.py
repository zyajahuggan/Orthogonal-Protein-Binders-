"""Bar chart of mean align RMSD per model, split by cognate vs. non-cognate and
faceted by dataset (dhd vs. cop). Reads the two fast_relax_riam
align-RMSD CSVs directly -- no recomputation."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RMSD_DIR = Path(__file__).resolve().parent.parent.parent / "metrics" / "rmsd"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "rmsd_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# same fixed model order as cross_model_analysis/plot_model_comparison.py, so
# model identity stays visually consistent across the whole project
MODEL_ORDER = ["af3", "boltz", "chai", "esmfold2"]
MODEL_LABELS = {"af3": "AF3", "boltz": "Boltz-2", "chai": "Chai", "esmfold2": "ESMFold2"}

DATASETS = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

# same blue/red pair win_rate_delta.py uses for its two-state comparisons
COGNATE_COLOR = "#2a78d6"
NON_COGNATE_COLOR = "#e34948"

# --- gather mean +/- std RMSD per (dataset, model, cognate_status) ---
summary_rows = []
for dataset in DATASETS:
    csv_path = RMSD_DIR / f"fast_relax_riam_{dataset}_align_rmsd.csv"
    df = pd.read_csv(csv_path)
    for model in MODEL_ORDER:
        for cognate_status, status_label in [(1, "cognate"), (0, "non_cognate")]:
            vals = df.loc[(df["model"] == model) & (df["cognate_status"] == cognate_status), "rmsd"]
            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "cognate_status": status_label,
                "mean_rmsd": vals.mean(),
                "std_rmsd": vals.std(),
                "n": len(vals),
            })
summary = pd.DataFrame(summary_rows)
summary.to_csv(RESULTS_DIR / "mean_rmsd_by_model_dataset_cognate.csv", index=False)

# --- bar chart: one subplot per dataset, cognate/non-cognate paired per model ---
x = np.arange(len(MODEL_ORDER))
width = 0.35
fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 5), sharey=True)
for ax, dataset in zip(axes, DATASETS):
    sub = summary[summary["dataset"] == dataset]
    for offset, (status_label, color) in zip(
        [-width / 2, width / 2],
        [("cognate", COGNATE_COLOR), ("non_cognate", NON_COGNATE_COLOR)],
    ):
        rows = sub[sub["cognate_status"] == status_label].set_index("model").loc[MODEL_ORDER]
        ax.bar(x + offset, rows["mean_rmsd"], width, yerr=rows["std_rmsd"], capsize=3,
               label=status_label.replace("_", "-"), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title(DATASET_LABELS[dataset])

axes[0].set_ylabel("Mean align RMSD (Å)")
axes[0].legend(fontsize=8)
fig.suptitle("Mean Align RMSD by Model, Dataset, and Cognate Status")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mean_rmsd_by_model_dataset_cognate.png", dpi=300)
plt.close()

print(summary.to_string(index=False))
print(f"\nSaved plot to {RESULTS_DIR / 'mean_rmsd_by_model_dataset_cognate.png'}")
print(f"Saved summary table to {RESULTS_DIR / 'mean_rmsd_by_model_dataset_cognate.csv'}")
