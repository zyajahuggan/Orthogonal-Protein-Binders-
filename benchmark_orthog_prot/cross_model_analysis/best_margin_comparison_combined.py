"""Same computation as best_margin_comparison.py (best metric per model, ranked by median
all-pairs standardized differential), but plotted as a single grouped-boxplot Axes --
one group per dataset, one box per model within the group, same layout convention as
best_winrate_all_datasets.py -- instead of two side-by-side subplots. New file, does not
modify or overwrite best_margin_comparison.py or its outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(42)

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}
DATASET_ORDER = ["dhd", "cop"]  # left-to-right group order
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

# (metrics path, ID column, separator) per project per dataset -- copied from
# best_margin_comparison.py
PATHS = {
    "AF3": {
        "cop": (BASE / "af3/metrics/cop_metrics_af3.csv", "sample", "_vs_"),
        "dhd": (BASE / "af3/metrics/dhd_metrics_af3_filtered.csv", "sample", "_vs_"),
    },
    "Boltz-2": {
        "cop": (BASE / "boltz/metrics/boltz_metrics_cop.csv", "job_name", "_vs_"),
        "dhd": (BASE / "boltz/metrics/boltz_metrics_dhd_filtered.csv", "job_name", "_vs_"),
    },
    "Chai": {
        "cop": (BASE / "chai/metrics/cop_metrics.csv", "project", "_vs_"),
        "dhd": (BASE / "chai/metrics/dhd_metrics_filtered.csv", "project", "_vs_"),
    },
    "ESMFold2": {
        "cop": (BASE / "esmfold2/metrics/cop_combined_metrics.csv", "job_name", "_vs_"),
        "dhd": (BASE / "esmfold2/metrics/dhd_combined_metrics_filtered.csv", "job_name", "__"),
    },
}

# copied from each project's own pairing_utils.py
LOWER_IS_BETTER = {
    "AF3": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0", "fraction_disordered"],
    "Boltz-2": ["complex_pde", "complex_ipde"],
    "Chai": [],
    "ESMFold2": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
}
# raw bookkeeping counts/normalization constants add_ipsae_metrics.py adds alongside the
# real ipSAE scores -- not confidence signals themselves, so excluded from the metric search
IPSAE_BOOKKEEPING = ["ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom", "ipsae_d0res", "ipsae_d0chn",
                      "ipsae_d0dom", "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2"]
UNWANTED = {
    "AF3": ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash",
            "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "Boltz-2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "Chai": ["project", "cognate_interaction", "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
             "chain_chain_clashes_1_0", "chain_chain_clashes_1_1", "has_inter_chain_clashes", "interface_hbonds"],
    "ESMFold2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
}


def find_non_cognate_indices(df, id_col, sep):
    """For every cognate row, find the row indices of every non-cognate row that shares
    one of its two proteins."""
    cognate_indices = df.index[df["cognate_interaction"] == 1]
    non_cognate_indices = {}
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, id_col].split(sep)
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, id_col].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    non_cognate_indices.setdefault(cog_idx, []).append(i)
    return non_cognate_indices


def orient_scores(df, metric, lower_is_better_columns):
    """Flip sign for error-like metrics so higher always means 'stronger cognate signal'."""
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def best_margin_for_dataset(dataset):
    best_metric_by_model = {}
    differentials_by_model = {}
    summary_rows = []

    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)
        non_cognate_indices = find_non_cognate_indices(df, id_col, sep)

        differentials_by_metric = {}
        for metric in df.columns:
            if metric.strip() in UNWANTED[model]:
                continue
            oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
            std = oriented.std()
            if std == 0:
                continue
            standardized = (oriented - oriented.mean()) / std
            diffs = []
            for cog_idx, non_cog_idxs in non_cognate_indices.items():
                cog_score = standardized.loc[cog_idx]
                diffs.extend(cog_score - standardized.loc[i] for i in non_cog_idxs)
            differentials_by_metric[metric] = diffs

        best_metric = max(differentials_by_metric, key=lambda m: np.median(differentials_by_metric[m]))
        best_metric_by_model[model] = best_metric
        differentials_by_model[model] = differentials_by_metric[best_metric]

        summary_rows.append({
            "dataset": dataset, "model": model, "best_metric": best_metric,
            "median_differential": float(np.median(differentials_by_metric[best_metric])),
            "total_differential": float(np.sum(differentials_by_metric[best_metric])),
            "n_comparisons": len(differentials_by_metric[best_metric]),
        })

    return best_metric_by_model, differentials_by_model, summary_rows


RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- compute best metric + differentials per model, per dataset ---
results = {}
all_summary_rows = []
for dataset in DATASET_ORDER:
    best_metric_by_model, differentials_by_model, summary_rows = best_margin_for_dataset(dataset)
    results[dataset] = (best_metric_by_model, differentials_by_model)
    all_summary_rows.extend(summary_rows)

# --- grouped boxplot: one group per dataset, one box per model within the group ---
fig, ax = plt.subplots(figsize=(9, 6))
n_models = len(MODEL_ORDER)
box_width = 0.8 / n_models
group_positions = np.arange(len(DATASET_ORDER))

legend_handles = [Patch(facecolor=MODEL_COLORS[model], label=model) for model in MODEL_ORDER]

for i, model in enumerate(MODEL_ORDER):
    offset = (i - (n_models - 1) / 2) * box_width
    for d_idx, dataset in enumerate(DATASET_ORDER):
        best_metric_by_model, differentials_by_model = results[dataset]
        data = differentials_by_model[model]
        pos = group_positions[d_idx] + offset

        bp = ax.boxplot([data], positions=[pos], widths=box_width * 0.9,
                         patch_artist=True, showfliers=False, zorder=2)
        for box in bp["boxes"]:
            box.set_facecolor("white")
        for median in bp["medians"]:
            median.set_color("black")

        x_scatter = rng.normal(pos, box_width * 0.12, size=len(data))
        ax.scatter(x_scatter, data, alpha=0.4, color=MODEL_COLORS[model], s=10, zorder=3)

        ax.text(pos, max(data) + 0.15, best_metric_by_model[model],
                ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_ylabel("Standardized differential (SD units)")
ax.margins(y=0.35)
ax.set_title("Best-margin metric per model, all-pairs differential")
ax.legend(handles=legend_handles, title="Model", loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "best_margin_comparison_combined.png", dpi=300, bbox_inches="tight")
plt.close()

summary_df = pd.DataFrame(all_summary_rows)
summary_df.to_csv(RESULTS_DIR / "best_margin_comparison_combined.csv", index=False)
print(summary_df.to_string(index=False))
