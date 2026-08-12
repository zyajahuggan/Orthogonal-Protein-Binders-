"""Poster version of cross_model_analysis/best_margin_wilcoxon.py: the statistical
companion to 03_best_margin_metric_per_model.py -- paired Wilcoxon signed-rank test
between every pair of models' best-margin metric, Bonferroni-corrected across the 6
model-pair comparisons per dataset. Uses the "per-comparison" pass (one point per
cognate-pair/non-cognate-competitor match); best_margin_wilcoxon.py's own "cognate-pair
collapsed" pass is a robustness check on a much smaller N and isn't reproduced here.

Combined into one figure (DHD + COP side by side) instead of best_margin_wilcoxon.py's
two separate per-dataset heatmaps, matching 03's combined layout, with poster-sized
fonts.

New file -- does not modify or overwrite best_margin_wilcoxon.py or its outputs.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.titleweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

ALPHA = 0.05
SIG_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIAGONAL_COLOR = "#e1e0d9"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
DATASET_ORDER = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}
N_COMPARISONS_PER_DATASET = len(list(combinations(MODEL_ORDER, 2)))  # 6

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
LOWER_IS_BETTER = {
    "AF3": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0", "fraction_disordered"],
    "Boltz-2": ["complex_pde", "complex_ipde"],
    "Chai": [],
    "ESMFold2": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
}
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


def find_non_cognate_pairs(df, id_col, sep):
    cognate_indices = df.index[df["cognate_interaction"] == 1]
    matches = []
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, id_col].split(sep)
        cognate_key = tuple(sorted(p.lower() for p in cognate_pair))
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, id_col].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    competitor_key = tuple(sorted(p.lower() for p in sample_pair))
                    matches.append((cog_idx, i, cognate_key, competitor_key))
    return matches


def orient_scores(df, metric, lower_is_better_columns):
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def best_metric_diffs(model, dataset):
    path, id_col, sep = PATHS[model][dataset]
    df = pd.read_csv(path)
    matches = find_non_cognate_pairs(df, id_col, sep)

    diffs_by_metric = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED[model]:
            continue
        oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
        std = oriented.std()
        if std == 0:
            continue
        standardized = (oriented - oriented.mean()) / std
        keyed = {
            (cognate_key, competitor_key): standardized.loc[cog_idx] - standardized.loc[noncog_idx]
            for cog_idx, noncog_idx, cognate_key, competitor_key in matches
        }
        diffs_by_metric[metric] = keyed

    best_metric = max(diffs_by_metric, key=lambda m: np.median(list(diffs_by_metric[m].values())))
    return best_metric, diffs_by_metric[best_metric]


def wilcoxon_family(diffs_by_model):
    rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        shared_keys = sorted(set(diffs_by_model[model_a]) & set(diffs_by_model[model_b]))
        a_vals = np.array([diffs_by_model[model_a][k] for k in shared_keys])
        b_vals = np.array([diffs_by_model[model_b][k] for k in shared_keys])
        stat, p = wilcoxon(a_vals, b_vals)
        rows.append({
            "model_a": model_a, "model_b": model_b,
            "p_value_bonferroni": float(min(p * N_COMPARISONS_PER_DATASET, 1.0)),
            "n_matched_pairs": len(shared_keys),
        })
    return rows


fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
n = len(MODEL_ORDER)

for ax, dataset in zip(axes, DATASET_ORDER):
    best_metric, diffs = {}, {}
    for model in MODEL_ORDER:
        best_metric[model], diffs[model] = best_metric_diffs(model, dataset)
    rows = wilcoxon_family(diffs)

    neg_log_p = np.full((n, n), np.nan)
    raw_p = np.full((n, n), np.nan)
    for row in rows:
        i, j = MODEL_ORDER.index(row["model_a"]), MODEL_ORDER.index(row["model_b"])
        val = -np.log10(max(row["p_value_bonferroni"], 1e-300))
        neg_log_p[i, j] = neg_log_p[j, i] = val
        raw_p[i, j] = raw_p[j, i] = row["p_value_bonferroni"]

    vmax = max(np.nanmax(neg_log_p), -np.log10(ALPHA))
    masked = np.ma.masked_invalid(neg_log_p)
    cmap = SIG_CMAP.copy()
    cmap.set_bad(DIAGONAL_COLOR)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            frac = neg_log_p[i, j] / vmax
            text_color = "white" if frac > 0.55 else "#0b0b0b"
            ax.text(j, i, f"p={raw_p[i, j]:.2g}", ha="center", va="center",
                    color=text_color, fontsize=15)
            if raw_p[i, j] < ALPHA:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="black", linewidth=2.5))

    n_pairs = rows[0]["n_matched_pairs"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title(f"{DATASET_LABELS[dataset]} (N={n_pairs})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10(p, Bonferroni)", fontsize=15)
    cbar.ax.tick_params(labelsize=14)

fig.suptitle("Statistical Validation of Best-Margin Metric per Model\n"
             "(Bordered = Significant at Bonferroni-Corrected α=0.05)", fontsize=23, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT_DIR / "best_margin_statistical_validation.png", dpi=300, bbox_inches="tight")
plt.close()

print("saved best_margin_statistical_validation.png")
