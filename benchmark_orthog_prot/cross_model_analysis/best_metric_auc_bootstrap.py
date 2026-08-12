"""Bootstrap significance test comparing each model's best-metric AUC (the numbers
tabulated in best_metric_all_datasets.csv) against every other model's, per dataset.

best_metric_all_datasets.csv only holds point-estimate AUCs -- one number per model per
dataset, nothing to test significance against. This script goes back to the raw
per-sample metric values those AUCs were computed from and rebuilds a distribution via
stratified bootstrap resampling, the same method each model's own
scripts/analysis/auc_stats.py already uses (sample sizes here are too small -- dhd has
only 6 cognate rows -- for asymptotic tests like DeLong's to be reliable).

The one wrinkle auc_stats.py doesn't have to handle: it only ever compares metrics
within one model's own dataframe, where row position i always means the same
comparison. Here the four models' raw CSVs cover the same 78 (dhd) / 56 (cop)
comparisons but in different row orders (verified in best_margin_wilcoxon.py's
docstring), so rows are instead keyed by a canonical, order-independent
(protein_a, protein_b) identity and resampled by that key -- the same fix
best_margin_wilcoxon.py made for the same reason. Resampling the identical set of keys
for all four models on a given draw is what lets shared-row correlation cancel out of
the a-vs-b AUC diff, exactly as auc_stats.py's top-metric-vs-others comparison relies on
reusing one `idx` for both metrics each draw.

Runs all C(4,2) = 6 pairwise model comparisons per dataset: a two-sided bootstrap
p-value (percentile method: 2 * min(P(diff<=0), P(diff>=0))), Bonferroni-corrected by
the family of 6 -- same convention as best_margin_wilcoxon.py, whose heatmap style this
also reuses.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ALPHA = 0.05
N_RESAMPLES = 2000
rng = np.random.default_rng(42)

SIG_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIAGONAL_COLOR = "#e1e0d9"

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
N_COMPARISONS_PER_DATASET = len(list(combinations(MODEL_ORDER, 2)))  # 6
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

# same PATHS/LOWER_IS_BETTER dicts as best_margin_wilcoxon.py (copied, not imported, to
# keep this script standalone)
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


def canonical_key(id_value, sep):
    return tuple(sorted(p.lower() for p in id_value.split(sep)))


def load_keyed(model, dataset, metric):
    """Raw rows for one model, indexed by canonical (protein_a, protein_b) identity so
    they can be aligned against the other three models' rows regardless of file order."""
    path, id_col, sep = PATHS[model][dataset]
    df = pd.read_csv(path)
    df["key"] = df[id_col].apply(lambda v: canonical_key(v, sep))
    score = -df[metric] if metric in LOWER_IS_BETTER[model] else df[metric]
    return pd.DataFrame({"score": score.to_numpy(), "label": df["cognate_interaction"].to_numpy()},
                         index=df["key"]).sort_index()


def auc_score(labels, scores):
    """AUC via the Mann-Whitney rank-sum formula, same as auc_stats.py."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    ranks = rankdata(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def stratified_resample_idx(pos_idx, neg_idx):
    resampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    resampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([resampled_pos, resampled_neg])


best_metric_table = pd.read_csv(RESULTS_DIR / "best_metric_all_datasets.csv")

for dataset in ["dhd", "cop"]:
    results_dir = RESULTS_DIR / dataset
    dataset_label = DATASET_LABELS[dataset]
    row = best_metric_table[best_metric_table["dataset"] == dataset_label]
    best_metric = dict(zip(row["model"], row["best_metric"]))

    keyed = {m: load_keyed(m, dataset, best_metric[m]) for m in MODEL_ORDER}

    # sanity check: all four models must cover the exact same set of comparisons, with
    # the same cognate/non-cognate label per comparison, or the resampling below would
    # silently compare mismatched rows
    shared_keys = sorted(set.intersection(*(set(df.index) for df in keyed.values())))
    assert all(len(keyed[m]) == len(shared_keys) for m in MODEL_ORDER), \
        f"{dataset}: models don't all cover the same {len(shared_keys)} comparisons"
    labels_arr = keyed[MODEL_ORDER[0]].loc[shared_keys, "label"].to_numpy()
    for m in MODEL_ORDER[1:]:
        assert np.array_equal(keyed[m].loc[shared_keys, "label"].to_numpy(), labels_arr), \
            f"{dataset}: {m} disagrees with {MODEL_ORDER[0]} on which rows are cognate"

    scores_arr = {m: keyed[m].loc[shared_keys, "score"].to_numpy() for m in MODEL_ORDER}
    pos_idx = np.flatnonzero(labels_arr == 1)
    neg_idx = np.flatnonzero(labels_arr == 0)

    # --- stratified bootstrap: the SAME resampled comparisons score every model each
    # draw, so shared-row correlation affects all four equally and mostly cancels out
    # of the pairwise diffs below ---
    boot_auc = {m: np.empty(N_RESAMPLES) for m in MODEL_ORDER}
    for i in range(N_RESAMPLES):
        idx = stratified_resample_idx(pos_idx, neg_idx)
        resampled_labels = labels_arr[idx]
        for m in MODEL_ORDER:
            boot_auc[m][i] = auc_score(resampled_labels, scores_arr[m][idx])

    rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        diffs = boot_auc[model_a] - boot_auc[model_b]
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        p_value = float(min(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)), 1.0))
        p_bonferroni = float(min(p_value * N_COMPARISONS_PER_DATASET, 1.0))
        rows.append({
            "dataset": dataset_label,
            "model_a": model_a, "metric_a": best_metric[model_a], "auc_a": float(boot_auc[model_a].mean()),
            "model_b": model_b, "metric_b": best_metric[model_b], "auc_b": float(boot_auc[model_b].mean()),
            "n_comparisons": len(shared_keys),
            "auc_diff_a_minus_b": float(diffs.mean()),
            "diff_ci_low": ci_low, "diff_ci_high": ci_high,
            "p_value": p_value,
            "p_value_bonferroni": p_bonferroni,
            "significant": bool(p_bonferroni < ALPHA),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_dir / "best_metric_auc_bootstrap.csv", index=False)

    print(f"--- {dataset_label} ---")
    print(results_df.to_string(index=False))
    print()

    # --- heatmap: -log10(p, Bonferroni-corrected) per model pair, symmetric, diagonal masked ---
    n = len(MODEL_ORDER)
    neg_log_p = np.full((n, n), np.nan)
    raw_p = np.full((n, n), np.nan)
    for r in rows:
        i, j = MODEL_ORDER.index(r["model_a"]), MODEL_ORDER.index(r["model_b"])
        val = -np.log10(max(r["p_value_bonferroni"], 1e-300))
        neg_log_p[i, j] = neg_log_p[j, i] = val
        raw_p[i, j] = raw_p[j, i] = r["p_value_bonferroni"]

    vmax = max(np.nanmax(neg_log_p), -np.log10(ALPHA))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
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
                     color=text_color, fontsize=8)
            if raw_p[i, j] < ALPHA:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="black", linewidth=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title(f"Bootstrap AUC comparison, best metric per model — {dataset_label} (N={len(shared_keys)})\n"
                 f"(bordered = significant at Bonferroni-corrected α={ALPHA})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(p, Bonferroni-corrected)")
    plt.tight_layout()
    plt.savefig(results_dir / "best_metric_auc_bootstrap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
