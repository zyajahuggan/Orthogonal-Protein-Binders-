"""Paired Wilcoxon signed-rank test between every pair of models' best-margin metric
(the same "best metric" selection best_margin_comparison.py makes), for both datasets.

Wilcoxon requires that the two samples be paired -- the same underlying unit measured
twice. Here that unit is a single (cognate pair, non-cognate competitor) comparison.
best_margin_comparison.py's own diff lists are ordered by row position in each model's
own CSV, which is NOT guaranteed to line up across models. This script instead tags
every diff with a canonical (cognate_key, competitor_key) identity -- a case-insensitive,
order-independent tuple of the two protein names -- and only pairs diffs across models
that share that identity. (Verified by hand first: after lowercasing, all four models'
dhd files cover the exact same 78 row-keys, and all four cop files cover the
same 56 -- see conversation. Case differs only in the "DHD"/"dhd" prefix, e.g. AF3 writes
"dhd150a_vs_dhd150b" while Boltz-2 writes "DHD150a_vs_DHD150b".)

Same PATHS/LOWER_IS_BETTER/UNWANTED dicts and diff-computation logic as
best_margin_comparison.py (copied, not imported, to keep this script standalone).

Runs all C(4,2) = 6 model-pair comparisons per dataset. Since that's 6 tests drawn from
the same data (a multiple-comparisons problem), a Bonferroni-corrected p-value
(raw p * 6, capped at 1) is reported alongside the raw one for each dataset's family of
6 tests.

Runs that family TWICE per dataset:
  1. per-comparison -- one point per (cognate pair, competitor), same as above.
  2. cognate-pair collapsed -- the ~20 competitor diffs sharing a cognate pair are
     medianed down to a single point per cognate pair first. The per-comparison points
     aren't independent (they share the same cognate score within a cognate pair), which
     Wilcoxon assumes they are; collapsing removes that correlation at the cost of a much
     smaller, less powerful N (6 cognate pairs for dhd, 28 for cop, vs. 132/56).
     If a result holds up in both passes, it's on much firmer ground.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ALPHA = 0.05
# sequential blue ramp, light->dark (steps 100-700 from the project's palette)
SIG_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIAGONAL_COLOR = "#e1e0d9"

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
N_COMPARISONS_PER_DATASET = len(list(combinations(MODEL_ORDER, 2)))  # 6

# copied verbatim from best_margin_comparison.py
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


def find_non_cognate_pairs(df, id_col, sep):
    """Same matching rule as best_margin_comparison.py's find_non_cognate_indices, but
    returns each match's canonical (cognate_key, competitor_key) identity too -- a
    lowercased, order-independent tuple of the two protein names -- so diffs can be
    matched across models regardless of ID separator, capitalization, or row order."""
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
    """Recomputes the model's best-margin metric (highest median all-pairs
    differential -- same selection rule as best_margin_comparison.py), keyed by
    (cognate_key, competitor_key) identity instead of row position."""
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
            continue  # zero-variance metric, same skip rule as best_margin_comparison.py
        standardized = (oriented - oriented.mean()) / std
        keyed = {
            (cognate_key, competitor_key): standardized.loc[cog_idx] - standardized.loc[noncog_idx]
            for cog_idx, noncog_idx, cognate_key, competitor_key in matches
        }
        diffs_by_metric[metric] = keyed

    best_metric = max(diffs_by_metric, key=lambda m: np.median(list(diffs_by_metric[m].values())))
    return best_metric, diffs_by_metric[best_metric]


def collapse_by_cognate(keyed_diffs):
    """Medians the diffs sharing a cognate_key down to one point per cognate pair --
    removes the within-cognate-pair correlation (they all share the same cognate score)
    that the per-comparison version leaves in, at the cost of a much smaller N."""
    by_cognate = {}
    for (cognate_key, _competitor_key), diff in keyed_diffs.items():
        by_cognate.setdefault(cognate_key, []).append(diff)
    return {cognate_key: float(np.median(vals)) for cognate_key, vals in by_cognate.items()}


def run_wilcoxon_family(diffs_by_model, best_metric, dataset, results_dir, suffix, title_note):
    """Runs all C(4,2) pairwise Wilcoxon tests over diffs_by_model (model -> {key: diff}),
    saves the results table and a p-value heatmap. Shared by both the per-comparison pass
    and the cognate-pair-collapsed pass -- only the diffs passed in differ."""
    rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        shared_keys = sorted(set(diffs_by_model[model_a]) & set(diffs_by_model[model_b]))
        a_vals = np.array([diffs_by_model[model_a][k] for k in shared_keys])
        b_vals = np.array([diffs_by_model[model_b][k] for k in shared_keys])

        stat, p = wilcoxon(a_vals, b_vals)
        rows.append({
            "dataset": dataset,
            "model_a": model_a, "metric_a": best_metric[model_a],
            "model_b": model_b, "metric_b": best_metric[model_b],
            "n_matched_pairs": len(shared_keys),
            "n_a_total": len(diffs_by_model[model_a]), "n_b_total": len(diffs_by_model[model_b]),
            "median_diff_a_minus_b": float(np.median(a_vals - b_vals)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "p_value_bonferroni": float(min(p * N_COMPARISONS_PER_DATASET, 1.0)),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_dir / f"best_margin_wilcoxon{suffix}.csv", index=False)

    print(f"--- {dataset}{title_note} ---")
    print(results_df.to_string(index=False))
    print()

    # --- heatmap: -log10(p, Bonferroni-corrected) per model pair, symmetric, diagonal masked ---
    n = len(MODEL_ORDER)
    neg_log_p = np.full((n, n), np.nan)
    raw_p = np.full((n, n), np.nan)
    for row in rows:
        i, j = MODEL_ORDER.index(row["model_a"]), MODEL_ORDER.index(row["model_b"])
        val = -np.log10(max(row["p_value_bonferroni"], 1e-300))
        neg_log_p[i, j] = neg_log_p[j, i] = val
        raw_p[i, j] = raw_p[j, i] = row["p_value_bonferroni"]

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

    n_pairs = rows[0]["n_matched_pairs"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title(f"Wilcoxon signed-rank, best-margin metric — {dataset}{title_note} (N={n_pairs})\n"
                 f"(bordered = significant at Bonferroni-corrected α={ALPHA})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(p, Bonferroni-corrected)")
    plt.tight_layout()
    plt.savefig(results_dir / f"best_margin_wilcoxon{suffix}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    return results_df


for dataset in ["cop", "dhd"]:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    best_metric, diffs = {}, {}
    for model in MODEL_ORDER:
        best_metric[model], diffs[model] = best_metric_diffs(model, dataset)

    run_wilcoxon_family(diffs, best_metric, dataset, results_dir,
                         suffix="", title_note=" (per-comparison)")

    collapsed_diffs = {model: collapse_by_cognate(diffs[model]) for model in MODEL_ORDER}
    run_wilcoxon_family(collapsed_diffs, best_metric, dataset, results_dir,
                         suffix="_collapsed", title_note=" (cognate-pair collapsed)")
