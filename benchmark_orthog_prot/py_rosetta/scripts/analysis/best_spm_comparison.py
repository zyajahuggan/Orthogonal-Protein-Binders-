"""Within riam, and separately within fast_relax_riam, compares the four structure-
prediction models (af3, boltz, chai, esmfold2) against each other: for each spm, picks
its single best-differentiating Rosetta metric (highest median all-pairs standardized
differential -- same selection rule cross_model_analysis/best_margin_comparison.py
uses), then tests whether the four spms' best-metric margins actually differ.

This answers "which spm's structures does riam (or fast_relax_riam) score best?" --
the physics-based analogue of cross_model_analysis/best_margin_comparison.py and
best_margin_wilcoxon.py, which do the same thing across DL models' own confidence
metrics. Reuses pairing_utils.py (orient_scores, LOWER_IS_BETTER_COLUMNS,
UNWANTED_COLUMNS, METRICS_DIR) since this lives in the same directory as the rest of
the rosetta analysis scripts; duplicates the keyed-pairing and Wilcoxon-family logic
from cross_model_analysis/best_margin_wilcoxon.py (copied, not imported, matching that
script's own precedent of duplicating rather than cross-importing between projects).

Outputs land in results/best_spm_comparison/<project>/<dataset>/ -- a new top-level
results folder, alongside the existing win_rate_comparison/rmsd_comparison ones that
already hold cross-cutting (not per-spm) comparisons.

Two passes per project/dataset, same reasoning as best_margin_wilcoxon.py:
  1. per-comparison -- one point per (cognate pair, competitor).
  2. cognate-pair collapsed -- competitor diffs sharing a cognate pair are medianed
     down to one point per cognate pair first, removing the within-cognate-pair
     correlation the per-comparison pass leaves in (dhd has only 6 cognate pairs,
     cop has 28).
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon
from pairing_utils import orient_scores, UNWANTED_COLUMNS, METRICS_DIR, BENCHMARK_DIR

ALPHA = 0.05
SIG_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIAGONAL_COLOR = "#e1e0d9"
SPM_COLORS = {"af3": "#2a78d6", "boltz": "#1baf7a", "chai": "#eda100", "esmfold2": "#008300"}

RESULTS_DIR = BENCHMARK_DIR / "results" / "best_spm_comparison"
SPM_ORDER = ["af3", "boltz", "chai", "esmfold2"]
PROJECTS = ["riam", "fast_relax_riam"]
DATASETS = ["dhd", "cop"]
N_COMPARISONS = len(list(combinations(SPM_ORDER, 2)))  # 6
rng = np.random.default_rng(42)


def find_non_cognate_pairs(df, sep="_vs_"):
    """Same matching rule as pairing_utils.find_non_cognate_indices, but returns each
    match's canonical (cognate_key, competitor_key) identity too -- an order-independent
    tuple of the two protein names -- so diffs can be matched across spms regardless of
    row order (all four spms' CSVs were verified to cover the exact same protein_pair
    keys, per-dataset, before writing this)."""
    cognate_indices = df.index[df["cognate_status"] == 1]
    matches = []
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, "protein_pair"].split(sep)
        cognate_key = tuple(sorted(p.lower() for p in cognate_pair))
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, "protein_pair"].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    competitor_key = tuple(sorted(p.lower() for p in sample_pair))
                    matches.append((cog_idx, i, cognate_key, competitor_key))
    return matches


def best_metric_diffs(project, spm, dataset):
    """Recomputes spm's best Rosetta metric for this project (highest median all-pairs
    differential), keyed by (cognate_key, competitor_key) identity."""
    path = METRICS_DIR / f"{project}_{spm}_{dataset}_combined.csv"
    df = pd.read_csv(path)
    matches = find_non_cognate_pairs(df)

    diffs_by_metric = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED_COLUMNS:
            continue
        oriented = orient_scores(df, metric)
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


def collapse_by_cognate(keyed_diffs):
    by_cognate = {}
    for (cognate_key, _competitor_key), diff in keyed_diffs.items():
        by_cognate.setdefault(cognate_key, []).append(diff)
    return {cognate_key: float(np.median(vals)) for cognate_key, vals in by_cognate.items()}


def plot_comparison_boxplot(diffs_by_spm, best_metric, project, dataset, out_dir):
    labels = [f"{spm}\n({best_metric[spm]})" for spm in SPM_ORDER]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([list(diffs_by_spm[spm].values()) for spm in SPM_ORDER], tick_labels=labels, showfliers=False)
    for i, spm in enumerate(SPM_ORDER, start=1):
        y = list(diffs_by_spm[spm].values())
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, alpha=0.4, color=SPM_COLORS[spm], s=12, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Standardized differential (SD units)")
    ax.set_title(f"Best-margin metric per spm, all-pairs differential — {project} ({dataset})")
    plt.tight_layout()
    plt.savefig(out_dir / "best_spm_comparison_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary_df = pd.DataFrame([
        {
            "spm": spm, "best_metric": best_metric[spm],
            "median_differential": float(np.median(list(diffs_by_spm[spm].values()))),
            "total_differential": float(np.sum(list(diffs_by_spm[spm].values()))),
            "n_comparisons": len(diffs_by_spm[spm]),
        }
        for spm in SPM_ORDER
    ])
    summary_df.to_csv(out_dir / "best_spm_comparison_summary.csv", index=False)
    return summary_df


def run_wilcoxon_family(diffs_by_spm, best_metric, project, dataset, out_dir, suffix, title_note):
    rows = []
    for spm_a, spm_b in combinations(SPM_ORDER, 2):
        shared_keys = sorted(set(diffs_by_spm[spm_a]) & set(diffs_by_spm[spm_b]))
        a_vals = np.array([diffs_by_spm[spm_a][k] for k in shared_keys])
        b_vals = np.array([diffs_by_spm[spm_b][k] for k in shared_keys])

        stat, p = wilcoxon(a_vals, b_vals)
        rows.append({
            "project": project, "dataset": dataset,
            "spm_a": spm_a, "metric_a": best_metric[spm_a],
            "spm_b": spm_b, "metric_b": best_metric[spm_b],
            "n_matched_pairs": len(shared_keys),
            "median_diff_a_minus_b": float(np.median(a_vals - b_vals)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "p_value_bonferroni": float(min(p * N_COMPARISONS, 1.0)),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / f"best_spm_wilcoxon{suffix}.csv", index=False)
    print(f"--- {project} {dataset}{title_note} ---")
    print(results_df.to_string(index=False))
    print()

    n = len(SPM_ORDER)
    neg_log_p = np.full((n, n), np.nan)
    raw_p = np.full((n, n), np.nan)
    for row in rows:
        i, j = SPM_ORDER.index(row["spm_a"]), SPM_ORDER.index(row["spm_b"])
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
            ax.text(j, i, f"p={raw_p[i, j]:.2g}", ha="center", va="center", color=text_color, fontsize=8)
            if raw_p[i, j] < ALPHA:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))

    n_pairs = rows[0]["n_matched_pairs"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SPM_ORDER)
    ax.set_yticklabels(SPM_ORDER)
    ax.set_title(f"Wilcoxon signed-rank, best spm metric — {project} ({dataset}){title_note} (N={n_pairs})\n"
                 f"(bordered = significant at Bonferroni-corrected α={ALPHA})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(p, Bonferroni-corrected)")
    plt.tight_layout()
    plt.savefig(out_dir / f"best_spm_wilcoxon{suffix}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    return results_df


for project in PROJECTS:
    for dataset in DATASETS:
        out_dir = RESULTS_DIR / project / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        best_metric, diffs = {}, {}
        for spm in SPM_ORDER:
            best_metric[spm], diffs[spm] = best_metric_diffs(project, spm, dataset)

        plot_comparison_boxplot(diffs, best_metric, project, dataset, out_dir)
        run_wilcoxon_family(diffs, best_metric, project, dataset, out_dir,
                             suffix="", title_note=" (per-comparison)")

        collapsed = {spm: collapse_by_cognate(diffs[spm]) for spm in SPM_ORDER}
        run_wilcoxon_family(collapsed, best_metric, project, dataset, out_dir,
                             suffix="_collapsed", title_note=" (cognate-pair collapsed)")
