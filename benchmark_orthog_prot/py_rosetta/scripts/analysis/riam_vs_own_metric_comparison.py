"""For each spm (af3, boltz, chai, esmfold2), compares three best-differentiating
margin metrics on that spm's own structures: riam's best Rosetta metric, fast_relax_riam's
best Rosetta metric, and the spm's own best confidence metric (e.g. AF3's iptm) -- the
same "own best metric" cross_model_analysis/best_margin_comparison.py picks. Answers:
does relaxing before scoring (fast_relax_riam) help Rosetta separate cognate from
non-cognate better than scoring the raw prediction (riam)? And does either physics-based
score beat the DL model's own confidence metric on its own structures?

Reuses pairing_utils.py's METRICS_DIR/BENCHMARK_DIR (same directory). The "own metric"
PATHS/LOWER_IS_BETTER/UNWANTED dicts are copied from cross_model_analysis/
best_margin_wilcoxon.py (lowercased spm keys to match this project's convention) --
duplicated rather than imported, matching that script's own precedent for crossing
between project directories. The keyed-pairing/collapse/Wilcoxon-family logic is
likewise duplicated from best_spm_comparison.py, generalized over id_col/sep/cognate_col
since riam/fast_relax_riam use "protein_pair"/"_vs_"/"cognate_status" while each spm's
own metrics file uses its own id column, separator, and "cognate_interaction".

Same two-pass structure as best_spm_comparison.py: per-comparison, then cognate-pair
collapsed (dhd: N=6 cognate pairs, cop: N=28) to correct for the
within-cognate-pair correlation the per-comparison pass leaves in. 3 pairwise tests per
family (riam vs fast_relax_riam, riam vs own, fast_relax_riam vs own) -> Bonferroni
factor of 3.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon
from pairing_utils import (
    orient_scores as orient_rosetta_scores,
    UNWANTED_COLUMNS as ROSETTA_UNWANTED,
    METRICS_DIR,
    BENCHMARK_DIR,
)

ALPHA = 0.05
SIG_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIAGONAL_COLOR = "#e1e0d9"
GROUP_ORDER = ["riam", "fast_relax_riam", "own"]
GROUP_COLORS = {"riam": "#2a78d6", "fast_relax_riam": "#eda100", "own": "#008300"}
N_COMPARISONS = len(list(combinations(GROUP_ORDER, 2)))  # 3

REPO_ROOT = BENCHMARK_DIR.parent  # pairing_utils.BENCHMARK_DIR is py_rosetta/ itself
RESULTS_DIR = BENCHMARK_DIR / "results" / "riam_vs_own_comparison"
SPM_ORDER = ["af3", "boltz", "chai", "esmfold2"]
DATASETS = ["dhd", "cop"]
rng = np.random.default_rng(42)

# copied from cross_model_analysis/best_margin_wilcoxon.py, lowercased spm keys
OWN_PATHS = {
    "af3": {
        "cop": (REPO_ROOT / "af3/metrics/cop_metrics_af3.csv", "sample", "_vs_"),
        "dhd": (REPO_ROOT / "af3/metrics/dhd_metrics_af3_filtered.csv", "sample", "_vs_"),
    },
    "boltz": {
        "cop": (REPO_ROOT / "boltz/metrics/boltz_metrics_cop.csv", "job_name", "_vs_"),
        "dhd": (REPO_ROOT / "boltz/metrics/boltz_metrics_dhd_filtered.csv", "job_name", "_vs_"),
    },
    "chai": {
        "cop": (REPO_ROOT / "chai/metrics/cop_metrics.csv", "project", "_vs_"),
        "dhd": (REPO_ROOT / "chai/metrics/dhd_metrics_filtered.csv", "project", "_vs_"),
    },
    "esmfold2": {
        "cop": (REPO_ROOT / "esmfold2/metrics/cop_combined_metrics.csv", "job_name", "_vs_"),
        "dhd": (REPO_ROOT / "esmfold2/metrics/dhd_combined_metrics_filtered.csv", "job_name", "__"),
    },
}
OWN_LOWER_IS_BETTER = {
    "af3": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0", "fraction_disordered"],
    "boltz": ["complex_pde", "complex_ipde"],
    "chai": [],
    "esmfold2": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
}
# raw bookkeeping counts/normalization constants add_ipsae_metrics.py adds alongside the
# real ipSAE scores -- not confidence signals themselves, so excluded from the metric search
IPSAE_BOOKKEEPING = ["ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom", "ipsae_d0res", "ipsae_d0chn",
                      "ipsae_d0dom", "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2"]
OWN_UNWANTED = {
    "af3": ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash",
            "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "boltz": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "chai": ["project", "cognate_interaction", "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
             "chain_chain_clashes_1_0", "chain_chain_clashes_1_1", "has_inter_chain_clashes", "interface_hbonds"],
    "esmfold2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
}


def find_non_cognate_pairs(df, id_col, sep, cognate_col):
    """Same matching rule as pairing_utils.find_non_cognate_indices / cross_model_analysis's
    find_non_cognate_pairs, generalized over id_col/sep/cognate_col since riam/
    fast_relax_riam and each spm's own metrics file don't share a schema. Returns each
    match's canonical (cognate_key, competitor_key) identity so diffs can be matched
    across all three groups regardless of ID format."""
    cognate_indices = df.index[df[cognate_col] == 1]
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


def rosetta_best_metric_diffs(project, spm, dataset):
    path = METRICS_DIR / f"{project}_{spm}_{dataset}_combined.csv"
    df = pd.read_csv(path)
    matches = find_non_cognate_pairs(df, "protein_pair", "_vs_", "cognate_status")

    diffs_by_metric = {}
    for metric in df.columns:
        if metric.strip() in ROSETTA_UNWANTED:
            continue
        oriented = orient_rosetta_scores(df, metric)
        std = oriented.std()
        if std == 0:
            continue
        standardized = (oriented - oriented.mean()) / std
        diffs_by_metric[metric] = {
            (ck, ck2): standardized.loc[ci] - standardized.loc[ni] for ci, ni, ck, ck2 in matches
        }
    best_metric = max(diffs_by_metric, key=lambda m: np.median(list(diffs_by_metric[m].values())))
    return best_metric, diffs_by_metric[best_metric]


def own_best_metric_diffs(spm, dataset):
    path, id_col, sep = OWN_PATHS[spm][dataset]
    df = pd.read_csv(path)
    matches = find_non_cognate_pairs(df, id_col, sep, "cognate_interaction")

    diffs_by_metric = {}
    for metric in df.columns:
        if metric.strip() in OWN_UNWANTED[spm]:
            continue
        oriented = -df[metric] if metric in OWN_LOWER_IS_BETTER[spm] else df[metric]
        std = oriented.std()
        if std == 0:
            continue
        standardized = (oriented - oriented.mean()) / std
        diffs_by_metric[metric] = {
            (ck, ck2): standardized.loc[ci] - standardized.loc[ni] for ci, ni, ck, ck2 in matches
        }
    best_metric = max(diffs_by_metric, key=lambda m: np.median(list(diffs_by_metric[m].values())))
    return best_metric, diffs_by_metric[best_metric]


def collapse_by_cognate(keyed_diffs):
    by_cognate = {}
    for (cognate_key, _competitor_key), diff in keyed_diffs.items():
        by_cognate.setdefault(cognate_key, []).append(diff)
    return {cognate_key: float(np.median(vals)) for cognate_key, vals in by_cognate.items()}


def plot_comparison_boxplot(diffs_by_group, label_by_group, spm, dataset, out_dir):
    labels = [label_by_group[g] for g in GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.boxplot([list(diffs_by_group[g].values()) for g in GROUP_ORDER], tick_labels=labels, showfliers=False)
    for i, g in enumerate(GROUP_ORDER, start=1):
        y = list(diffs_by_group[g].values())
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, alpha=0.4, color=GROUP_COLORS[g], s=12, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Standardized differential (SD units)")
    ax.set_title(f"riam vs fast_relax_riam vs own metric — {spm} ({dataset})")
    plt.tight_layout()
    plt.savefig(out_dir / "riam_vs_own_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_wilcoxon_family(diffs_by_group, label_by_group, spm, dataset, out_dir, suffix, title_note):
    rows = []
    for g_a, g_b in combinations(GROUP_ORDER, 2):
        shared_keys = sorted(set(diffs_by_group[g_a]) & set(diffs_by_group[g_b]))
        a_vals = np.array([diffs_by_group[g_a][k] for k in shared_keys])
        b_vals = np.array([diffs_by_group[g_b][k] for k in shared_keys])

        stat, p = wilcoxon(a_vals, b_vals)
        rows.append({
            "spm": spm, "dataset": dataset,
            "group_a": g_a, "label_a": label_by_group[g_a],
            "group_b": g_b, "label_b": label_by_group[g_b],
            "n_matched_pairs": len(shared_keys),
            "median_diff_a_minus_b": float(np.median(a_vals - b_vals)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "p_value_bonferroni": float(min(p * N_COMPARISONS, 1.0)),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / f"riam_vs_own_wilcoxon{suffix}.csv", index=False)
    print(f"--- {spm} {dataset}{title_note} ---")
    print(results_df.to_string(index=False))
    print()

    n = len(GROUP_ORDER)
    neg_log_p = np.full((n, n), np.nan)
    raw_p = np.full((n, n), np.nan)
    for row in rows:
        i, j = GROUP_ORDER.index(row["group_a"]), GROUP_ORDER.index(row["group_b"])
        val = -np.log10(max(row["p_value_bonferroni"], 1e-300))
        neg_log_p[i, j] = neg_log_p[j, i] = val
        raw_p[i, j] = raw_p[j, i] = row["p_value_bonferroni"]

    vmax = max(np.nanmax(neg_log_p), -np.log10(ALPHA))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
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
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_yticklabels(GROUP_ORDER)
    ax.set_title(f"Wilcoxon signed-rank — {spm} ({dataset}){title_note} (N={n_pairs})\n"
                 f"(bordered = significant at Bonferroni-corrected α={ALPHA})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(p, Bonferroni-corrected)")
    plt.tight_layout()
    plt.savefig(out_dir / f"riam_vs_own_wilcoxon{suffix}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


for spm in SPM_ORDER:
    for dataset in DATASETS:
        out_dir = RESULTS_DIR / spm / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        riam_metric, riam_diffs = rosetta_best_metric_diffs("riam", spm, dataset)
        fr_metric, fr_diffs = rosetta_best_metric_diffs("fast_relax_riam", spm, dataset)
        own_metric, own_diffs = own_best_metric_diffs(spm, dataset)

        diffs = {"riam": riam_diffs, "fast_relax_riam": fr_diffs, "own": own_diffs}
        labels = {
            "riam": f"riam\n({riam_metric})",
            "fast_relax_riam": f"fast_relax_riam\n({fr_metric})",
            "own": f"{spm}\n({own_metric})",
        }

        plot_comparison_boxplot(diffs, labels, spm, dataset, out_dir)
        run_wilcoxon_family(diffs, labels, spm, dataset, out_dir,
                             suffix="", title_note=" (per-comparison)")

        collapsed = {g: collapse_by_cognate(diffs[g]) for g in GROUP_ORDER}
        run_wilcoxon_family(collapsed, labels, spm, dataset, out_dir,
                             suffix="_collapsed", title_note=" (cognate-pair collapsed)")
