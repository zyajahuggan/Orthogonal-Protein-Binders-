"""Compares each model's single BEST metric (by median all-pairs standardized differential)
against the others' best metrics, per dataset -- the margin analogue of
best_metric_comparison.py's AUC comparison and best_rank_comparison.py's rank comparison.

"Best metric" and the differential itself are computed the same way each project's own
all_score_differential.py does: for every cognate pair, its standardized difference against
EVERY non-cognate competitor that shares a protein with it (not just the toughest one, which
is what score_margin.py does instead) -- ranked by median, same ordering
all_score_differential.py's own boxplot uses. Neither script saves these per-pair
differentials to CSV (only the boxplot/barplot PNGs), so this recomputes them, generalized
over each model's own ID column and separator -- same PATHS/LOWER_IS_BETTER/UNWANTED dicts
as best_rank_comparison.py.

Output is a single box+jitter plot per dataset, one column per model showing that model's
own best metric's full distribution -- same visual style as all_score_differential.py's
boxplot, but with "model (its best metric)" on the x-axis instead of "metric" for one model,
so the four models' winning distributions sit side by side.

Rosetta and ProteinMPNN aren't included, for the same reason as best_metric_comparison.py
and best_rank_comparison.py: neither has metrics in the per-sample CSV format this logic
expects.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(42)

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

# (metrics path, ID column, separator) per project per dataset -- copied from
# best_rank_comparison.py, which already established these per-project quirks
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
    one of its two proteins. Same logic as each project's pairing_utils.py, generalized
    over id_col instead of hardcoding it."""
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


for dataset in ["cop", "dhd"]:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

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
                continue  # zero-variance metric, same skip rule as all_score_differential.py
            standardized = (oriented - oriented.mean()) / std
            diffs = []
            for cog_idx, non_cog_idxs in non_cognate_indices.items():
                cog_score = standardized.loc[cog_idx]
                diffs.extend(cog_score - standardized.loc[i] for i in non_cog_idxs)
            differentials_by_metric[metric] = diffs

        # highest median all-pairs differential wins -- same ranking all_score_differential.py's
        # own boxplot uses (diff_df.median().sort_values(...))
        best_metric = max(differentials_by_metric, key=lambda m: np.median(differentials_by_metric[m]))
        best_metric_by_model[model] = best_metric
        differentials_by_model[model] = differentials_by_metric[best_metric]

        summary_rows.append({
            "model": model, "best_metric": best_metric,
            "median_differential": float(np.median(differentials_by_metric[best_metric])),
            "total_differential": float(np.sum(differentials_by_metric[best_metric])),
            "n_comparisons": len(differentials_by_metric[best_metric]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "best_margin_comparison.csv", index=False)

    # --- box+jitter, one column per model showing its own best metric's full distribution ---
    labels = [f"{model}\n({best_metric_by_model[model]})" for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([differentials_by_model[model] for model in MODEL_ORDER], tick_labels=labels, showfliers=False)
    for i, model in enumerate(MODEL_ORDER, start=1):
        y = differentials_by_model[model]
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, alpha=0.4, color=MODEL_COLORS[model], s=12, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Standardized differential (SD units)")
    ax.set_title(f"Best-margin metric per model, all-pairs differential — {dataset}")
    plt.tight_layout()
    plt.savefig(results_dir / "best_margin_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"--- {dataset} ---")
    print(summary_df.to_string(index=False))
