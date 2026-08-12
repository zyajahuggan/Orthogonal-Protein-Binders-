"""Compares each model's single BEST metric (by mean rank of the cognate pair among its
non-cognates) against the others' best metrics, per dataset -- the ranking analogue of
best_metric_comparison.py's AUC comparison.

Each model's topk_stats.py only saves a boxplot PNG, not a CSV of rank per metric, so
there's nothing to read off disk here. This recomputes rank the same way topk_stats.py
does (same formula: rank = 1 + count of non-cognates scoring higher than the cognate pair),
reimplemented generically over each model's own ID column and separator rather than
importing each project's pairing_utils.py, since find_non_cognate_indices hardcodes a
different ID column name per project.

Rosetta and ProteinMPNN aren't included, for the same reason as best_metric_comparison.py:
neither has metrics in the per-sample CSV format this ranking logic expects.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

# (metrics path, ID column, separator) per project per dataset -- copied from
# compare_models.py, which already established these per-project quirks
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
    "ESMFold2": ["job_name", "cognate_interaction", "interface_hbonds", "ipsae_iptm_af"] + IPSAE_BOOKKEEPING,
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

    best = []
    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)
        non_cognate_indices = find_non_cognate_indices(df, id_col, sep)

        # expected rank if a metric carried no signal at all: a uniformly random score
        # places the cognate pair anywhere among itself + its non-cognates with equal
        # probability, so its expected rank is (non-cognate count + 2) / 2, averaged
        # across cognate pairs the same way mean_rank_by_metric averages real ranks below
        expected_random_rank = np.mean(
            [(len(non_cog_idxs) + 2) / 2 for non_cog_idxs in non_cognate_indices.values()]
        )

        mean_rank_by_metric = {}
        for metric in df.columns:
            if metric.strip() in UNWANTED[model]:
                continue
            oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
            ranks = []
            for cog_idx, non_cog_idxs in non_cognate_indices.items():
                cog_score = oriented.loc[cog_idx]
                rank = 1 + sum(oriented.loc[i] > cog_score for i in non_cog_idxs)
                ranks.append(rank)
            mean_rank_by_metric[metric] = np.mean(ranks)

        # lower mean rank is better -- rank 1 means the cognate pair beat every non-cognate
        best_metric = min(mean_rank_by_metric, key=mean_rank_by_metric.get)
        best.append({
            "model": model, "best_metric": best_metric, "mean_rank": mean_rank_by_metric[best_metric],
            "expected_random_rank": expected_random_rank,
        })

    best_df = pd.DataFrame(best)
    best_df.to_csv(results_dir / "best_rank_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        best_df["model"], best_df["mean_rank"],
        color=[MODEL_COLORS[m] for m in best_df["model"]],
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="rank 1 (perfect)")
    random_rank = best_df["expected_random_rank"].mean()
    ax.axhline(random_rank, linestyle=":", color="firebrick", linewidth=0.8, label=f"random (~{random_rank:.2f})")
    ax.set_ylabel("Mean rank of cognate pair among non-cognates\n(lower is better)")
    ax.set_ylim(0, max(best_df["mean_rank"].max(), random_rank) * 1.2)
    ax.set_title(f"Best-ranking metric per model — {dataset}")
    ax.legend(fontsize=8)

    for bar, metric, rank in zip(bars, best_df["best_metric"], best_df["mean_rank"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + best_df["mean_rank"].max() * 0.02,
            f"{metric}\n{rank:.2f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(results_dir / "best_rank_comparison.png", dpi=300)
    plt.close()

    print(f"--- {dataset} ---")
    print(best_df.to_string(index=False))
