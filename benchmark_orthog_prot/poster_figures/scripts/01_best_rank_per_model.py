"""Poster version of cross_model_analysis/best_rank_comparison.py: same computation
(each model's single best metric by mean rank of the cognate pair among its
non-cognates), replotted with poster-sized fonts and Title Case labels. Saves one PNG
per dataset (DHD, COP) into poster_figures/, matching the original's per-dataset split.

New file -- does not modify or overwrite best_rank_comparison.py or its outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent / "cross_model_analysis"
OUT_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 27,
    "axes.titleweight": "bold",
    "axes.labelsize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}
DATASET_TITLES = {"dhd": "DHD", "cop": "COP"}

# (metrics path, ID column, separator) per project per dataset -- copied from
# best_rank_comparison.py
PATHS = {
    "AF3": {
        "cop": (BASE.parent / "af3/metrics/cop_metrics_af3.csv", "sample", "_vs_"),
        "dhd": (BASE.parent / "af3/metrics/dhd_metrics_af3_filtered.csv", "sample", "_vs_"),
    },
    "Boltz-2": {
        "cop": (BASE.parent / "boltz/metrics/boltz_metrics_cop.csv", "job_name", "_vs_"),
        "dhd": (BASE.parent / "boltz/metrics/boltz_metrics_dhd_filtered.csv", "job_name", "_vs_"),
    },
    "Chai": {
        "cop": (BASE.parent / "chai/metrics/cop_metrics.csv", "project", "_vs_"),
        "dhd": (BASE.parent / "chai/metrics/dhd_metrics_filtered.csv", "project", "_vs_"),
    },
    "ESMFold2": {
        "cop": (BASE.parent / "esmfold2/metrics/cop_combined_metrics.csv", "job_name", "_vs_"),
        "dhd": (BASE.parent / "esmfold2/metrics/dhd_combined_metrics_filtered.csv", "job_name", "__"),
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
    "ESMFold2": ["job_name", "cognate_interaction", "interface_hbonds", "ipsae_iptm_af"] + IPSAE_BOOKKEEPING,
}


def find_non_cognate_indices(df, id_col, sep):
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
    return -df[metric] if metric in lower_is_better_columns else df[metric]


for dataset in ["dhd", "cop"]:
    best = []
    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)
        non_cognate_indices = find_non_cognate_indices(df, id_col, sep)

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

        best_metric = min(mean_rank_by_metric, key=mean_rank_by_metric.get)
        best.append({
            "model": model, "best_metric": best_metric, "mean_rank": mean_rank_by_metric[best_metric],
            "expected_random_rank": expected_random_rank,
        })

    best_df = pd.DataFrame(best)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(
        best_df["model"], best_df["mean_rank"],
        color=[MODEL_COLORS[m] for m in best_df["model"]],
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2, label="Rank 1 (perfect)")
    random_rank = best_df["expected_random_rank"].mean()
    ax.axhline(random_rank, linestyle=":", color="firebrick", linewidth=1.2, label=f"Random (~{random_rank:.2f})")
    ax.set_ylabel("Mean Rank of Cognate Pair Among\nNon-Cognates")
    ax.set_ylim(0, max(best_df["mean_rank"].max(), random_rank) * 1.25)
    ax.set_title(f"Best Ranking Metric per Model — {DATASET_TITLES[dataset]}")
    ax.legend()

    for bar, metric, rank in zip(bars, best_df["best_metric"], best_df["mean_rank"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + best_df["mean_rank"].max() * 0.03,
            f"{metric}\n{rank:.2f}", ha="center", va="bottom", fontsize=15,
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"best_rank_per_model_{dataset}.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    print(f"--- {dataset} ---")
    print(best_df.to_string(index=False))
