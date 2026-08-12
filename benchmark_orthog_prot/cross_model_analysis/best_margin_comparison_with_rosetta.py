"""Score-margin (standardized differential) comparison extended with Rosetta's two scoring
variants (RIAM, FastRelax), alongside best_margin_comparison.py's AF3/Boltz-2/Chai/ESMFold2
boxplot columns. New file -- does not modify or overwrite best_margin_comparison.py or its
outputs.

Rosetta re-scores each DL model's predicted structures rather than generating its own, so
there's one riam_{spm}_{dataset}_combined.csv and one fast_relax_riam_{spm}_{dataset}_combined.csv
per spm (af3/boltz/chai/esmfold2). Same reasoning as best_winrate_comparison_with_rosetta.py:
each spm file is standardized (z-scored) and matched to its own non-cognate competitors
independently (protein-pair casing differs across spm files, so cross-spm matches would be
silently wrong), then the resulting per-comparison differentials are pooled across the 4
spms per candidate metric before picking the metric with the highest pooled median -- same
"highest median all-pairs differential wins" rule best_margin_comparison.py uses per DL
model, just pooled over 4 structure sets before picking instead of computed on 1.

ROSETTA_LOWER_IS_BETTER/ROSETTA_UNWANTED are copied from py_rosetta/scripts/analysis/
pairing_utils.py, same cross-project duplication precedent as py_rosetta/scripts/analysis/
riam_vs_own_metric_comparison.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROSETTA_METRICS_DIR = BASE / "py_rosetta" / "metrics" / "metrics"
rng = np.random.default_rng(42)

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2", "RIAM", "FastRelax"]
MODEL_COLORS = {
    "AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300",
    "RIAM": "#8456ce", "FastRelax": "#c0392b",
}
SPMS = ["af3", "boltz", "chai", "esmfold2"]
ROSETTA_VARIANTS = {"RIAM": "riam", "FastRelax": "fast_relax_riam"}

# (metrics path, ID column, separator) per project per dataset
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

# copied from py_rosetta/scripts/analysis/pairing_utils.py
ROSETTA_LOWER_IS_BETTER = [
    "binder_score", "surface_hydrophobicity", "interface_dG",
    "interface_dG_SASA_ratio", "interface_delta_unsat_hbonds",
    "interface_delta_unsat_hbonds_percentage",
]
ROSETTA_UNWANTED = ["protein_pair", "cognate_status"]


def find_non_cognate_indices(df, id_col, sep, cognate_col):
    """For every cognate row, find the row indices of every non-cognate row that shares
    one of its two proteins."""
    cognate_indices = df.index[df[cognate_col] == 1]
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


def dl_model_best_margin(model, dataset):
    path, id_col, sep = PATHS[model][dataset]
    df = pd.read_csv(path)
    non_cognate_indices = find_non_cognate_indices(df, id_col, sep, "cognate_interaction")

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
    return best_metric, differentials_by_metric[best_metric]


def rosetta_pooled_best_margin(variant_project, dataset):
    """Per spm, z-score and match cognate/non-cognate independently; pool the resulting
    diffs across the 4 spms per candidate metric; pick the metric with the highest pooled
    median."""
    per_spm = []
    for spm in SPMS:
        path = ROSETTA_METRICS_DIR / f"{variant_project}_{spm}_{dataset}_combined.csv"
        df = pd.read_csv(path)
        non_cognate_indices = find_non_cognate_indices(df, "protein_pair", "_vs_", "cognate_status")
        per_spm.append((df, non_cognate_indices))

    pooled_diffs_by_metric = {}
    for metric in per_spm[0][0].columns:
        if metric.strip() in ROSETTA_UNWANTED:
            continue
        diffs = []
        for df, non_cognate_indices in per_spm:
            oriented = orient_scores(df, metric, ROSETTA_LOWER_IS_BETTER)
            std = oriented.std()
            if std == 0:
                diffs = None
                break
            standardized = (oriented - oriented.mean()) / std
            for cog_idx, non_cog_idxs in non_cognate_indices.items():
                cog_score = standardized.loc[cog_idx]
                diffs.extend(cog_score - standardized.loc[i] for i in non_cog_idxs)
        if diffs:
            pooled_diffs_by_metric[metric] = diffs

    best_metric = max(pooled_diffs_by_metric, key=lambda m: np.median(pooled_diffs_by_metric[m]))
    return best_metric, pooled_diffs_by_metric[best_metric]


for dataset in ["cop", "dhd"]:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    best_metric_by_model = {}
    differentials_by_model = {}
    summary_rows = []

    for model in ["AF3", "Boltz-2", "Chai", "ESMFold2"]:
        best_metric, diffs = dl_model_best_margin(model, dataset)
        best_metric_by_model[model] = best_metric
        differentials_by_model[model] = diffs
        summary_rows.append({
            "model": model, "best_metric": best_metric,
            "median_differential": float(np.median(diffs)),
            "total_differential": float(np.sum(diffs)),
            "n_comparisons": len(diffs),
        })

    for model, variant_project in ROSETTA_VARIANTS.items():
        best_metric, diffs = rosetta_pooled_best_margin(variant_project, dataset)
        best_metric_by_model[model] = best_metric
        differentials_by_model[model] = diffs
        summary_rows.append({
            "model": model, "best_metric": best_metric,
            "median_differential": float(np.median(diffs)),
            "total_differential": float(np.sum(diffs)),
            "n_comparisons": len(diffs),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "best_margin_comparison_with_rosetta.csv", index=False)

    # --- box+jitter, one column per model showing its own best metric's full distribution ---
    labels = [f"{model}\n({best_metric_by_model[model]})" for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot([differentials_by_model[model] for model in MODEL_ORDER], tick_labels=labels, showfliers=False)
    for i, model in enumerate(MODEL_ORDER, start=1):
        y = differentials_by_model[model]
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, alpha=0.4, color=MODEL_COLORS[model], s=12, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Standardized differential (SD units)")
    ax.set_title(f"Best-margin metric per model, incl. Rosetta (pooled) — {dataset}")
    plt.tight_layout()
    plt.savefig(results_dir / "best_margin_comparison_with_rosetta.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"--- {dataset} ---")
    print(summary_df.to_string(index=False))
