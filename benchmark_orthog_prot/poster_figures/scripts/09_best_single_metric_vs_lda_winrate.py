"""Win-rate analog of 09_best_single_metric_vs_lda.py (which used AUC). Same three-way
comparison -- LDA in-sample, each model's own best single metric, LDA leave-one-out
cross-validated -- but scored by both-cognates-win-rate (same find_combos/win logic as
best_winrate_per_model_by_dataset.py) instead of AUC, since this poster isn't showing
ROC-AUC numbers. LDA is fit per model (not pooled across models, matching
cross_model_analysis/lda_comparison.py's scope) on that model's own native metric
columns.

New file -- does not modify or overwrite lda_comparison.py, best_winrate_all_datasets.py,
or 09_best_single_metric_vs_lda.py.
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 23,
    "figure.titlesize": 27,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
DATASETS = ["dhd", "cop"]
DATASET_TITLES = {"dhd": "DHD", "cop": "COP"}

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


def orient_scores(df, metric, lower_is_better_columns):
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def find_combos(df, id_col, sep):
    cognate_indices = df.index[df["cognate_interaction"] == 1].tolist()
    chain_sets = {i: set(df.loc[i, id_col].split(sep)) for i in cognate_indices}

    combos = []
    for i, j in itertools.combinations(cognate_indices, 2):
        chains_i, chains_j = chain_sets[i], chain_sets[j]
        cross_rows = [
            idx for idx in df.index
            if idx not in (i, j)
            and len(set(df.loc[idx, id_col].split(sep)) & chains_i) == 1
            and len(set(df.loc[idx, id_col].split(sep)) & chains_j) == 1
        ]
        if cross_rows:
            combos.append((i, j, cross_rows))
    return combos


def win_rate_from_scores(scores, combos):
    n_win = sum(
        min(scores[i], scores[j]) > max(scores[c] for c in cross_rows)
        for i, j, cross_rows in combos
    )
    return n_win, len(combos)


def best_single_metric_winrate(df, id_col, model, combos):
    win_rates = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED[model]:
            continue
        oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
        n_win, n_combos = win_rate_from_scores(oriented, combos)
        win_rates[metric] = n_win
    best_metric = max(win_rates, key=win_rates.get)
    return best_metric, win_rates[best_metric]


IN_SAMPLE_COLOR = "#2a78d6"
BEST_METRIC_COLOR = "#4a3aa7"
CV_COLOR = "#e34948"
RANDOM_COLOR = "#898781"

rates_by_dataset = {}
for dataset in DATASETS:
    rows = []
    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)
        combos = find_combos(df, id_col, sep)
        n_combos = len(combos)

        metric_columns = [c for c in df.columns if c not in UNWANTED[model]]
        data = df[metric_columns].copy()
        for col in LOWER_IS_BETTER[model]:
            if col in data.columns:
                data[col] = -data[col]

        X = StandardScaler().fit_transform(data.values)
        y = df["cognate_interaction"].values

        lda = LinearDiscriminantAnalysis()
        lda_in_sample_scores = pd.Series(lda.fit_transform(X, y).ravel(), index=df.index)
        n_win_in_sample, _ = win_rate_from_scores(lda_in_sample_scores, combos)

        lda_cv_scores = pd.Series(
            cross_val_predict(lda, X, y, cv=LeaveOneOut(), method="decision_function"), index=df.index)
        n_win_cv, _ = win_rate_from_scores(lda_cv_scores, combos)

        best_metric, n_win_best = best_single_metric_winrate(df, id_col, model, combos)

        rows.append({
            "model": model, "n_combos": n_combos,
            "in_sample_rate": n_win_in_sample / n_combos,
            "best_metric_rate": n_win_best / n_combos,
            "cv_rate": n_win_cv / n_combos,
        })
        print(f"{dataset} {model}: LDA in-sample {n_win_in_sample}/{n_combos}  "
              f"best metric ({best_metric}) {n_win_best}/{n_combos}  LDA CV {n_win_cv}/{n_combos}")

    rates_by_dataset[dataset] = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER]

y_max = max(
    rates_by_dataset[d][col].max()
    for d in DATASETS for col in ["in_sample_rate", "best_metric_rate", "cv_rate"]
)
y_top = y_max + 0.18

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

for ax, dataset in zip(axes, DATASETS):
    df_rates = rates_by_dataset[dataset]
    x = np.arange(len(MODEL_ORDER))
    width = 0.26
    ax.bar(x - width, df_rates["in_sample_rate"], width, color=IN_SAMPLE_COLOR, label="LDA In-Sample")
    ax.bar(x, df_rates["best_metric_rate"], width, color=BEST_METRIC_COLOR, label="Best Single Metric")
    ax.bar(x + width, df_rates["cv_rate"], width, color=CV_COLOR, label="LDA Cross-Validated")
    ax.axhline(1 / 6, linestyle="--", linewidth=1.5, color=RANDOM_COLOR, zorder=0, label="Random (1/6)")

    for xi, row in zip(x, df_rates.itertuples()):
        ax.text(xi - width, row.in_sample_rate + 0.015, f"{row.in_sample_rate:.2f}",
                ha="center", fontsize=13, color="#52514e")
        ax.text(xi, row.best_metric_rate + 0.015, f"{row.best_metric_rate:.2f}",
                ha="center", fontsize=14, color="#0b0b0b", fontweight="bold")
        ax.text(xi + width, row.cv_rate + 0.015, f"{row.cv_rate:.2f}",
                ha="center", fontsize=14, color="#0b0b0b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, y_top)
    ax.set_title(DATASET_TITLES[dataset])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Win Rate")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=16)

fig.suptitle(
    "Best Single Metric Win Rate vs. Full-Metric LDA Combination",
    fontsize=24, fontweight="bold",
)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
plt.savefig(OUT_DIR / "best_single_metric_vs_lda_winrate.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"saved {OUT_DIR / 'best_single_metric_vs_lda_winrate.png'}")
