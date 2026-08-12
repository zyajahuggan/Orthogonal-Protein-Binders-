"""For every structure-prediction model (spm) and dataset, find the single metric with
the highest both-cognates-win rate (same combo + win-rate logic as pair_stats.py),
computed three ways: on the spm's own "native" confidence metrics (read directly from
that spm's own benchmark directory, e.g. ../../../af3/metrics/), on the "riam" project,
and on the "fast_relax_riam" project. Then report how the best win rate changes across
those three, for every spm/dataset combination.

NATIVE_CONFIG mirrors the id column / cognate column / excluded columns / lower-is-better
columns that each spm's own pairing_utils.py already defines -- if those files change,
this table needs updating too.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pairing_utils import UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS, PROJECTS, spms, datasets, BENCHMARK_DIR

BENCHMARK_ROOT = BENCHMARK_DIR.parent

# diverging blue/red pair (dataviz skill palette) -- blue reads as "improved",
# red as "got worse" when fast_relax_riam's win rate is compared to riam's
INCREASE_COLOR = "#2a78d6"
DECREASE_COLOR = "#e34948"

# fixed categorical colors (dataviz skill palette, slots 1/2/7) for the 3-way grouped plot
NATIVE_COLOR = "#2a78d6"
RIAM_COLOR = "#008300"
FAST_RELAX_COLOR = "#4a3aa7"

# raw bookkeeping counts/normalization constants add_ipsae_metrics.py adds alongside the
# real ipSAE scores -- not confidence signals themselves, so excluded from the metric search
IPSAE_BOOKKEEPING = ["ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom", "ipsae_d0res", "ipsae_d0chn",
                      "ipsae_d0dom", "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2"]

NATIVE_CONFIG = {
    "af3": {
        "id_column": "sample",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["sample", "best_seed", "best_sample", "cognate_interaction",
                              "has_clash", "interface_hbonds"] + IPSAE_BOOKKEEPING,
        "lower_is_better_columns": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0",
                                     "fraction_disordered"],
        "dhd": ("dhd_metrics_af3_filtered.csv", "_vs_"),
        "cop": ("cop_metrics_af3.csv", "_vs_"),
    },
    "boltz": {
        "id_column": "job_name",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
        "lower_is_better_columns": ["complex_pde", "complex_ipde"],
        "dhd": ("boltz_metrics_dhd_filtered.csv", "_vs_"),
        "cop": ("boltz_metrics_cop.csv", "_vs_"),
    },
    "chai": {
        "id_column": "project",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["project", "cognate_interaction",
                              "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
                              "chain_chain_clashes_1_0", "chain_chain_clashes_1_1",
                              "has_inter_chain_clashes", "interface_hbonds"],
        "lower_is_better_columns": [],
        "dhd": ("dhd_metrics_filtered.csv", "_vs_"),
        "cop": ("cop_metrics.csv", "_vs_"),
    },
    "esmfold2": {
        "id_column": "job_name",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
        "lower_is_better_columns": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
        "dhd": ("dhd_combined_metrics_filtered.csv", "__"),
        "cop": ("cop_combined_metrics.csv", "_vs_"),
    },
}


def best_win_rate(df, sep, id_column, cognate_column, unwanted_columns, lower_is_better_columns):
    """Runs pair_stats.py's combo-finding + win-rate computation for every metric in
    df, and returns only the name and win rate of the single best-scoring metric."""
    cognate_indices = df.index[df[cognate_column] == 1].tolist()

    # chain_sets[i] is the set of the two proteins in cognate row i
    chain_sets = {}
    for i in cognate_indices:
        chain_sets[i] = set(df.loc[i, id_column].split(sep))

    # every pair of cognate rows that has "cross" non-cognate rows between them
    combos = []
    for i, j in itertools.combinations(cognate_indices, 2):
        chains_i = chain_sets[i]
        chains_j = chain_sets[j]
        cross_rows = []
        for idx in df.index:
            if idx == i or idx == j:
                continue
            row_chains = set(df.loc[idx, id_column].split(sep))
            if len(row_chains & chains_i) == 1 and len(row_chains & chains_j) == 1:
                cross_rows.append(idx)
        if cross_rows:
            combos.append((i, j, cross_rows))

    best_metric = None
    best_rate = None
    for metric in df.columns:
        if metric.strip() in unwanted_columns:
            continue
        if metric in lower_is_better_columns:
            oriented = -df[metric]
        else:
            oriented = df[metric]

        successes = []
        for i, j, cross_rows in combos:
            values = [oriented.loc[i], oriented.loc[j]]
            for idx in cross_rows:
                values.append(oriented.loc[idx])

            has_missing = False
            for v in values:
                if pd.isna(v):
                    has_missing = True
            if has_missing:
                continue  # skip this combination for this metric if any value is missing

            worst_cognate = min(oriented.loc[i], oriented.loc[j])
            best_noncognate = oriented.loc[cross_rows[0]]
            for idx in cross_rows[1:]:
                val = oriented.loc[idx]
                if val > best_noncognate:
                    best_noncognate = val
            successes.append(worst_cognate > best_noncognate)

        if not successes:
            continue
        rate = sum(successes) / len(successes)
        if best_rate is None or rate > best_rate:
            best_rate = rate
            best_metric = metric

    return best_metric, best_rate


# --- compute the best metric + win rate for every project/spm/dataset combination ---
# keyed by (spm, dataset, project) so the comparison loop below can look all three up
best_by_combo = {}
for cfg in PROJECTS:
    df = pd.read_csv(cfg.metrics_path)
    metric, rate = best_win_rate(df, cfg.sep, "protein_pair", "cognate_status",
                                  UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS)
    best_by_combo[(cfg.spm, cfg.dataset, cfg.project)] = (metric, rate)
    print(f"{cfg.project}_{cfg.spm}_{cfg.dataset}: best metric = {metric} ({rate:.0%} win rate)")

# --- same computation on each spm's own native confidence metrics ---
for spm in spms:
    config = NATIVE_CONFIG[spm]
    for dataset in datasets:
        filename, sep = config[dataset]
        metrics_path = BENCHMARK_ROOT / spm / "metrics" / filename
        df = pd.read_csv(metrics_path)
        metric, rate = best_win_rate(df, sep, config["id_column"], config["cognate_column"],
                                      config["unwanted_columns"], config["lower_is_better_columns"])
        best_by_combo[(spm, dataset, "native")] = (metric, rate)
        print(f"native_{spm}_{dataset}: best metric = {metric} ({rate:.0%} win rate)")

# --- compare native vs riam vs fast_relax_riam, once per spm/dataset combination ---
comparison_rows = []
for spm in spms:
    for dataset in datasets:
        native_metric, native_rate = best_by_combo[(spm, dataset, "native")]
        riam_metric, riam_rate = best_by_combo[(spm, dataset, "riam")]
        fr_metric, fr_rate = best_by_combo[(spm, dataset, "fast_relax_riam")]

        riam_vs_native = riam_rate - native_rate
        fr_vs_native = fr_rate - native_rate
        fr_vs_riam = fr_rate - riam_rate

        print(f"{spm} {dataset}: native = {native_metric} ({native_rate:.0%}), "
              f"riam = {riam_metric} ({riam_rate:.0%}), "
              f"fast_relax_riam = {fr_metric} ({fr_rate:.0%})")
        print(f"    riam - native = {riam_vs_native:+.0%}  "
              f"fast_relax_riam - native = {fr_vs_native:+.0%}  "
              f"fast_relax_riam - riam = {fr_vs_riam:+.0%}")

        comparison_rows.append({
            "spm": spm,
            "dataset": dataset,
            "native_best_metric": native_metric,
            "native_win_rate": native_rate,
            "riam_best_metric": riam_metric,
            "riam_win_rate": riam_rate,
            "fast_relax_riam_best_metric": fr_metric,
            "fast_relax_riam_win_rate": fr_rate,
            "riam_vs_native_delta": riam_vs_native,
            "fast_relax_riam_vs_native_delta": fr_vs_native,
            "fast_relax_riam_vs_riam_delta": fr_vs_riam,
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_dir = BENCHMARK_DIR / "results" / "win_rate_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(comparison_dir / "riam_vs_fast_relax_riam_win_rate_delta.csv", index=False)

# --- diverging barplot: one bar per spm/dataset combo, height is the fast_relax_riam
# vs riam win-rate delta ---
labels = []
deltas = []
bar_colors = []
for row in comparison_rows:
    labels.append(f"{row['spm']}\n{row['dataset']}")
    deltas.append(row["fast_relax_riam_vs_riam_delta"])
    if row["fast_relax_riam_vs_riam_delta"] >= 0:
        bar_colors.append(INCREASE_COLOR)
    else:
        bar_colors.append(DECREASE_COLOR)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, deltas, color=bar_colors)
ax.axhline(0, color="#898781", linewidth=1)  # baseline: no change

# direct label on every bar since there are only 8 -- shows the exact delta
for bar, delta in zip(bars, deltas):
    offset = 0.01 if delta >= 0 else -0.01
    va = "bottom" if delta >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, delta + offset, f"{delta:+.0%}",
            ha="center", va=va, fontsize=9)

ax.margins(y=0.15)  # headroom so the direct labels don't clip against the plot edge
ax.set_ylabel("Change in best-metric win rate (fast_relax_riam − riam)")
ax.set_title("Effect of fast_relax on the best-metric win rate, per spm/dataset")

increase_patch = plt.Rectangle((0, 0), 1, 1, color=INCREASE_COLOR, label="Win rate increased")
decrease_patch = plt.Rectangle((0, 0), 1, 1, color=DECREASE_COLOR, label="Win rate decreased")
ax.legend(handles=[increase_patch, decrease_patch])

plt.tight_layout()
plt.savefig(comparison_dir / "riam_vs_fast_relax_riam_win_rate_delta.png", dpi=300)
plt.close()

# --- grouped barplot: native vs riam vs fast_relax_riam absolute win rates, per
# spm/dataset combo ---
group_labels = []
native_rates = []
riam_rates = []
fr_rates = []
for row in comparison_rows:
    group_labels.append(f"{row['spm']}\n{row['dataset']}")
    native_rates.append(row["native_win_rate"])
    riam_rates.append(row["riam_win_rate"])
    fr_rates.append(row["fast_relax_riam_win_rate"])

x = np.arange(len(group_labels))  # one slot per spm/dataset group
bar_width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - bar_width, native_rates, width=bar_width, color=NATIVE_COLOR, label="native")
ax.bar(x, riam_rates, width=bar_width, color=RIAM_COLOR, label="riam")
ax.bar(x + bar_width, fr_rates, width=bar_width, color=FAST_RELAX_COLOR, label="fast_relax_riam")

ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylim(0, 1)
ax.set_ylabel("Best-metric win rate")
ax.set_title("Best-metric win rate: native spm metrics vs riam vs fast_relax_riam")
ax.legend()
plt.tight_layout()
plt.savefig(comparison_dir / "native_vs_riam_vs_fast_relax_riam_win_rate.png", dpi=300)
plt.close()
