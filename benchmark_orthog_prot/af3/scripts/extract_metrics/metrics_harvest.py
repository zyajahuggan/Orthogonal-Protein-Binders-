import os
import json
import pandas as pd
import numpy as np

base_dir = "/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/af3_benchmark/outputs/dhd"

rows = []

for sample_name in os.listdir(base_dir):
    sample_dir = os.path.join(base_dir, sample_name)
    if not os.path.isdir(sample_dir):
        continue

    # Load summary confidences
    summary_file = os.path.join(sample_dir, f"{sample_name}_summary_confidences.json")
    if not os.path.exists(summary_file):
        print(f"Warning: no summary confidences found for {sample_name}")
        continue

    with open(summary_file) as f:
        data = json.load(f)

    # Find best-ranked seed/sample from ranking_scores.csv
    ranking_file = os.path.join(sample_dir, "ranking_scores.csv")
    if not os.path.exists(ranking_file):
        print(f"Warning: no ranking_scores.csv found for {sample_name}")
        continue

    ranking_df = pd.read_csv(ranking_file)
    best = ranking_df.loc[ranking_df["ranking_score"].idxmax()]
    best_seed = int(best["seed"])
    best_sample = int(best["sample"])

    assert abs(data["ranking_score"] - best["ranking_score"]) < 1e-6, f"Mismatch for {sample_name}"

    # Load plddt from best seed/sample
    conf_file = os.path.join(sample_dir, f"seed-{best_seed}_sample-{best_sample}", "confidences.json")
    if not os.path.exists(conf_file):
        print(f"Warning: no confidences.json found for {sample_name} seed-{best_seed}_sample-{best_sample}")
        continue

    with open(conf_file) as f:
        conf_data = json.load(f)

    mean_plddt = float(np.mean(conf_data["atom_plddts"]))

    row = {
        "sample":                 sample_name,
        "ranking_score":          data["ranking_score"],
        "iptm":                   data["iptm"],
        "ptm":                    data["ptm"],
        "mean_plddt":             mean_plddt,
        "best_seed":              best_seed,
        "best_sample":            best_sample,
        "fraction_disordered":    data["fraction_disordered"],
        "has_clash":              data["has_clash"],
        "chain_iptm_0":           data["chain_iptm"][0],
        "chain_iptm_1":           data["chain_iptm"][1],
        "chain_ptm_0":            data["chain_ptm"][0],
        "chain_ptm_1":            data["chain_ptm"][1],
        "chain_pair_iptm_0_1":    data["chain_pair_iptm"][0][1],
        "chain_pair_iptm_1_0":    data["chain_pair_iptm"][1][0],
        "chain_pair_pae_min_0_1": data["chain_pair_pae_min"][0][1],
        "chain_pair_pae_min_1_0": data["chain_pair_pae_min"][1][0],
    }
    rows.append(row)


df = pd.DataFrame(rows)
df = df.sort_values("ranking_score", ascending=False)
af3_dir = "/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/af3_benchmark/"
metrics_dir = os.path.join(af3_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)
df.to_csv(os.path.join(metrics_dir, "dhd_metrics_af3.csv"), index=False)
print(f"Harvested {len(df)} samples -> {metrics_dir}/af3_metrics.csv")
print(f"Harvested {len(df)} samples -> af3_metrics.csv")
print(df.head(10))

