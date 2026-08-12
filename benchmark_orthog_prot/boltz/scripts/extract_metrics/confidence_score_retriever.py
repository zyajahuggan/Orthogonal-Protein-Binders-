import os
import json
import pandas as pd
from pathlib import Path

benchmark_dir = Path(__file__).resolve().parent.parent.parent
output_dir = benchmark_dir / "outputs" / "cross_docking_outputs"
rows = []

for job_name in os.listdir(output_dir):
    job_dir = os.path.join(output_dir, job_name)
    if not os.path.isdir(job_dir):
        continue
    
    # Some outputs have a boltz_results_{job_name} subdirectory, others don't
    confidence_file = os.path.join(
        job_dir,
        f"boltz_results_{job_name}",
        "predictions",
        job_name,
        f"confidence_{job_name}_model_0.json"
    )
    if not os.path.exists(confidence_file):
        confidence_file = os.path.join(
            job_dir,
            "predictions",
            job_name,
            f"confidence_{job_name}_model_0.json"
        )

    if not os.path.exists(confidence_file):
        print(f"Warning: no confidence file found for {job_name}")
        continue

    with open(confidence_file) as f:
        data = json.load(f)

    row = {
        "job_name":         job_name,
        "confidence_score": data["confidence_score"],
        "ptm":              data["ptm"],
        "iptm":             data["iptm"],
        "protein_iptm":     data["protein_iptm"],
        "complex_plddt":    data["complex_plddt"],
        "complex_iplddt":   data["complex_iplddt"],
        "complex_pde":      data["complex_pde"],
        "complex_ipde":     data["complex_ipde"],
        "chain_0_ptm":      data["chains_ptm"]["0"],
        "chain_1_ptm":      data["chains_ptm"]["1"],
        "iptm_0_1":         data["pair_chains_iptm"]["0"]["1"],
        "iptm_1_0":         data["pair_chains_iptm"]["1"]["0"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values("confidence_score", ascending=False)
metrics_csv_path = benchmark_dir / "metrics" / "boltz_metrics_cop.csv"
df.to_csv(metrics_csv_path, index=False)
print(f"Harvested {len(df)} jobs -> {metrics_csv_path}")
print(df.head())

