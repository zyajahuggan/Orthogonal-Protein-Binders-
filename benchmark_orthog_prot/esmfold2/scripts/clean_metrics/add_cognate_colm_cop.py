import pandas as pd
from pathlib import Path

metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "cop_combined_metrics.csv"

df = pd.read_csv(metrics_path)

cognate_status = []
for job in df["job_name"]:
    ligand, receptor = job.split("_vs_") #tuple unpacking
    if ligand + "s" == receptor:
        cognate_status.append(1)
    else:
        cognate_status.append(0)

df["cognate_interaction"] = cognate_status
df.to_csv(metrics_path, index=False)
