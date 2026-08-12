import pandas as pd
from pathlib import Path
import re



metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "boltz_metrics_dhd_filtered.csv"

df = pd.read_csv(metrics_path)

cognate_status = []
pattern = re.compile(r"^(.+)(?:a_vs_\1b|b_vs_\1a)$")
for job in df['job_name']:
    found = pattern.search(job)
    if found is None:
        cognate_status.append(0)
    else:
        cognate_status.append(1)
df["cognate_interaction"] = cognate_status
df.to_csv(metrics_path, index = False)
