import pandas as pd
from pathlib import Path
import re



metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "dhd_combined_metrics_filtered.csv"

df = pd.read_csv(metrics_path)

cognate_status = []
pattern = re.compile(r"^(.+)(?:a__\1b|b__\1a)$")  # esmfold2 dhd job names join pairs with "__"
for job in df['job_name']:
    found = pattern.search(job)
    if found is None:
        cognate_status.append(0)
    else:
        cognate_status.append(1)

df["cognate_interaction"] = cognate_status
df.to_csv(metrics_path, index = False)
