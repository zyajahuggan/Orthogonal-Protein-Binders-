import pandas as pd
from pathlib import Path
import re 



metrics_path = Path(__file__).resolve().parent.parent / "dhd_metrics_af3.csv" 

df = pd.read_csv(f"{metrics_path}")

cognate_status = [] 
pattern = re.compile(r"[.*a.*a|.*b.*b]")
for job in df['sample']:
    found = None
    found = pattern.search(job)
    if found == None:
        cognate_status.append(0)
    else:
        cognate_status.append(1)
df["cognate_interaction"] = cognate_status
df.to_csv(f"{metrics_path}", index = False)