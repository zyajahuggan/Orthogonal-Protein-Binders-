import pandas as pd
from pathlib import Path
import re 

metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "dhd_metrics_af3.csv"
output_path = metrics_path.with_stem(metrics_path.stem + "_filtered")  
# -> dhd_metrics_af3_filtered.csv, same folder

df = pd.read_csv(f"{metrics_path}")

patterns_to_drop = ['6a', '6b', '13_xaaaa', '13_xaaab']

mask = df['sample'].str.contains('|'.join(patterns_to_drop), case=False, na=False)
df = df[~mask]

df.to_csv(f"{output_path}", index=False)