import pandas as pd
from pathlib import Path

metrics_dir = Path(__file__).resolve().parent.parent.parent / "metrics"

pairs = [
    ("dhd_metrics.csv", "dhd_pae_summary.csv", "dhd_combined_metrics.csv"),
    ("cop_metrics.csv", "cop_pae_summary.csv", "cop_combined_metrics.csv"),
]

for metrics_file, pae_file, out_file in pairs:
    metrics_df = pd.read_csv(metrics_dir / metrics_file)
    pae_df = pd.read_csv(metrics_dir / pae_file)

    combined_df = metrics_df.merge(pae_df, on="job_name", validate="one_to_one")

    combined_df.to_csv(metrics_dir / out_file, index=False)
    print(f"wrote {out_file} ({len(combined_df)} rows)")
