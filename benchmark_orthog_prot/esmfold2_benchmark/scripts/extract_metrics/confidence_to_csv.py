import json
import csv
from pathlib import Path

output_path = Path("outputs/cross_docking_outputs")
metrics_path = Path("metrics/cross_docking_metrics.csv")
metrics_path.parent.mkdir(parents=True, exist_ok=True)

# collect all rows first so we know all the fieldnames before writing the header
rows = []

for project in output_path.iterdir():
    json_file = project / f"{project.name}_metrics.json"
    if not json_file.is_file():
        print(f"WARNING: no metrics.json found in {project.name}")
        continue

    with open(json_file) as f:
        data = json.load(f)

    row = {"job_name": project.name, **data}   # merge project name + metrics into one dict
    rows.append(row)

# write all rows to CSV
fieldnames = ["job_name", "plddt", "ptm", "iptm"]   # define column order explicitly

with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {metrics_path}")



     
        