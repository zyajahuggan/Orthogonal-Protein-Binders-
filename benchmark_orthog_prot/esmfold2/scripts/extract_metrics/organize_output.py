from pathlib import Path
import shutil

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent.parent
outputs_dir = project_dir / "outputs"

cif_files = sorted(outputs_dir.glob("*.cif"))

for cif_file in cif_files:
    job_name = cif_file.stem
    job_dir = outputs_dir / job_name
    job_dir.mkdir(exist_ok=True)
    # exact stem match ("job_name.cif") or "job_name_" prefix ("job_name_pae.npy",
    # "job_name_metrics.json") -- NOT a substring match, so job "1" doesn't also sweep up "12"
    for item in outputs_dir.iterdir():
        if item.is_file() and (item.stem == job_name or item.name.startswith(job_name + "_")):
            shutil.move(str(item), str(job_dir))



