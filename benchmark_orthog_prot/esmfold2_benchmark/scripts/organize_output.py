from pathlib import Path
import shutil 

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
outputs_dir = project_dir / "outputs"

cif_files = sorted(outputs_dir.glob("*.cif"))

for i in range(len(cif_files)):
    new_dir = outputs_dir / cif_files[i].stem
    new_dir.mkdir(exist_ok=True)
    for item in outputs_dir.iterdir():
        if new_dir.name in f"{item}": 
            shutil.move(f"{item}",f"{new_dir}")
    new_dir = None



