from pathlib import Path
import shutil

counter = 0
unwanted_patterns = ['6a', '6b', '13_xaaaa', '13_xaaab']
project = "dhd"
output_path = Path(__file__).resolve().parent.parent.parent / "outputs" / project

collected_dir = Path(__file__).resolve().parent.parent.parent /  f"{project}_collected_cif_files"
collected_dir.mkdir(parents=True, exist_ok=True)

log_file_path = Path(__file__).resolve().parent.parent.parent / f"{project}_cif_paths.txt"

with open(log_file_path, "w") as f:
    for file in output_path.iterdir():
        if not any(pattern in file.name for pattern in unwanted_patterns):
            cif_file_path = output_path / file.name / f"{file.name}_model.cif"

            if cif_file_path.exists():
                shutil.copy(cif_file_path, collected_dir)
                f.write(f"{cif_file_path}\n")
                counter += 1

print(f"Total files copied: {counter}")
print(f"Files collected in: {collected_dir}")
        
        
