import yaml
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"

for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".yaml"):
        path = os.path.join(OUTPUT_DIR, filename)

        try:
            with open(path) as f:
                yaml.safe_load(f)

            print(f"✓ {filename}")

        except Exception as e:
            print(f"✗ {filename}")
            print(e)