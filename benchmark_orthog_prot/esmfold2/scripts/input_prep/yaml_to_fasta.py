"""
Convert cross-docking YAML files to FASTA format.

Each YAML file contains two protein sequences (chain A and B).
Output FASTA format mirrors dhd_inputs/dhd.fasta:
  >entry_id
  seqA:seqB
"""

import argparse
import sys
from pathlib import Path

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent


def yaml_to_fasta_entry(yaml_path: Path) -> str:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    proteins = data.get("sequences", [])
    seqs = {}
    for entry in proteins:
        protein = entry.get("protein", {})
        seqs[protein["id"]] = protein["sequence"]

    if "A" not in seqs or "B" not in seqs:
        print(f"WARNING: {yaml_path.name} missing chain A or B, skipping", file=sys.stderr)
        return ""

    entry_id = yaml_path.stem
    return f">{entry_id}\n{seqs['A']}:{seqs['B']}\n"


def main():
    parser = argparse.ArgumentParser(description="Convert cross-docking YAML files to FASTA")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(BENCHMARK_DIR / "inputs" / "cross_docking"),
        help="Directory containing .yaml files (default: <benchmark_dir>/inputs/cross_docking)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output FASTA file (default: <input_dir>/cross_docking.fasta)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"ERROR: {input_dir} is not a directory")

    yaml_files = sorted(input_dir.glob("*.yaml"))
    if not yaml_files:
        sys.exit(f"ERROR: No .yaml files found in {input_dir}")

    output_path = Path(args.output) if args.output else input_dir / "cross_docking.fasta"

    entries = []
    for yaml_file in yaml_files:
        entry = yaml_to_fasta_entry(yaml_file)
        if entry:
            entries.append(entry)

    with open(output_path, "w") as f:
        f.write("".join(entries))

    print(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
