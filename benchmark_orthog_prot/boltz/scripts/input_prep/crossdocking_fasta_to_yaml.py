import os
import argparse
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent

def parse_fasta(filepath):
    sequences = {}
    current_name = None
    current_seq = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    sequences[current_name] = "".join(current_seq)
                # header format: >protein | name
                current_name = line.split("|")[-1].strip()
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        sequences[current_name] = "".join(current_seq)
    return sequences

def to_yaml(seq_a, seq_b):
    return (
        "sequences:\n"
        "    - protein:\n"
        "        id: A\n"
        f"        sequence: {seq_a}\n"
        "\n"
        "    - protein:\n"
        "        id: B\n"
        f"        sequence: {seq_b}\n"
        "    "
    )

def main():
    parser = argparse.ArgumentParser(
        description="Convert cross-docking FASTA files to Boltz YAML format"
    )
    parser.add_argument(
        "--input-dir",
        default=str(BENCHMARK_DIR / "cross_docking"),
        help="Directory containing cross-docking .fasta files (default: <benchmark_dir>/cross_docking)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_DIR / "inputs" / "cross_docking"),
        help="Directory to write .yaml files into (default: <benchmark_dir>/inputs/cross_docking)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fasta_files = [f for f in os.listdir(args.input_dir) if f.endswith(".fasta")]
    if not fasta_files:
        print(f"No .fasta files found in {args.input_dir}")
        return

    written = 0
    for fname in sorted(fasta_files):
        fpath = os.path.join(args.input_dir, fname)
        sequences = parse_fasta(fpath)

        if len(sequences) != 2:
            print(f"Warning: {fname} has {len(sequences)} sequence(s), expected 2 — skipping")
            continue

        names = list(sequences.keys())
        seq_a = sequences[names[0]]
        seq_b = sequences[names[1]]

        out_name = os.path.splitext(fname)[0] + ".yaml"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w") as f:
            f.write(to_yaml(seq_a, seq_b))
        written += 1

    print(f"Wrote {written} YAML files to {args.output_dir}")

if __name__ == "__main__":
    main()

