"""
Compute inter-chain and intra-chain PAE for every folding job in a project.

Each job in the input fasta has a sequence like "CHAIN_A:CHAIN_B". The PAE
matrix saved for that job is (len_A + len_B) x (len_A + len_B), where the
first len_A rows/cols correspond to chain A and the rest to chain B. We use
the chain lengths from the fasta to slice out:
  - the intra-chain A block
  - the intra-chain B block
  - the inter-chain block (A-B and B-A, averaged together)

To run this for a different project, just change PROJECT below.
"""

from pathlib import Path
import csv
import numpy as np

# ---- change this to switch projects ----
PROJECT = "cross_docking"          # or "dhd"
FASTA_NAME = "cross_docking.fasta"  # or "dhd.fasta"
INPUT_DIR_NAME = "cross_docking"    # or "dhd_inputs"
OUTPUT_DIR_NAME = "cross_docking_outputs"  # or "dhd_outputs"
# -----------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
FASTA_PATH = PROJECT_DIR / "inputs" / INPUT_DIR_NAME / FASTA_NAME
OUTPUTS_DIR = PROJECT_DIR / "outputs" / OUTPUT_DIR_NAME
# combine_metrics_pae.py reads its PAE summary from metrics/, so write it there too
RESULTS_CSV = PROJECT_DIR / "metrics" / f"{PROJECT}_pae_summary.csv"


def parse_fasta_chain_lengths(fasta_path):
    """Return {job_name: [len_chain_A, len_chain_B, ...]} from a fasta file."""
    chain_lengths = {}
    with open(fasta_path) as f:
        name, seq = None, ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    chain_lengths[name] = [len(c) for c in seq.split(":")]
                name = line[1:]
                seq = ""
            else:
                seq += line
        if name:
            chain_lengths[name] = [len(c) for c in seq.split(":")]
    return chain_lengths


def intra_inter_pae(pae, chain_lengths):
    """Given a PAE matrix and chain lengths, return (intra_means, inter_mean).

    intra_means is a list with one mean per chain (its own diagonal block).
    inter_mean is the mean of all off-diagonal (between-chain) blocks.
    """
    bounds = np.cumsum([0] + chain_lengths)

    intra_means = []
    for i in range(len(chain_lengths)):
        start, end = bounds[i], bounds[i + 1]
        block = pae[start:end, start:end]
        intra_means.append(float(block.mean()))

    inter_values = []
    for i in range(len(chain_lengths)):
        for j in range(len(chain_lengths)):
            if i == j:
                continue
            i_start, i_end = bounds[i], bounds[i + 1]
            j_start, j_end = bounds[j], bounds[j + 1]
            inter_values.append(pae[i_start:i_end, j_start:j_end].ravel())
    inter_mean = float(np.concatenate(inter_values).mean()) if inter_values else None

    return intra_means, inter_mean


def main():
    chain_lengths_by_job = parse_fasta_chain_lengths(FASTA_PATH)

    rows = []
    for job_name, chain_lengths in chain_lengths_by_job.items():
        pae_file = OUTPUTS_DIR / job_name / f"{job_name}_pae.npy"
        if not pae_file.exists():
            print(f"Skipping {job_name}: no PAE file found")
            continue

        pae = np.load(pae_file)
        intra_means, inter_mean = intra_inter_pae(pae, chain_lengths)

        row = {"job_name": job_name, "inter_chain_pae": inter_mean}
        for i, m in enumerate(intra_means):
            row[f"intra_chain_{i}_pae"] = m
        rows.append(row)
        print(f"{job_name}: intra={intra_means}, inter={inter_mean:.3f}")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
