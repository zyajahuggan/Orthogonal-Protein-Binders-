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
PROJECT_DIR = SCRIPT_DIR.parent
FASTA_PATH = PROJECT_DIR / "inputs" / INPUT_DIR_NAME / FASTA_NAME
OUTPUTS_DIR = PROJECT_DIR / "outputs" / OUTPUT_DIR_NAME
RESULTS_CSV = PROJECT_DIR / "outputs" / f"{PROJECT}_pae_summary.csv"


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
        
print(parse_fasta_chain_lengths(FASTA_PATH))