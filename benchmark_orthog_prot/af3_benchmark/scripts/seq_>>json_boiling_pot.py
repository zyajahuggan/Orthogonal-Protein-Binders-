import json
import os
from itertools import combinations_with_replacement

# Define sequences
proteins = {
    "protA": "MKTAYIAKQR...",
    "protB": "GASVLRMLSS...",
    "protC": "EVQLLESGGGL...",
    "protD": "DIQMTQSPSTL...",
    "protE": "PDQSKRWNRYR...",
    "protF": "ANOTHESEQ...",
}

SEEDS = list(range(1, 11))  # adjust seeds (250)
OUTPUT_DIR = "af3_inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for prot1, prot2 in combinations_with_replacement(proteins.keys(), 2):
    name = f"{prot1}_vs_{prot2}"

    # Handle homodimer — same sequence, use id: ["A", "B"]
    if prot1 == prot2:
        sequences = [{"protein": {"id": ["A", "B"], "sequence": proteins[prot1]}}]
    else:
        sequences = [
            {"protein": {"id": ["A"], "sequence": proteins[prot1]}},
            {"protein": {"id": ["B"], "sequence": proteins[prot2]}},
        ]

    job = {
        "name": name,
        "dialect": "alphafold3",
        "version": 1,
        "sequences": sequences,
        "modelSeeds": SEEDS,
    }

    with open(f"{OUTPUT_DIR}/{name}.json", "w") as f:
        json.dump(job, f, indent=4)

    print(f"Created: {name}.json")

print(f"\nTotal jobs: {sum(1 for _ in combinations_with_replacement(proteins, 2))}")
