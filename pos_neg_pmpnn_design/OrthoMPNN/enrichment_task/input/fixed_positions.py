#!/usr/bin/env python3
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

chain_lengths = {
    "A": 97, "B": 97,
    "C": 273, "D": 273,
}

structure_name = "triple_renumbered"

design_positions = [22, 26, 29, 31, 49, 52, 68]
design_chains = ["A", "B"]

fixed = {}

for ch, L in chain_lengths.items():
    if ch in design_chains:
        fixed[ch] = [i for i in range(1, L + 1)
                     if i not in design_positions]
    else:
        fixed[ch] = list(range(1, L + 1))

with open(args.output_path, "w") as f:
    f.write(json.dumps({structure_name: fixed}) + "\n")

print(f"✅ Wrote {args.output_path}")