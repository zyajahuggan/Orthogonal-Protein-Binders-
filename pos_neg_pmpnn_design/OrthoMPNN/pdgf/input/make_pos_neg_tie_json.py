#!/usr/bin/env python3
import argparse
import json

def main(args):
    # Load parsed_pdbs.jsonl (one structure per line)
    with open(args.input_path) as f:
        result = json.loads(f.readline())

    # Interface residues (local indexing)
    intf_resi = {
        "A": [22, 26, 29, 31, 49, 52, 68],
        "B": [22, 26, 29, 31, 49, 52, 68],
    }

    tied_positions_list = []

    # Build tied groups ONLY for interface residues
    for i in intf_resi["A"]:
        group = {
            "A": [[i], [ 1.0]],
            "B": [[i], [ 1.0]],
            "E": [[i], [0.0]],
            "F": [[i], [0.0]],
        }
        tied_positions_list.append(group)

    out = {result["name"]: tied_positions_list}

    with open(args.output_path, "w") as f:
        f.write(json.dumps(out) + "\n")

    print(f"Wrote {args.output_path}")
    print(f"Tied {len(tied_positions_list)} interface positions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True,
                        help="parsed_pdbs.jsonl")
    parser.add_argument("--output_path", required=True,
                        help="tied_positions.jsonl")
    args = parser.parse_args()
    main(args)
