#! /usr/bin/env python3
"""
Parses ProteinMPNN .fa output, identifies mutations in chains A and B vs WT,
applies them to the ABCD complex pdb, and saves one mutated pdb per unique
mutation profile for use as Rosetta input.
"""
import os
import re
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from collections import defaultdict

# One-letter to three-letter amino acid map
AA1_TO_AA3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}
AA3_TO_AA1 = {v: k for k, v in AA1_TO_AA3.items()}


# -----------------------------
# Parse .fa file
# -----------------------------
def parse_fa(fa_path):
    """
    Parse ProteinMPNN .fa output.
    Sequence line format: <chainA_seq>/<chainB_seq>
    First entry is WT, subsequent entries are samples.

    Returns:
        wt_A, wt_B : WT sequences for chains A and B
        samples    : list of dicts with keys sample, score, seq_A, seq_B
    """
    samples = []
    wt_A = wt_B = None

    with open(fa_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    first = True
    while i < len(lines):
        if lines[i].startswith('>'):
            header   = lines[i]
            seq_line = lines[i + 1] if i + 1 < len(lines) else ""
            i += 2

            parts = seq_line.split('/')
            seq_A = parts[0] if len(parts) > 0 else ""
            seq_B = parts[1] if len(parts) > 1 else ""

            if first:
                wt_A  = seq_A
                wt_B  = seq_B
                first = False
            else:
                sample_match = re.search(r'sample=(\d+)', header)
                score_match  = re.search(r'score=([\d.]+)', header)
                samples.append({
                    'sample': int(sample_match.group(1)) if sample_match else -1,
                    'score':  float(score_match.group(1)) if score_match else None,
                    'seq_A':  seq_A,
                    'seq_B':  seq_B,
                })
        else:
            i += 1

    return wt_A, wt_B, samples


# -----------------------------
# Identify mutations vs WT
# -----------------------------
def get_mutations(wt_seq, mut_seq, chain):
    """
    Compare mutant to WT, return list of (chain, resnum, aa3).
    Residue numbering is 1-based, matching PDB numbering directly.
    """
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if wt_aa != mut_aa:
            if mut_aa not in AA1_TO_AA3:
                raise ValueError(f"Unknown amino acid one-letter code: {mut_aa}")
            mutations.append((chain, i + 1, AA1_TO_AA3[mut_aa]))
    return mutations


# -----------------------------
# Apply mutations to pose
# -----------------------------
def apply_mutations(pose, mutations):
    """
    Apply mutations [(chain, resnum, aa3)] to pose using PDB chain+resnum.
    """
    for chain, resnum, aa3 in mutations:
        pose_res = pose.pdb_info().pdb2pose(chain, resnum)
        if pose_res == 0:
            raise ValueError(f"Residue {chain}{resnum} not found in pose.")
        MutateResidue(pose_res, aa3).apply(pose)


# -----------------------------
# Build readable mutation string
# -----------------------------
def mutation_label(mutations):
    """e.g. [(A, 27, LYS), (B, 54, VAL)] -> 'A27K_B54V'"""
    return '_'.join(
        f"{chain}{resnum}{AA3_TO_AA1[aa3]}"
        for chain, resnum, aa3 in mutations
    )


# -----------------------------
# Main
# -----------------------------
def main(
    fa_path,         # path to MPNN .fa output file
    abcd_pdb,        # path to ABCD complex pdb
    out_pdb_dir,     # directory to save mutated pdbs
    out_csv,         # path to save mutation summary csv
):
    os.makedirs(out_pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pyrosetta.init("-mute all")

    print(f"Parsing {fa_path}...")
    wt_A, wt_B, samples = parse_fa(fa_path)
    print(f"  WT chain A length : {len(wt_A)}")
    print(f"  WT chain B length : {len(wt_B)}")
    print(f"  Total samples     : {len(samples)}")

    # Group samples by unique mutation profile
    profile_to_samples   = defaultdict(list)
    profile_to_mutations = {}

    for s in samples:
        muts_A   = get_mutations(wt_A, s['seq_A'], 'A')
        muts_B   = get_mutations(wt_B, s['seq_B'], 'B')
        all_muts = muts_A + muts_B
        key      = mutation_label(all_muts) if all_muts else "WT"

        profile_to_samples[key].append(s['sample'])
        profile_to_mutations[key] = all_muts

    print(f"  Unique mutation profiles : {len(profile_to_mutations)}")
    wt_count = len(profile_to_samples.get("WT", []))
    print(f"  Samples recovering WT    : {wt_count}")

    # Apply mutations and save pdbs
    rows  = []
    saved = 0

    for mut_key, mutations in profile_to_mutations.items():
        if mut_key == "WT":
            print("  Skipping WT profile.")
            continue
        try:
            pose = pose_from_pdb(abcd_pdb)
            apply_mutations(pose, mutations)

            out_fname = f"{mut_key}.pdb"
            out_path  = os.path.join(out_pdb_dir, out_fname)
            pose.dump_pdb(out_path)
            saved += 1

            samples_str = ', '.join(map(str, profile_to_samples[mut_key]))
            print(f"  Saved: {out_fname}  (samples: {samples_str})")

            rows.append({
                'mutation_profile' : mut_key,
                'mutations'        : ', '.join(f"{c}{r}{AA3_TO_AA1[a]}" for c, r, a in mutations),
                'num_mutations'    : len(mutations),
                'sample_indices'   : samples_str,
                'pdb'              : out_fname,
            })

        except Exception as e:
            print(f"  ERROR on {mut_key}: {e}")

    # Save csv summary
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nDone. {saved} mutated pdb(s) saved to: {out_pdb_dir}")
    print(f"Mutation summary saved to: {out_csv}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main(
        fa_path     = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/scripts/r1/seqs/triple_renumbered.fa",
        abcd_pdb    = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/triple_renumbered.pdb",
        out_pdb_dir = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_inputs",
        out_csv     = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/mutation_summary.csv",
    )