#! /usr/bin/env python3
"""
Scores the AB_C interface for all reference replicate pdbs in a folder,
averages across replicates, and saves results to a csv.
The avg_dG_interface value can then be used as the reference ddG baseline
for subsequent mutant scoring rounds.
"""
import os
import re
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover
from multiprocessing import Pool


# -----------------------------
# Worker initializer
# -----------------------------
def init_worker():
    pyrosetta.init("-mute all")


# -----------------------------
# Core scoring routine — AB_C interface
# -----------------------------
def calculate_interface_score(pdb_path):
    pose = pose_from_pdb(pdb_path)
    ClearConstraintsMover().apply(pose)

    scorefxn = create_score_function("ref2015")

    iam = InterfaceAnalyzerMover("AB_C")
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.apply(pose)

    return iam.get_interface_dG()


# -----------------------------
# Score one pdb
# -----------------------------
def score_one(pdb_path):
    try:
        dG    = calculate_interface_score(pdb_path)
        fname = os.path.basename(pdb_path)
        print(f"  Scored {fname}: dG_interface = {dG:.3f}")
        return {"description": fname, "dG_interface": dG}
    except Exception as e:
        print(f"  Skipping {pdb_path}: {e}")
        return None


# -----------------------------
# Main
# -----------------------------
def main(
    ref_pdb_dir,   # folder containing reference replicate pdbs
    out_csv,       # output csv path
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pyrosetta.init("-mute all", silent=True)

    pdb_files = [
        os.path.join(ref_pdb_dir, f)
        for f in os.listdir(ref_pdb_dir)
        if f.endswith(".pdb")
    ]
    print(f"Found {len(pdb_files)} reference replicate(s) to score.")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Using {n_workers} worker(s)")

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        raw_results = pool.map(score_one, pdb_files)

    records = [r for r in raw_results if r is not None]
    df = pd.DataFrame(records)

    avg_dG = df["dG_interface"].mean()
    std_dG = df["dG_interface"].std()

    print(f"\nReference AB_C interface dG:")
    print(f"  Mean : {avg_dG:.3f}")
    print(f"  Std  : {std_dG:.3f}")
    print(f"  N    : {len(df)}")

    # Save per-replicate scores + summary row
    summary = pd.DataFrame([{
        "description":  "AVERAGE",
        "dG_interface": avg_dG,
    }])
    out_df = pd.concat([df, summary], ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")
    print(f"\n*** Use avg_dG_interface = {avg_dG:.3f} as your reference for mutant ddG calculations ***")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main(
        ref_pdb_dir = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_outputs",
        out_csv     = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_outputs/scores/reference_scores.csv",
    )