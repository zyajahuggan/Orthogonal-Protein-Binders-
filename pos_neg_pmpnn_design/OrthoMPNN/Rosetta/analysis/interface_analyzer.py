#! /usr/bin/env python3
"""
Scores the AB_C interface for all pdbs in a folder,
computing ddG relative to a reference (wt) pdb.
Outputs a Rosetta-style scorefile.
"""
import os
import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from multiprocessing import Pool
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover


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

    dG_interface = iam.get_interface_dG()
    dG_complex   = iam.get_complex_energy()

    # Split by chain to get individual energies
    chain_poses   = pose.split_by_chain()
    ab_pose       = None
    c_pose        = None

    for p in chain_poses:
        chain_id = p.pdb_info().chain(1)
        if chain_id in ('A', 'B'):
            if ab_pose is None:
                ab_pose = p
            else:
                ab_pose.append_pose_by_jump(p, ab_pose.num_jump() + 1)
        elif chain_id == 'C':
            c_pose = p

    dG_ab = scorefxn(ab_pose) if ab_pose else None
    dG_c  = scorefxn(c_pose)  if c_pose  else None

    return dG_interface, dG_complex, dG_ab, dG_c


# -----------------------------
# One job = one PDB
# -----------------------------
def analyze_one(args):
    pdb_path, ref_dG_interface, ref_dG_complex, ref_dG_ab, ref_dG_c = args
    try:
        dG_interface, dG_complex, dG_ab, dG_c = calculate_interface_score(pdb_path)

        ddG_interface = dG_interface - ref_dG_interface
        ddG_complex   = dG_complex   - ref_dG_complex
        ddG_ab        = dG_ab        - ref_dG_ab
        ddG_c         = dG_c         - ref_dG_c

        return (pdb_path,
                dG_interface, ddG_interface,
                dG_complex,   ddG_complex,
                dG_ab,        ddG_ab,
                dG_c,         ddG_c)
    except Exception as e:
        print(f"Skipping {pdb_path}: {e}")
        return (pdb_path,) + (None,) * 8


# -----------------------------
# Main
# -----------------------------
def main(pdb_dir, wt_pdb, out_sc):
    pyrosetta.init("-mute all", silent=True)

    # Score reference once in main process
    print("Scoring reference pdb...")
    ref_dG_interface, ref_dG_complex, ref_dG_ab, ref_dG_c = calculate_interface_score(wt_pdb)
    print(f"  Reference dG_interface (AB_C) = {ref_dG_interface:.3f}")
    print(f"  Reference dG_complex          = {ref_dG_complex:.3f}")
    print(f"  Reference dG_AB               = {ref_dG_ab:.3f}")
    print(f"  Reference dG_C                = {ref_dG_c:.3f}")

    # Collect pdbs excluding the reference
    pdb_files = [
        os.path.join(pdb_dir, f)
        for f in os.listdir(pdb_dir)
        if f.endswith(".pdb")
        and os.path.abspath(os.path.join(pdb_dir, f)) != os.path.abspath(wt_pdb)
    ]
    print(f"Found {len(pdb_files)} pdb(s) to score.")

    jobs = [
        (pdb, ref_dG_interface, ref_dG_complex, ref_dG_ab, ref_dG_c)
        for pdb in pdb_files
    ]

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Using {n_workers} worker(s)")

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        results = pool.map(analyze_one, jobs)

    # Write scorefile
    os.makedirs(os.path.dirname(out_sc), exist_ok=True)
    with open(out_sc, "w") as f:
        f.write("SCORE: dG_interface ddG_interface dG_complex ddG_complex dG_AB ddG_AB dG_C ddG_C description\n")

        # Reference row
        f.write(f"SCORE: {ref_dG_interface:.3f} 0.000 "
                f"{ref_dG_complex:.3f} 0.000 "
                f"{ref_dG_ab:.3f} 0.000 "
                f"{ref_dG_c:.3f} 0.000 "
                f"{os.path.basename(wt_pdb)}\n")

        for (pdb_path,
             dG_i, ddG_i,
             dG_c, ddG_c,
             dG_ab, ddG_ab,
             dG_c2, ddG_c2) in results:
            if dG_i is not None:
                f.write(
                    f"SCORE: {dG_i:.3f} {ddG_i:.3f} "
                    f"{dG_c:.3f} {ddG_c:.3f} "
                    f"{dG_ab:.3f} {ddG_ab:.3f} "
                    f"{dG_c2:.3f} {ddG_c2:.3f} "
                    f"{os.path.basename(pdb_path)}\n"
                )

    print(f"\nDone. Scorefile written to: {out_sc}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main(
        pdb_dir = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/output",
        wt_pdb  = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/abcd_complex.pdb",
        out_sc  = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/scores/abcd_reference.sc",
    )