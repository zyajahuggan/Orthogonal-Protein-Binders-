#! /usr/bin/env python3
import os
import argparse
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
# Core scoring routine
# -----------------------------
def calculate_interaction_metrics(pdb_path, interface):
    pose = pose_from_pdb(pdb_path)
    ClearConstraintsMover().apply(pose)

    # validate chains
    chains = sorted(set(pose.pdb_info().chain(i) for i in range(1, pose.size() + 1)))
    group1, group2 = interface.split("_")

    if not set(group1 + group2).issubset(set(chains)):
        raise ValueError(f"Interface {interface} chains not found in {pdb_path}")

    scorefxn = create_score_function("ref2015")

    # --- Interface analysis ---
    iam = InterfaceAnalyzerMover(interface)
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.apply(pose)

    dG_interface = iam.get_interface_dG()
    dG_complex   = iam.get_complex_energy()

    # --- Monomer energies ---
    chain_poses = pose.split_by_chain()

    receptor_pose = None
    ligand_pose = None

    for p in chain_poses:
        chain_id = p.pdb_info().chain(1)

        if chain_id in group1:
            if receptor_pose is None:
                receptor_pose = p
            else:
                receptor_pose.append_pose_by_jump(p, receptor_pose.num_jump() + 1)

        elif chain_id in group2:
            if ligand_pose is None:
                ligand_pose = p
            else:
                ligand_pose.append_pose_by_jump(p, ligand_pose.num_jump() + 1)

    if receptor_pose is None or ligand_pose is None:
        raise ValueError(f"Failed to build receptor/ligand for {pdb_path}")

    dG_receptor = scorefxn(receptor_pose)
    dG_ligand   = scorefxn(ligand_pose)

    return dG_interface, dG_complex, dG_receptor, dG_ligand


# -----------------------------
# One job = one PDB
# -----------------------------
def analyze_one(args):
    pdb_path, interface = args
    try:
        dG_i, dG_c, dG_r, dG_l = calculate_interaction_metrics(pdb_path, interface)
        return (pdb_path, dG_i, dG_c, dG_r, dG_l)
    except Exception as e:
        print(f"Skipping {pdb_path}: {e}")
        return (pdb_path, None, None, None, None)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score interface energies for PDB structures (A_B style interfaces)."
    )
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--pdbs", type=str, nargs="+", default=None)
    parser.add_argument("--interface", type=str, required=True, help="e.g. A_B")
    parser.add_argument("--out_sc", type=str, default="scores.sc")
    parser.add_argument("--nproc", type=int, default=None)

    args = parser.parse_args()

    # collect PDBs
    if args.pdbs:
        pdb_files = args.pdbs
    elif args.pdb_dir:
        pdb_files = sorted(
            os.path.join(args.pdb_dir, f)
            for f in os.listdir(args.pdb_dir)
            if f.endswith(".pdb")
        )
    else:
        raise ValueError("Provide --pdb_dir or --pdbs")

    if not pdb_files:
        raise ValueError("No PDB files found")

    print(f"Scoring {len(pdb_files)} structure(s)...")

    jobs = [(pdb, args.interface) for pdb in pdb_files]

    n_workers = args.nproc or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Using {n_workers} workers")

    # IMPORTANT: init only in workers
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        results = pool.map(analyze_one, jobs)

    # write scorefile
    with open(args.out_sc, "w") as f:
        f.write("SCORE: dG_interface dG_complex dG_receptor dG_ligand description\n")

        for (pdb_path, dG_i, dG_c, dG_r, dG_l) in results:
            name = os.path.basename(pdb_path)

            if dG_i is not None:
                f.write(f"SCORE: {dG_i:.3f} {dG_c:.3f} {dG_r:.3f} {dG_l:.3f} {name}\n")
            else:
                f.write(f"SCORE: FAILED FAILED FAILED FAILED {name}\n")

    print(f"Done → {args.out_sc}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()