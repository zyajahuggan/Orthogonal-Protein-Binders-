#! /usr/bin/env python3
import os
import numpy as np
import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from multiprocessing import Pool
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover


# -----------------------------
# Worker initializer
# -----------------------------
def init_worker():
    """Initialize PyRosetta inside each worker process."""
    pyrosetta.init("-mute all")

# -----------------------------
# Core scoring routine
# -----------------------------
def calculate_interaction_metrics(pdb_path, monomer_chain):
    pose = pose_from_pdb(pdb_path)
    ClearConstraintsMover().apply(pose)
    # collect chain IDs
    chains = sorted(set(pose.pdb_info().chain(i) for i in range(1, pose.size() + 1)))
    if monomer_chain not in chains:
        raise ValueError(f"Chain {monomer_chain} not found in {pdb_path}")

    other_chains = ''.join(c for c in chains if c != monomer_chain)
    interface = f"{monomer_chain}_{other_chains}"

    scorefxn = create_score_function("ref2015")

    # --- Interface analysis ---
    iam = InterfaceAnalyzerMover(interface)
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.set_interface_jump(2) 
    iam.apply(pose)

    dG_interface = iam.get_interface_dG()
    dG_complex   = iam.get_complex_energy()

    # --- Split by chain ---
    chain_poses = pose.split_by_chain()
    receptor_pose = None
    ligand_pose   = None

    for p in chain_poses:
        chain_id = p.pdb_info().chain(1)  # first residue’s chain ID
        if chain_id == monomer_chain:
            receptor_pose = p
        else:
            if ligand_pose is None:
                ligand_pose = p
            else:
                ligand_pose.append_pose_by_jump(p, ligand_pose.num_jump() + 1)

    dG_receptor = scorefxn(receptor_pose)
    dG_ligand   = scorefxn(ligand_pose)

    return dG_interface, dG_complex, dG_receptor, dG_ligand

# -----------------------------
# One job = one mutant PDB
# -----------------------------
def analyze_one(args):
    (pdb_path, wt_pdb, monomer_chain,
     dg_wt_interface, dg_wt_complex,
     dg_wt_receptor, dg_wt_ligand) = args
    try:
        dg_mut_interface, dg_mut_complex, dg_mut_receptor, dg_mut_ligand = calculate_interaction_metrics(
            pdb_path, monomer_chain
        )
        ddg_interface = dg_mut_interface - dg_wt_interface
        ddg_complex   = dg_mut_complex - dg_wt_complex
        ddg_receptor  = dg_mut_receptor - dg_wt_receptor
        ddg_ligand    = dg_mut_ligand - dg_wt_ligand

        return (pdb_path,
                dg_mut_interface, ddg_interface,
                dg_mut_complex, ddg_complex,
                dg_mut_receptor, ddg_receptor,
                dg_mut_ligand, ddg_ligand)
    except Exception as e:
        print(f"Skipping {pdb_path}: {e}")
        return (pdb_path,) + (None,) * 8

# -----------------------------
# Main 
# -----------------------------
def main(pdb_dir, wt_pdb, monomer_chain, out_sc):
    # initialize PyRosetta in the main process
    pyrosetta.init("-mute all", silent=True)

    # compute WT energies once (outside pool)
    dg_wt_interface, dg_wt_complex, dg_wt_receptor, dg_wt_ligand = calculate_interaction_metrics(wt_pdb, monomer_chain)
    print(f"WT Interface ΔG = {dg_wt_interface:.3f}, WT Complex = {dg_wt_complex:.3f}, "
          f"WT Receptor = {dg_wt_receptor:.3f}, WT Ligand = {dg_wt_ligand:.3f}")

    # collect mutant jobs
    
    pdb_files = [
        os.path.join(pdb_dir, f)
        for f in os.listdir(pdb_dir)
        if f.endswith(".pdb") and os.path.abspath(os.path.join(pdb_dir, f)) != os.path.abspath(wt_pdb)
    ]
    jobs = [
        (pdb, wt_pdb, monomer_chain,
         dg_wt_interface, dg_wt_complex,
         dg_wt_receptor, dg_wt_ligand)
        for pdb in pdb_files
    ]

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Starting pool with {n_workers} worker(s)")

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        results = pool.map(analyze_one, jobs)

    # write Rosetta-style scorefile
    with open(out_sc, "w") as f:
        f.write("SCORE: dG_interface ddG_interface dG_complex ddG_complex dG_receptor ddG_receptor dG_ligand ddG_ligand description\n")
        # WT row
        f.write(f"SCORE: {dg_wt_interface:.3f} 0.000 "
                f"{dg_wt_complex:.3f} 0.000 "
                f"{dg_wt_receptor:.3f} 0.000 "
                f"{dg_wt_ligand:.3f} 0.000 "
                f"{os.path.basename(wt_pdb)}\n")
        for (pdb_path,
             dg_i, ddg_i,
             dg_c, ddg_c,
             dg_r, ddg_r,
             dg_l, ddg_l) in results:
            if dg_i is not None:
                f.write(
                    f"SCORE: {dg_i:.3f} {ddg_i:.3f} "
                    f"{dg_c:.3f} {ddg_c:.3f} "
                    f"{dg_r:.3f} {ddg_r:.3f} "
                    f"{dg_l:.3f} {ddg_l:.3f} "
                    f"{os.path.basename(pdb_path)}\n"
                )

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main(
        pdb_dir="/scratch/jgray21/zyhuggan/tryingggg/pos_neg_design/OrthoMPNN/Rosetta/output_pdbs/enrich_wt",
        wt_pdb="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/Rosetta/Ligand_Analysis/input/5repeats_5_wt/relaxed_wt_1.pdb",
        monomer_chain="Y",
        out_sc="/scratch/jgray21/zyhuggan/tryingggg/pos_neg_design/OrthoMPNN/Rosetta/score_files/enrich_wt.sc"
    )
