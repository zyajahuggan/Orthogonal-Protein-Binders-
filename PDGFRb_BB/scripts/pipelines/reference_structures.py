#! /usr/bin/env python3
import os
import sys
import time
import logging
from multiprocessing import Pool

import pyrosetta
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

# ---------------- Logging ----------------
os.environ['TZ'] = 'America/New_York'
time.tzset()
logger = logging.getLogger()

def configure_logging(level=logging.DEBUG):
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(processName)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    ))
    root.addHandler(handler)
    logging.captureWarnings(True)
    for noisy in ("pyrosetta", "jax"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------- Globals in workers ----------------
GLOBAL_SCOREFXN = None
GLOBAL_WT_POSE = None
GLOBAL_ORTHO_POSE = None
GLOBAL_FRELAX = None
GLOBAL_IAM = None


def init_worker(wt_pdb, ortho_pdb, xml_file="./reference_structures.xml"):
    """Initialize PyRosetta and load common poses once per worker."""
    pyrosetta.init("-mute all", silent=True)
    global GLOBAL_SCOREFXN, GLOBAL_WT_POSE, GLOBAL_ORTHO_POSE, GLOBAL_FRELAX, GLOBAL_IAM

    GLOBAL_SCOREFXN = pyrosetta.rosetta.core.scoring.get_score_function(True)

    # Load poses once
    GLOBAL_WT_POSE = pyrosetta.io.pose_from_pdb(wt_pdb)
    GLOBAL_ORTHO_POSE = pyrosetta.io.pose_from_pdb(ortho_pdb)

    # Load relax mover once
    xmlobj = XmlObjects.create_from_file(xml_file)
    GLOBAL_FRELAX = xmlobj.get_mover("FastRelax")
    GLOBAL_IAM = xmlobj.get_mover("analyze_interface")


    logger.debug("Worker initialized with WT + Ortho poses")

# ---------------- Job runner ----------------
def run_relax_job(job):
    pdb_type, out_path = job  # "wt" or "ortho"
    logger.debug("Cloning poses")
    if pdb_type == "wt":
        pose = GLOBAL_WT_POSE.clone()
    elif pdb_type == "ortho":
        pose = GLOBAL_ORTHO_POSE.clone()
    else:
        raise ValueError(f"Unknown job type: {pdb_type}")
    logger.debug("Applying Relax...")
    GLOBAL_FRELAX.apply(pose)

    logger.debug("Applying InterfaceAnalyzer...")
    GLOBAL_IAM.apply(pose)

    logger.debug("Done...Dumping Now")
    pose.dump_pdb(out_path)
    return out_path

# ---------------- Helpers ----------------
def get_max_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# ---------------- Main ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt_pdb", type=str, required=False,
                        default='/scratch/jgray21/zyhuggan/Ortho_BB/structures/3mjgwt_ABY.pdb')
    parser.add_argument("--ortho_pdb", type=str, required=False,
                        default='/scratch/jgray21/zyhuggan/Ortho_BB/structures/3mjgortho_ABY.pdb')
    parser.add_argument("--out_wt", type=str, required=False,
                        default='/scratch/jgray21/zyhuggan/Ortho_BB/structures/5repeats_5_wt',
                        help="Output dir for WT")
    parser.add_argument("--out_ortho", type=str, required=False,
                        default='/scratch/jgray21/zyhuggan/Ortho_BB/structures/5repeats_5_ortho',
                        help="Output dir for Ortho")
    parser.add_argument("--nstruct", type=int, default=5,
                        help="Number of relax runs per structure")
    parser.add_argument("--workers", type=int, default=get_max_workers(),
                        help="Number of processes (default = all allocated CPUs)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    # Ensure output dirs exist
    os.makedirs(args.out_wt, exist_ok=True)
    os.makedirs(args.out_ortho, exist_ok=True)

    # Build job list
    jobs = []
    for i in range(1, args.nstruct + 1):
        jobs.append(("wt", os.path.join(args.out_wt, f"relaxed_wt_{i}.pdb")))
        jobs.append(("ortho", os.path.join(args.out_ortho, f"relaxed_ortho_{i}.pdb")))

    # Run in parallel
    with Pool(processes=args.workers,
              initializer=init_worker,
              initargs=(args.wt_pdb, args.ortho_pdb)) as pool:
        for out in pool.imap_unordered(run_relax_job, jobs):
            logger.info(f"Finished {out}")

if __name__ == "__main__":
    main()
    logger.info("Script execution completed")
