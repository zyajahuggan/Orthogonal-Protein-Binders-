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
GLOBAL_FRELAX = None
GLOBAL_IAM = None

def init_worker(xml_file):
    """Initialize PyRosetta and load movers once per worker."""
    pyrosetta.init("-mute all", silent=True)

    global GLOBAL_FRELAX, GLOBAL_IAM
    xmlobj = XmlObjects.create_from_file(xml_file)
    GLOBAL_FRELAX = xmlobj.get_mover("FastRelax")
    # if you don’t actually want InterfaceAnalyzer, comment next line
    GLOBAL_IAM = xmlobj.get_mover("analyze_interface")

    logger.debug("Worker initialized")

# ---------------- Job runner ----------------
def run_relax_job(job):
    pdb_in, pdb_out = job
    logger.info(f"Running Relax on {pdb_in}")

    pose = pyrosetta.io.pose_from_pdb(pdb_in)

    # Apply FastRelax
    GLOBAL_FRELAX.apply(pose)

    # Optional: InterfaceAnalyzerMover
    if GLOBAL_IAM is not None:
        GLOBAL_IAM.apply(pose)

    pose.dump_pdb(pdb_out)
    return pdb_out

# ---------------- Helpers ----------------
def get_max_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# ---------------- Main ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        default="/scratch4/jgray21/zhuggan1/projects/orthosystems/Receptor_Analysis/fullpdgf",
        help="Root folder containing Design_1, Design_2, ..."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/scratch4/jgray21/zhuggan1/projects/orthosystems/Receptor_Analysis/fullpdgf",
        help="Root folder where relaxed structures go"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="/scratch4/jgray21/zhuggan1/projects/orthosystems/Receptor_Analysis/pipelines/reference_structures.xml",
        help="RosettaScripts XML file with FastRelax & analyze_interface"
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=1,   # <-- only once per PDB
        help="How many relax repeats per PDB"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=get_max_workers()
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    # Make sure output_root exists
    os.makedirs(args.output_root, exist_ok=True)

    # ---- Build job list by scanning the structure tree ----
    jobs = []

    for root, dirs, files in os.walk(args.input_root):
        for f in files:
            if not f.endswith(".pdb"):
                continue

            full_in = os.path.join(root, f)

            # Mirror folder structure under output_root
            relative = os.path.relpath(root, args.input_root)
            out_dir = os.path.join(args.output_root, relative)
            os.makedirs(out_dir, exist_ok=True)

            for i in range(1, args.nstruct + 1):
                out_name = f"{os.path.splitext(f)[0]}_relaxed_{i:03d}.pdb"
                full_out = os.path.join(out_dir, out_name)
                jobs.append((full_in, full_out))

    logger.info(f"Total jobs: {len(jobs)}")

    # ---- Run jobs in parallel ----
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.xml,)
    ) as pool:
        for out_file in pool.imap_unordered(run_relax_job, jobs):
            logger.info(f"Finished {out_file}")

if __name__ == "__main__":
    main()
    logger.info("Script execution completed")
