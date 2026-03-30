#! /usr/bin/env python3

import os
import time
import sys

import pyrosetta
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

import logging
os.environ['TZ'] = 'America/New_York'
time.tzset()
logger = logging.getLogger(__name__)

GLOBAL_SCOREFXN = None

def init_worker():
    """
    Per-process initializer for multiprocessing workers:
    - Initialize PyRosetta silently (only once per process)s
    - Set up GLOBAL_SCOREFXN for use in BackMutator
    """
    # Mute output and initialize
    pyrosetta.init("-mute all", silent=True)
    # pyrosetta.init()
    # Assign the global score function
    global GLOBAL_SCOREFXN
    GLOBAL_SCOREFXN = pyrosetta.rosetta.core.scoring.get_score_function(True)

def configure_logging(level=logging.DEBUG):
    """Configure root logger so child loggers from imported modules propagate here.
    Removes any pre-existing handlers to avoid duplicate logs across re-runs.
    """
    root = logging.getLogger()  # root logger
    root.setLevel(level)

    # Remove existing handlers to avoid duplication when reloading
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(processName)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    ))
    root.addHandler(handler)

    # Capture warnings emitted via the warnings module
    logging.captureWarnings(True)

    # Silence chatty libraries
    for noisy in ("pyrosetta", "jax"): # 
        logging.getLogger(noisy).setLevel(logging.WARNING)

class Mutator:

    def __init__(self,
                 pdb_path: str,
                 mutation_bundle,
                 scorefxn = None):
        self.pdb_path        = pdb_path
        self.mutation_bundle = mutation_bundle
        self.pose            = pyrosetta.io.pose_from_pdb(pdb_path)
        # ScoreFunction will be fetched at runtime inside workers
        self._scorefxn_param = scorefxn  
        self.xmlobj = None
        self.frel_xml_obj = None
        self.int_analyzer = None
        self.xml = ('./mutant_relax.xml')

    def apply_mutation(self):
        sf = self._scorefxn_param if self._scorefxn_param is not None else GLOBAL_SCOREFXN

        # Work on a single cloned pose
        new_pose = self.pose.clone()

        for pdb_pos, aa in self.mutation_bundle:
            pdb_chain, pdb_idx = pdb_pos.split()
            pdb_idx = int(pdb_idx)

            pose2num = new_pose.pdb_info().pdb2pose(pdb_chain, pdb_idx)
            if pose2num == 0:
                raise ValueError(f"Residue {pdb_chain} {pdb_idx} not found in pose")

            wt = new_pose.residue(pose2num).name1()
            if wt == aa:
                logger.debug(f"Skipping WT->WT mutation at {pdb_chain}{pdb_idx}")
                continue

            logger.debug(
                f"Mutating PDB idx {pdb_idx} at chain {pdb_chain} "
                f"(Rosetta {pose2num}) {wt}->{aa}"
            )

            mutate_residue(
                new_pose,
                pose2num,
                aa,
                pack_radius   = 5.0,
                pack_scorefxn = sf,
            )

        # Update pose once after all mutations
        self.pose = new_pose

    def self_setup_xmlobj(self):
        self.xmlobj = XmlObjects.create_from_file(self.xml)

        return self.xmlobj

    def setup_frelax(self):
        """Set up and return the FastRelax XML object for structure refinement."""
        self.xmlobj = XmlObjects.create_from_file(self.xml) or self.self_setup_xmlobj()
        self.frel_xml_obj = self.xmlobj.get_mover('FastRelax')
        logger.debug("Successfully setup the FastRelax")
        logger.debug(f"This is the number of repeats {self.frel_xml_obj.default_repeats()}")

        return self.frel_xml_obj
    
    def setup_analyzer(self):
        self.xmlobj = XmlObjects.create_from_file(self.xml) or self.self_setup_xmlobj()
        self.int_analyzer = self.xmlobj.get_mover('analyze_interface')        
        logger.debug("Successfully setup the InterfaceAnalyzerMover")

        return self.int_analyzer
    
    def mutate_relax_analyze(self, out_path: str):
        self.apply_mutation()
        mover = self.frel_xml_obj or self.setup_frelax()
        logger.debug(f"Applying fastrelax to input: {os.path.basename(self.pose.pdb_info().name())} -> {os.path.basename(out_path)}")
        mover.apply(self.pose)
        inta = self.int_analyzer or self.setup_analyzer()
        logger.debug(f"Applying the interface analyzer to input: {os.path.basename(self.pose.pdb_info().name())} -> {os.path.basename(out_path)}")
        inta.apply(self.pose)
        final_remarks = ["UNSAT HBOND PYMOL SELECTION", str(inta.get_pymol_sel_hbond_unsat()), 
                         "INTERFACE PYMOL SELECTION", str(inta.get_pymol_sel_interface())]
        remarks_str = '\n'.join(final_remarks)
        with open(out_path, 'a') as f:
            f.write(f"\n{remarks_str}")
        self.pose.dump_pdb(out_path)
        logger.info(f"Output: {os.path.basename(out_path)}")

def safe_runner(job_tuple):
    input_pdb_path, mutation_bundle, out_pdb_path = job_tuple
    try:
        m = Mutator(input_pdb_path, mutation_bundle)
        m.mutate_relax_analyze(out_pdb_path)
    except Exception as e:
        logger.error(f"Failed trying to generate {out_pdb_path} with error: {e}")


def job_list_creator(args):
    # --- Use this directly ---
    mutation_bundle = [
        ("A 27", "G"),
        ("A 31", "R"),
        ("A 34", "R"),
        ("A 36", "I"),
        ("A 73", "F"),
        ("B 54", "K"),
        ("B 57", "T"),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = []

    base = os.path.basename(args.in_pdb)
    name_root = os.path.splitext(base)[0]

    # Treat it as a single bundle inside a list
    bundles = [mutation_bundle]

    for bundle in bundles:
        print(f"[DEBUG] Creating jobs for bundle: {bundle}")

        mut_tag = "_".join([
            f"{pos.replace(' ','')}{aa}" for pos, aa in bundle
        ])

        for i in range(1, args.nstruct + 1):
            out_fname = f"experi_{name_root}_{mut_tag}_rep{i}.pdb"
            out_pdb = os.path.join(args.out_dir, out_fname)

            print(f"[DEBUG] Output file: {out_pdb}")
            jobs.append((args.in_pdb, bundle, out_pdb))

    return jobs

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pdb", type=str,
                        default='/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/Rosetta/Ligand_Analysis/input/5repeats_5_wt/relaxed_wt_1.pdb') #average pdb 
    parser.add_argument("--out_dir", type=str, default="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/Experimental_Structures/outputs/wt" ,help="Output directory.")
    parser.add_argument("--nstruct", type=int, default=5,help="Number of structures")
    parser.add_argument("--debug", action="store_true", help="Show full traceback on error (for debugging)")
    args = parser.parse_args()
    if args.debug:
        configure_logging(logging.DEBUG)
    else:
        configure_logging(logging.INFO)

    return args

def main():

    from multiprocessing import Pool

    global ARGS
    ARGS = parse_args()

    job_list = job_list_creator(ARGS)
    logger.info(f"Processing a total of {len(job_list)} jobs")
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    logger.info(f"Starting pool with {n_workers} worker(s)")

    # job_tuple = ('./relaxed_ortho.pdb', 'A 23', 'G', './test.pdb')
    with Pool(processes=n_workers, initializer=init_worker) as pool: # maxtasksperchild=10
        pool.map(safe_runner, job_list)

if __name__ == "__main__":
    main()
    logger.info("Script execution completed")
