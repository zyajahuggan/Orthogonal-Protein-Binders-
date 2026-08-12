"""
Runs IPSAE (ipsae.py) at PAE cutoff=10, distance cutoff=10 on every AF3, Boltz-2,
and ESMFold2 prediction, and adds the resulting interface-confidence scores as new
columns onto the 6 metrics CSVs that cross_model_analysis/compare_models.py treats
as canonical (one dhd + one cop file per model).

Each model stores its structure + PAE data in a different layout, so LOCATORS below
maps each model to a function that finds the right two files given a prediction's
output folder. ESMFold2's files were produced by
esmfold2/scripts/extract_metrics/convert_for_ipsae.py.

Every complex here is a 2-chain (A/B) heterodimer, so IPSAE always writes exactly one
"max" row (the larger of the A->B / B->A directions) -- that's the row we keep.

ipsae.py writes 3 small output files (.txt, _byres.txt, .pml) next to each structure
file as a side effect; those are left in place.
"""

import subprocess
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
IPSAE_SCRIPT = BASE / "IPSAE" / "ipsae.py"
PAE_CUTOFF = "10"
DIST_CUTOFF = "10"

# IPSAE output column -> new column name added to our metrics CSVs
IPSAE_COLUMN_MAP = {
    "ipSAE": "ipsae_score",
    "ipSAE_d0chn": "ipsae_score_d0chn",
    "ipSAE_d0dom": "ipsae_score_d0dom",
    "ipTM_af": "ipsae_iptm_af",
    "ipTM_d0chn": "ipsae_iptm_d0chn",
    "pDockQ": "ipsae_pdockq",
    "pDockQ2": "ipsae_pdockq2",
    "LIS": "ipsae_lis",
    "n0res": "ipsae_n0res",
    "n0chn": "ipsae_n0chn",
    "n0dom": "ipsae_n0dom",
    "d0res": "ipsae_d0res",
    "d0chn": "ipsae_d0chn",
    "d0dom": "ipsae_d0dom",
    "nres1": "ipsae_nres1",
    "nres2": "ipsae_nres2",
    "dist1": "ipsae_dist1",
    "dist2": "ipsae_dist2",
}


def ensure_real_cif_or_pdb(cif_path: Path) -> Path:
    """Some AF3 and Boltz-2 runs wrote plain fixed-column PDB text under a '.cif'
    name (no '_atom_site.' mmCIF header) instead of real mmCIF -- IPSAE picks its
    parser from the extension, so those need a sibling '.pdb' copy or they crash.
    Real mmCIF files are returned unchanged."""
    if not cif_path.exists():
        return cif_path
    with open(cif_path) as f:
        head = f.read(4000)
    if "_atom_site." in head:
        return cif_path
    pdb_path = cif_path.with_suffix(".pdb")
    if not pdb_path.exists():
        pdb_path.write_text(cif_path.read_text())
    return pdb_path


def reconstruct_af3_cif(cif_path: Path) -> Path:
    """Some AF3 top-level model.cif files were re-saved with hydrogens added and
    atoms renumbered (not real mmCIF -- no '_atom_site.' header). IPSAE's AF3 branch
    looks up per-atom pLDDT by indexing straight into confidences.json's flat
    atom_plddts array using each atom's serial number, so renumbered atoms point at
    the wrong pLDDT entries (or crash with an out-of-range index).

    Confirmed with PyMOL's h_add (the same call af3/scripts/extract_metrics/get_hbonds.py
    uses) that adding hydrogens never reorders or removes heavy atoms -- it only
    appends new H atoms after each residue's existing atoms. So stripping hydrogens
    and renumbering the remaining heavy atoms 1..N in file order exactly reconstructs
    the original AF3 atom order/count that atom_plddts was indexed against.

    Real (untouched) mmCIF files are returned unchanged."""
    if not cif_path.exists():
        return cif_path
    with open(cif_path) as f:
        head = f.read(4000)
    if "_atom_site." in head:
        return cif_path

    out_path = cif_path.with_name(cif_path.stem + "_reconstructed.cif")
    if out_path.exists():
        return out_path

    heavy_atoms = []
    with open(cif_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[76:78].strip() == "H":
                continue
            heavy_atoms.append(
                (line[12:16].strip(), line[17:20].strip(), line[21].strip(),
                 line[22:26].strip(), line[30:38].strip(), line[38:46].strip(), line[46:54].strip())
            )

    with open(out_path, "w") as f:
        f.write(
            "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
            "_atom_site.auth_asym_id\n"
        )
        for i, (atom_name, resname, chain_id, resnum, x, y, z) in enumerate(heavy_atoms, start=1):
            f.write(f"ATOM {i} {atom_name} {resname} {resnum} {x} {y} {z} {chain_id}\n")

    return out_path


def af3_files(pair_dir: Path, raw_id: str):
    cif_path = pair_dir / f"{raw_id}_model.cif"
    return pair_dir / f"{raw_id}_confidences.json", reconstruct_af3_cif(cif_path)


def boltz_files(pair_dir: Path, raw_id: str):
    pred_dir = pair_dir / f"boltz_results_{raw_id}" / "predictions" / raw_id
    cif_path = pred_dir / f"{raw_id}_model_0.cif"
    return pred_dir / f"pae_{raw_id}_model_0.npz", ensure_real_cif_or_pdb(cif_path)


def esmfold2_files(pair_dir: Path, raw_id: str):
    return pair_dir / f"pae_{raw_id}.npz", pair_dir / f"{raw_id}.pdb"


PROJECTS = [
    dict(
        label="af3/dhd",
        metrics_csv=BASE / "af3/metrics/dhd_metrics_af3_filtered.csv",
        id_col="sample",
        outputs_dir=BASE / "af3/outputs/dhd",
        locate=af3_files,
    ),
    dict(
        label="af3/cop",
        metrics_csv=BASE / "af3/metrics/cop_metrics_af3.csv",
        id_col="sample",
        outputs_dir=BASE / "af3/outputs/cross_docking",
        locate=af3_files,
    ),
    dict(
        label="boltz/dhd",
        metrics_csv=BASE / "boltz/metrics/boltz_metrics_dhd_filtered.csv",
        id_col="job_name",
        outputs_dir=BASE / "boltz/outputs/dhd",
        locate=boltz_files,
    ),
    dict(
        label="boltz/cop",
        metrics_csv=BASE / "boltz/metrics/boltz_metrics_cop.csv",
        id_col="job_name",
        outputs_dir=BASE / "boltz/outputs/cross_docking",
        locate=boltz_files,
    ),
    dict(
        label="esmfold2/dhd",
        metrics_csv=BASE / "esmfold2/metrics/dhd_combined_metrics_filtered.csv",
        id_col="job_name",
        outputs_dir=BASE / "esmfold2/outputs/dhd",
        locate=esmfold2_files,
    ),
    dict(
        label="esmfold2/cop",
        metrics_csv=BASE / "esmfold2/metrics/cop_combined_metrics.csv",
        id_col="job_name",
        outputs_dir=BASE / "esmfold2/outputs/cross_docking",
        locate=esmfold2_files,
    ),
]


def run_ipsae(pae_path: Path, struct_path: Path):
    """Run ipsae.py on one prediction and return its 'max'-type score row as a dict,
    or None if inputs are missing or the run failed."""
    if not pae_path.exists() or not struct_path.exists():
        return None

    result = subprocess.run(
        ["python3", str(IPSAE_SCRIPT), str(pae_path), str(struct_path), PAE_CUTOFF, DIST_CUTOFF],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    out_txt = struct_path.with_name(f"{struct_path.stem}_{int(PAE_CUTOFF):02d}_{int(DIST_CUTOFF):02d}.txt")
    if not out_txt.exists():
        return None

    with open(out_txt) as f:
        rows = [line.split() for line in f if line.strip()]
    header, data_rows = rows[0], rows[1:]

    for row in data_rows:
        record = dict(zip(header, row))
        if record.get("Type") == "max":
            return record
    return None


def process_project(label, metrics_csv, id_col, outputs_dir, locate):
    df = pd.read_csv(metrics_csv)
    new_columns = {new_col: [] for new_col in IPSAE_COLUMN_MAP.values()}
    n_ok, n_fail = 0, 0

    for raw_id in df[id_col]:
        pae_path, struct_path = locate(outputs_dir / raw_id, raw_id)
        record = run_ipsae(pae_path, struct_path)

        if record is None:
            n_fail += 1
            for new_col in new_columns:
                new_columns[new_col].append(None)
            continue

        n_ok += 1
        for orig_col, new_col in IPSAE_COLUMN_MAP.items():
            new_columns[new_col].append(float(record[orig_col]))

    for new_col, values in new_columns.items():
        df[new_col] = values

    df.to_csv(metrics_csv, index=False)
    print(f"{label}: {n_ok} ok, {n_fail} failed -> {metrics_csv}")


def main():
    for project in PROJECTS:
        process_project(**project)


if __name__ == "__main__":
    main()
