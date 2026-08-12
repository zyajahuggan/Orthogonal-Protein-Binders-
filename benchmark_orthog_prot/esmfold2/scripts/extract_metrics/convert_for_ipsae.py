"""
Convert ESMFold2 outputs into the file layout IPSAE (ipsae.py) expects for
its "Boltz-style" input branch: <stem>.pdb + pae_<stem>.npz + plddt_<stem>.npz

Why this is needed:
- ESMFold2's "*.cif" files are actually plain fixed-column PDB text, not real
  mmCIF (no "_atom_site." header block). IPSAE picks its parser purely from
  the file extension, so a ".cif" file gets routed to the mmCIF parser and
  fails. Renaming/copying the content to ".pdb" routes it to IPSAE's PDB
  parser instead, which matches the actual format.
- ESMFold2 saves the full per-residue PAE matrix as a bare "*_pae.npy" array.
  IPSAE wants it inside an ".npz" file under the key "pae".
- IPSAE also wants a per-residue pLDDT array in a "plddt_<stem>.npz" file.
  ESMFold2 doesn't write one separately, but it stores plDDT (0-100 scale)
  in the B-factor column of every atom, repeated for all atoms in a residue.
  We pull one value per residue from each CA atom's B-factor column.

This script only reads the existing .cif/.npy files and writes new files
alongside them -- nothing already there is modified or deleted.
"""

import argparse
from pathlib import Path

import numpy as np

# Standard fixed-column PDB ATOM record layout (0-indexed, end-exclusive)
ATOM_NAME_COL = slice(12, 16)
BFACTOR_COL = slice(60, 66)


def extract_ca_bfactors(pdb_text_path: Path) -> np.ndarray:
    """Return one pLDDT value per residue, in file order, from CA-atom B-factors."""
    bfactors = []
    with open(pdb_text_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[ATOM_NAME_COL].strip()
            if atom_name == "CA":
                bfactors.append(float(line[BFACTOR_COL]))
    return np.array(bfactors, dtype=np.float32)


def convert_one(pred_dir: Path, force: bool) -> bool:
    """Convert a single ESMFold2 prediction folder. Returns True if converted."""
    stem = pred_dir.name
    cif_path = pred_dir / f"{stem}.cif"
    pae_npy_path = pred_dir / f"{stem}_pae.npy"

    if not cif_path.exists() or not pae_npy_path.exists():
        print(f"  skip {stem}: missing .cif or _pae.npy")
        return False

    pdb_out = pred_dir / f"{stem}.pdb"
    pae_out = pred_dir / f"pae_{stem}.npz"
    plddt_out = pred_dir / f"plddt_{stem}.npz"

    if not force and pdb_out.exists() and pae_out.exists() and plddt_out.exists():
        return False

    pdb_out.write_text(cif_path.read_text())

    pae_matrix = np.load(pae_npy_path)
    np.savez(pae_out, pae=pae_matrix)

    plddt = extract_ca_bfactors(cif_path)
    np.savez(plddt_out, plddt=plddt)

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs",
        help="esmfold2/outputs directory containing dhd/ and cross_docking/ subfolders",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-convert folders even if outputs already exist"
    )
    args = parser.parse_args()

    pred_dirs = sorted(p for p in args.root.glob("*/*") if p.is_dir())
    print(f"Found {len(pred_dirs)} prediction folders under {args.root}")

    converted = 0
    for pred_dir in pred_dirs:
        if convert_one(pred_dir, args.force):
            converted += 1

    print(f"Converted {converted} folders (skipped {len(pred_dirs) - converted} already done/missing input)")


if __name__ == "__main__":
    main()
