from pathlib import Path
import pymol2
import csv

cif_dir = Path("/weka/scratch/jgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/af3_benchmark/dhd_collected_cif_files")

hbond_cutoff = 3.5
hbond_angle = 45

results = {}

with pymol2.PyMOL() as p:
    cmd = p.cmd

    for cif_file in sorted(cif_dir.glob("*.cif")):
        cmd.reinitialize()
        cmd.load(str(cif_file), "structure")
        cmd.h_add("structure")

        chains = cmd.get_chains("structure")

        chain_a = chains[0]
        chain_b = chains[1]

        pairs_a_donor = cmd.find_pairs(
            f"(structure and chain {chain_a} and donor)",
            f"(structure and chain {chain_b} and acceptor)",
            cutoff=hbond_cutoff,
            angle=hbond_angle
        )

        pairs_b_donor = cmd.find_pairs(
            f"(structure and chain {chain_b} and donor)",
            f"(structure and chain {chain_a} and acceptor)",
            cutoff=hbond_cutoff,
            angle=hbond_angle
        )

        total_interface_hbonds = len(pairs_a_donor) + len(pairs_b_donor)

        results[cif_file.name] = total_interface_hbonds
        print(f"{cif_file.name}: {total_interface_hbonds} interface h-bonds")

print("\n--- Summary ---")
for filename, count in results.items():
    print(f"{filename}: {count}")

results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "dhd"
results_dir.mkdir(parents=True, exist_ok=True)
results_csv_path = results_dir / "dhd_af3_interface_hbond_results.csv"

with open(results_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "interface_hbonds"])
    for filename, count in results.items():
        writer.writerow([filename, count])

print(f"\nResults saved to {results_csv_path}")