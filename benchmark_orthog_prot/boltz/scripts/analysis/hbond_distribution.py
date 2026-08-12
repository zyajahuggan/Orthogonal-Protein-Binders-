import matplotlib.pyplot as plt
from pairing_utils import iter_datasets

MODEL_NAME = "Boltz-2"
COGNATE_COLOR = "#2a78d6"
NON_COGNATE_COLOR = "#1baf7a"


def save_and_report(fig, path, label):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {label} to {path}")


for df, _, project, _, results_dir in iter_datasets():
    cog_hbonds = df.loc[df["cognate_interaction"] == 1, "interface_hbonds"]
    non_cog_hbonds = df.loc[df["cognate_interaction"] == 0, "interface_hbonds"]
    assert not cog_hbonds.empty and not non_cog_hbonds.empty, \
        f"no cognate or non-cognate rows found for {project}"

    fig, ax = plt.subplots()
    ax.violinplot([cog_hbonds, non_cog_hbonds], [2, 4], widths=2, showmeans=True, showmedians=True, showextrema=True)

    ax.set_xticks([2, 4])
    ax.set_xticklabels(["Cognate", "Non-cognate"])
    ax.set_xlabel("Pair Type")
    ax.set_ylabel("Number of Interface H-bonds")
    ax.set_title(f"Interface H-bonds by Group ({MODEL_NAME}, {project})")

    out_dir = results_dir / "hbond_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_and_report(fig, out_dir / f"{project}_hbonds_violin_by_cognate.png", "violin plot")

    # density, not raw counts, since cognate/non-cognate group sizes are imbalanced
    bins = range(
        int(min(cog_hbonds.min(), non_cog_hbonds.min())),
        int(max(cog_hbonds.max(), non_cog_hbonds.max())) + 2,
    )

    fig, ax = plt.subplots()
    ax.hist(non_cog_hbonds, bins=bins, density=True, alpha=0.6, color=NON_COGNATE_COLOR, edgecolor="white", label="Non-cognate")
    ax.hist(cog_hbonds, bins=bins, density=True, alpha=0.6, color=COGNATE_COLOR, edgecolor="white", label="Cognate")

    ax.set_xlabel("Number of Interface H-bonds")
    ax.set_ylabel("Density")
    ax.set_title(f"Interface H-bonds Distribution ({MODEL_NAME}, {project})")
    ax.legend()

    save_and_report(fig, out_dir / f"{project}_hbonds_histogram_by_cognate.png", "histogram")
