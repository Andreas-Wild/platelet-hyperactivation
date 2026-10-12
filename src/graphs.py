"""The box plots of the paper: concentration and platelet activation by cohort.

Reads the per-donor CSVs that `main.py` writes, so run that first. One vector
PDF per figure lands in `output/`:

    obj_per_uL_<population>_new_method.pdf    primary endpoint, one per population
    median_<marker>_platelet_new_method.pdf   per-donor median PAC-1 / CD62P

    uv run src/graphs.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import utils

OUTPUT_DIR = utils.ROOT / "output"

METHOD = "new_method"
ENDPOINT = "obj_per_uL"
COHORTS = ["control", "pasc"]
POP_COLORS = {"platelet": "#4477AA", "rbc": "#CC6677", "aggregate": "#DDAA33"}
#: Activation is only read on the platelets, one figure per marker.
MARKERS = ["PAC1", "CD62"]
#: Marker names as they should appear in figures; the CSVs use the short forms.
MARKER_LABELS = {"PAC1": "PAC-1", "CD62": "CD62P"}
STATISTIC = "median"

plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3 — journals reject Type 3


def boxplot(groups, color, ylabel, filename, legend=False):
    """Control vs Long COVID boxplot, one box per cohort, saved as a vector PDF."""
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    bp = ax.boxplot(
        groups,
        tick_labels=["Control", "Long COVID"],
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops={"facecolor": color, "alpha": 0.5},
        medianprops={"color": "black"},
        meanprops={"color": "black", "linestyle": "--", "linewidth": 1},
    )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=12)
    if legend:
        # boxplot draws one median/mean line per box; either box's line is a fine
        # legend handle since both are styled identically.
        ax.legend(
            [bp["medians"][0], bp["means"][0]],
            ["Median", "Mean"],
            loc="upper left", fontsize=10, frameon=False,
        )
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# %% Primary endpoint -- concentration, one figure per population

df = pd.read_csv(OUTPUT_DIR / "endpoints_per_donor.csv")
df = df[df["method"] == METHOD]

for population, color in POP_COLORS.items():
    df_pop = df[df["population"] == population]
    boxplot(
        [df_pop[df_pop["cohort"] == c][ENDPOINT] for c in COHORTS],
        color,
        "Objects per µL",
        OUTPUT_DIR / f"{ENDPOINT}_{population}_{METHOD}.pdf",
        legend=population == "rbc",
    )


# %% Activation -- per-donor median platelet intensity, one figure per marker

activation = pd.read_csv(OUTPUT_DIR / "staining_per_donor.csv")
activation = activation[
    (activation["method"] == METHOD)
    & (activation["population"] == "platelet")
    & (activation["statistic"] == STATISTIC)
]

for marker in MARKERS:
    df_marker = activation[activation["marker"] == marker]
    boxplot(
        [df_marker[df_marker["cohort"] == c]["value"] for c in COHORTS],
        POP_COLORS["platelet"],
        f"Median {MARKER_LABELS[marker]} intensity (a.u.)",
        OUTPUT_DIR / f"{STATISTIC}_{marker}_platelet_{METHOD}.pdf",
        legend=marker == MARKERS[0],
    )
