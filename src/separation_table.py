"""Turn the per-donor staining and endpoint CSVs into the sample-preparation table.

Same shape as `endpoints_table.py`: the config cell is the part you edit, the rest
applies it. Run top to bottom and the table lands in
`output/separation_table.tex`; \\input it from the report.

The table ranks the preparation methods on three things:

- the median platelet intensity of each marker, which needs no reference
  population and so is defined for every method, PRP included;
- the platelet/RBC fluorescence ratio, i.e. platelet-marker signal against the
  RBC internal negative control;
- the number of platelet and aggregate events per RBC event, i.e. how strongly a
  method enriches the populations of interest relative to RBCs. A within-sample
  ratio, so the method-specific dilutions cancel.

The last two use RBCs as the reference. That is only meaningful in fractions
that contain RBCs: PRP is RBC-depleted by design, so what the pipeline calls
RBCs in PRP are mostly misclassified platelets (median area about half that of
a true RBC). PRP's reference-based cells are therefore printed as n/a rather
than as numbers that look comparable but are not.

All numbers are per donor first and pooled second: each donor contributes one
value per method, and the table reports the median across donors with the
interquartile range. A ratio of medians is not the median of ratios, and the
per-donor form is the one that matches how the ratios are argued.

Preamble needs `\\usepackage{booktabs}` and `\\usepackage{siunitx}`; see the
fontenc warning at the top of `endpoints_table.py`.
"""

# %% Imports

import pandas as pd

import utils

STAINING_CSV = utils.ROOT / "output" / "staining_per_donor.csv"
ENDPOINTS_CSV = utils.ROOT / "output" / "endpoints_per_donor.csv"
TEX_OUT = utils.ROOT / "output" / "separation_table.tex"

# %% Config -- this is the part you edit

#: Which per-cell summary the intensity columns are built from.
STATISTIC = "median"

#: The reference population the ratios divide by. RBCs express neither
#: platelet marker, so they are the internal negative control -- but only in
#: fractions that contain RBCs.
REFERENCE = "rbc"

#: Methods without a real RBC population. Their reference-based cells print
#: as n/a.
RBC_DEPLETED = ["prp"]

#: Row order. RBC-depleted methods last, below a rule.
MARKERS = ["PAC1", "CD62"]
METHODS = ["new_method", "normal_method", "whole_blood", "prp"]

#: Populations whose events are counted against the reference.
ENRICHED = ["platelet", "aggregate"]

LABELS = {
    "new_method": "\\gls{hpo}",
    "normal_method": "Haematocrit",
    "prp": "\\gls{prp}",
    "whole_blood": "\\gls{wb}",
    "PAC1": "PAC-1",
    "CD62": "CD62P",
    "platelet": "Platelets",
    "aggregate": "Aggregates",
}

#: The quantiles the bracketed interval spans.
IQR = (0.25, 0.75)

NA = "n/a$^\\dagger$"

CAPTION = (
    "Sample preparation comparison. Each donor ($n=20$) contributes one value per "
    "method; entries are the median [IQR] across donors. Platelet intensity is the "
    "median platelet fluorescence (a.u.). Platelet/RBC fluorescence is the ratio of "
    "platelet to RBC median fluorescence, a ratio near \\num{1} indicates platelet "
    "signal indistinguishable from the RBC negative control. Events per RBC is the "
    "number of platelet or aggregate events per RBC event in the same sample."
)
NOTE = (
    "$^\\dagger$\\gls{prp} is RBC-depleted by design; RBC-referenced "
    "measures are undefined."
)
LABEL = "tab:separation"

SISETUP = "\\sisetup{mode=match, reset-text-series=false, propagate-math-font=true}"

# %% Per-donor values

staining = pd.read_csv(STAINING_CSV)
staining = staining[staining["statistic"] == STATISTIC]

#: One row per donor x method, one column per marker x population.
intensity = staining.pivot_table(
    index=["method", "donor"], columns=["marker", "population"], values="value"
)

endpoints = pd.read_csv(ENDPOINTS_CSV)
counts = endpoints.pivot_table(
    index=["method", "donor"], columns="population", values="count"
)

per_donor = pd.DataFrame(index=intensity.index)
for marker in MARKERS:
    per_donor[f"{marker}_intensity"] = intensity[(marker, "platelet")]
    per_donor[f"{marker}_ratio"] = (
        intensity[(marker, "platelet")] / intensity[(marker, REFERENCE)]
    )
for population in ENRICHED:
    per_donor[f"{population}_per_rbc"] = counts[population] / counts[REFERENCE]

# %% Pooled across donors

donors = per_donor.groupby("method")
pooled = {
    name: donors.quantile(quantile)
    for name, quantile in [("q1", IQR[0]), ("median", 0.5), ("q3", IQR[1])]
}

print(pooled["median"].round(3).to_string())

# %% Render


def spread(method: str, column: str, digits: int) -> str:
    """Median [IQR], or n/a when the column needs RBCs the method does not have."""
    if method in RBC_DEPLETED and not column.endswith("_intensity"):
        return NA
    median, q1, q3 = (pooled[name].loc[method, column] for name in ("median", "q1", "q3"))
    fmt = f".{digits}f"
    return (
        f"\\num{{{median:{fmt}}}} [\\num{{{q1:{fmt}}}}--\\num{{{q3:{fmt}}}}]"
    )


COLUMNS = [
    *((f"{marker}_intensity", 0) for marker in MARKERS),
    *((f"{marker}_ratio", 2) for marker in MARKERS),
    *((f"{population}_per_rbc", 3) for population in ENRICHED),
]


def row(cells: list[str]) -> str:
    return " & ".join(cells) + " \\\\"


rows = []
for method in METHODS:
    #: A rule above the RBC-depleted block, so it reads as set apart.
    if method in RBC_DEPLETED and method == next(m for m in METHODS if m in RBC_DEPLETED):
        rows.append("\\midrule")
    rows.append(row([LABELS[method], *(spread(method, c, d) for c, d in COLUMNS)]))

groups = [
    ("Platelet intensity (a.u.)", len(MARKERS)),
    ("Platelet/RBC fluorescence", len(MARKERS)),
    ("Events per RBC", len(ENRICHED)),
]
headers = []
rules = []
start = 2
for title, width in groups:
    headers.append(f"\\multicolumn{{{width}}}{{c}}{{{title}}}")
    rules.append(f"\\cmidrule(lr){{{start}-{start + width - 1}}}")
    start += width

body = [
    f"\\begin{{tabular}}{{@{{}}l{'c' * len(COLUMNS)}@{{}}}}",
    "\\toprule",
    row(["", *headers]),
    " ".join(rules),
    row(
        [
            "Method",
            *(LABELS[m] for m in MARKERS),
            *(LABELS[m] for m in MARKERS),
            *(LABELS[p] for p in ENRICHED),
        ]
    ),
    "\\midrule",
    *rows,
    "\\bottomrule",
    f"\\multicolumn{{{1 + len(COLUMNS)}}}{{@{{}}l}}{{\\footnotesize {NOTE}}}",
    "\\end{tabular}",
]

tex = "\n".join(
    [
        "% Generated by separation_table.py from output/staining_per_donor.csv"
        " and output/endpoints_per_donor.csv -- do not edit.",
        "\\begin{table*}[htbp]",
        "\t\\centering",
        "\t\\footnotesize",
        f"\t{SISETUP}",
        f"\t\\caption{{{CAPTION}}}",
        f"\t\\label{{{LABEL}}}",
        *(f"\t{line}" for line in body),
        "\\end{table*}",
        "",
    ]
)

TEX_OUT.write_text(tex)
print(tex)
print(f"Written to {TEX_OUT}")
