"""Turn `output/endpoints_stats.csv` into a LaTeX table for the report.

Everything you would want to change per table -- which rows, which columns, what
order, how each number is formatted -- lives in the config cell below. The rest
just applies it. Run top to bottom and the table lands in
`output/endpoints_table.tex`; \\input it from the report.

The preamble needs `\\usepackage{booktabs}` and `\\usepackage{siunitx}`.

On lualatex or xelatex with fontspec that is all -- do NOT also load fontenc,
which resets the encoding to T1 and silently drops the document back to
Computer Modern. Only on pdflatex add `\\usepackage[T1]{fontenc}`: under the
default OT1 encoding siunitx sets the "<" of a "<0.001" p-value from the text
font, where that slot is an inverted exclamation mark.
"""

# %% Imports

import math

import pandas as pd

import utils

STATS_CSV = utils.ROOT / "output" / "endpoints_stats.csv"
TEX_OUT = utils.ROOT / "output" / "endpoints_table.tex"

# %% Config -- this is the part you edit

#: Rows to keep. Every entry is a column name and the values allowed in it;
#: drop a line (or set it to None) to keep every value of that column.
FILTERS = {
    "method": ["new_method", "normal_method", "prp", "whole_blood"],
    "population": ["platelet", "aggregate", "rbc"],
    "endpoint": ["obj_per_uL", "count", "pct_total", "area_um2_median"],
}

#: Columns to report, in the order they should appear, as
#: csv_column -> (LaTeX header, formatter). Comment a line out to drop it;
#: move a line to move the column.
COLUMNS = {
    "endpoint": ("Endpoint", lambda v: LABELS.get(v, v)),
    "method": ("Method", lambda v: LABELS.get(v, v)),
    "population": ("Population", lambda v: LABELS.get(v, v)),
    # "n_control": ("$n_\\text{ctrl}$", "{:d}".format),
    # "n_pasc": ("$n_\\text{PASC}$", "{:d}".format),
    "control_median": ("Control median", lambda v: num(v)),
    "pasc_median": ("PASC median", lambda v: num(v)),
    "cliffs_delta": ("Cliff's $\\delta$", lambda v: num(v, decimals=2)),
    "magnitude": ("Effect Magnitude", lambda v: LABELS.get(v, v)),
    # "p_value": ("$p$", lambda v: format_p(v)),
    "p_bh": ("$p_\\text{BH}$", lambda v: format_p(v)),
}

#: Sort keys, outermost first. Prefix with "-" for descending.
SORT_BY = ["endpoint", "method", "population"]

#: Rows are ordered by these, not alphabetically, wherever they are sort keys.
ORDERINGS = {
    "method": ["new_method", "normal_method", "prp", "whole_blood"],
    "population": ["platelet", "rbc", "aggregate",],
    "endpoint": ["obj_per_uL", "count", "pct_total", "area_um2_median"],
    "magnitude": ["large", "medium", "small", "negligible"],
}

#: Anything the report should call something other than its csv spelling.
LABELS = {
    "new_method": "New method",
    "normal_method": "Normal method",
    "prp": "PRP",
    "whole_blood": "Whole blood",
    "platelet": "Platelet",
    "rbc": "RBC",
    "aggregate": "Aggregate",
    "obj_per_uL": "Concentration (obj/\\si{\\micro\\l})",
    "count": "Absolute count",
    "pct_total": "\\% of total",
    "area_um2_median": "Median area (\\si{\\micro\\metre\\squared})",
    "negligible": "Negligible",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}

#: Significant figures for the medians. Effect sizes and p-values set their own.
SIGNIFICANT_FIGURES = 3

#: Matches main.py. Adjusted p-values below it are bolded.
ALPHA = 0.05

CAPTION = (
    "Control vs PASC comparison of the primary and exploratory endpoints. "
    "$p_\\text{BH}$ is the Benjamini--Hochberg adjusted $p$-value within each "
    "population; effect sizes are Cliff's $\\delta$ with Romano et al. magnitudes."
)
LABEL = "tab:endpoints"

#: Emitted inside the table, so it applies here and not to the rest of the
#: report. Without it siunitx sets every number in its own math font and
#: `\textbf` around a `\num` is silently discarded, so nothing comes out bold.
SISETUP = "\\sisetup{mode=match, reset-text-series=false, propagate-math-font=true}"

# %% Formatting helpers

#: Escaped last, so LaTeX written in LABELS survives. Only characters that would
#: otherwise break compilation and never appear in our markup are listed.
_ESCAPES = {"&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_"}


def escape(text: str) -> str:
    """Make a raw csv string safe to typeset, leaving our own markup alone."""
    if any(marker in text for marker in ("\\", "$")):
        return text
    return "".join(_ESCAPES.get(char, char) for char in text)


def num(value: float, decimals: int | None = None) -> str:
    """A number for siunitx, in fixed notation with grouped digits.

    Fixed `decimals` if given, else SIGNIFICANT_FIGURES significant figures --
    the endpoints span four orders of magnitude, and `%g` would flip the larger
    ones into an exponent halfway down the column.
    """
    if pd.isna(value):
        return "--"
    if decimals is None:
        exponent = 0 if value == 0 else math.floor(math.log10(abs(value)))
        decimals = max(SIGNIFICANT_FIGURES - 1 - exponent, 0)
    return f"\\num{{{value:.{decimals}f}}}"


def format_p(value: float) -> str:
    """Report small p-values as a bound, and bold the significant ones."""
    if pd.isna(value):
        return "--"
    text = "\\num{<0.001}" if value < 0.001 else f"\\num{{{value:.3f}}}"
    return f"\\textbf{{{text}}}" if value < ALPHA else text


def order(series: pd.Series) -> pd.Series:
    """Sort key: the position in ORDERINGS if we have one, else the value."""
    levels = ORDERINGS.get(series.name)
    if levels is None:
        return series
    return series.map({value: i for i, value in enumerate(levels)})


# %% Select the rows and columns

stats = pd.read_csv(STATS_CSV)

table = stats
for column, keep in FILTERS.items():
    if keep is not None:
        table = table[table[column].isin(keep)]

keys = [column.lstrip("-") for column in SORT_BY]
table = table.sort_values(
    keys,
    ascending=[not column.startswith("-") for column in SORT_BY],
    key=order,
    kind="stable",
)
table = table[list(COLUMNS)]

print(f"{len(table)} of {len(stats)} rows, {len(table.columns)} columns")
print(table.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))

# %% Render

cells = pd.DataFrame(
    {
        header: table[column].map(formatter).map(escape)
        for column, (header, formatter) in COLUMNS.items()
    }
)

#: Text left, numbers right.
column_format = "".join(
    "l" if table[column].dtype == "str" else "r" for column in COLUMNS
)


def row(cells: list[str]) -> str:
    return " & ".join(cells) + " \\\\"


body = [
    f"\\begin{{tabular}}{{{column_format}}}",
    "\\toprule",
    row(list(cells.columns)),
    "\\midrule",
    *(row(list(values)) for values in cells.itertuples(index=False)),
    "\\bottomrule",
    "\\end{tabular}",
]

tex = "\n".join(
    [
        "% Generated by src/endpoints_table.py from output/endpoints_stats.csv"
        " -- do not edit.",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  {SISETUP}",
        f"  \\caption{{{CAPTION}}}",
        f"  \\label{{{LABEL}}}",
        *(f"  {line}" for line in body),
        "\\end{table}",
        "",
    ]
)

TEX_OUT.write_text(tex)
print(tex)
print(f"Written to {TEX_OUT}")
