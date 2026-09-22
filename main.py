"""Statistical analysis for the platelet hyperactivation study.

Reproduces the Methodology section of the paper, and nothing else:
the primary endpoint (cell concentration, control vs PASC) with three
exploratory endpoints, and a secondary analysis of whether the four preparation
methods stain equally well. README.md describes what each test is and why.

Written as `# %%` cells: run it top to bottom, or step through it in a REPL.

The console shows the primary endpoint only. Everything else is summarised in
`output/output.txt`, with the full results in the CSVs beside it.
"""

# %% Imports

import itertools

import numpy as np
import pandas as pd
from scipy import stats

import utils

#----------------------------------Initialisation----------------------------------
#: The primary endpoint, and the three exploratory ones. Each is one number per
#: donor x population; the four together form one BH family per population.
ENDPOINTS = {
    "obj_per_uL": "concentration (obj/uL)",
    "count": "absolute count",
    "pct_total": "% of total",
    "area_um2_median": "median area (um^2)",
}
PRIMARY_ENDPOINT = "obj_per_uL"

#: The method chosen a priori for the primary endpoint. Its concentrations form
#: their own BH family and are reported in the console. The other three methods
#: are analysed identically; their results go to the CSVs.
HEADLINE_METHOD = "new_method"

#: Hyperactivation is read on one method only (Methodology: "of the new_method"),
#: from the same per-donor staining summaries the secondary analysis uses.
ACTIVATION_METHOD = "new_method"
ACTIVATION_STATISTICS = ("median", "p95")
#: Row order: the population the question is about first, then the two that act
#: as its controls.
ACTIVATION_POPULATIONS = ("platelet", "aggregate", "rbc")

ALPHA = 0.05

#: Romano et al. thresholds for interpreting |Cliff's delta|: below the first
#: cut the effect is negligible, above the last it is large.
DELTA_CUTS = (0.147, 0.33, 0.474)
DELTA_LABELS = ("negligible", "small", "medium", "large")

OUTPUT_DIR = utils.ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def as_table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda v: f"{v:,.4g}")


def cliffs_delta(control: np.ndarray, pasc: np.ndarray) -> float:
    """How much more often PASC exceeds control than the reverse.

    Positive means the quantity is higher in PASC. Nonparametric and unpaired,
    matching the Mann-Whitney U test it is reported beside.
    """
    pasc_higher = np.sum(pasc[:, None] > control[None, :])
    pasc_lower = np.sum(pasc[:, None] < control[None, :])
    return (pasc_higher - pasc_lower) / (control.size * pasc.size)


# %% Donor metadata and the common donor set
donors = pd.read_csv(utils.DONORS_CSV)
cohorts = donors.drop_duplicates("donor").set_index("donor")["cohort"]
volumes = donors.set_index(["method", "donor"])["volume_uL"]

# The secondary analysis blocks on donor, so only donors imaged under every
# method are usable -- and using the same set everywhere keeps n and the
# classifier's inputs identical across methods.
donors_per_method = (set(group["donor"]) for _, group in donors.groupby("method"))
common = sorted(set.intersection(*donors_per_method))

cohort_of = cohorts.reindex(common)
cohort_sizes = cohort_of.value_counts()
# Segmentation & Feature Extraction pipelines are not included in this repository.
#----------------------------------Data Analysis----------------------------------
# %% Classify every object and collapse to one number per donor
# One method at a time: the marker positivity gates are a property of the
# method, so they are fixed over all its donors before any donor is classified.
endpoint_rows: list[pd.DataFrame] = []
staining_rows: list[pd.DataFrame] = []
gate_rows: list[dict] = []

total_objects = 0
for method in utils.METHODS:
    gates = utils.marker_gates(method, common)
    gate_rows.append({"method": method, **gates})

    for donor in common:
        # Classification happens inside the object_table fucntion
        objects = utils.object_table(method, donor, gates)
        total_objects += len(objects)

        endpoints = utils.donor_endpoints(objects, volumes[method, donor])
        staining = utils.donor_staining(objects)
        for frame in (endpoints, staining):
            frame.insert(0, "donor", donor)
            frame.insert(0, "cohort", cohorts[donor])
            frame.insert(0, "method", method)
        endpoint_rows.append(endpoints)
        staining_rows.append(staining)

per_donor = pd.concat(endpoint_rows, ignore_index=True)
per_donor_staining = pd.concat(staining_rows, ignore_index=True)
n_cells = int(per_donor["count"].sum())

#---------------------------------Statistical Analysis---------------------------------
# %% Primary and exploratory endpoints -- control vs PASC
# The donor is the unit of analysis: every cell feature is already collapsed to
# one number per donor, so treating cells as independent observations (which
# would be pseudoreplication) cannot happen here.
comparisons = []
for (method, population), group in per_donor.groupby(["method", "population"]):
    for endpoint in ENDPOINTS:
        control = group.loc[group["cohort"] == "control", endpoint].dropna().to_numpy()
        pasc = group.loc[group["cohort"] == "pasc", endpoint].dropna().to_numpy()
        delta = cliffs_delta(control, pasc)
        magnitude = DELTA_LABELS[np.searchsorted(DELTA_CUTS, abs(delta), side="right")]
        comparisons.append(
            {
                "method": method,
                "population": population,
                "endpoint": endpoint,
                "n_control": control.size,
                "n_pasc": pasc.size,
                "control_median": np.median(control),
                "pasc_median": np.median(pasc),
                "cliffs_delta": delta,
                "magnitude": magnitude,
                # Unpaired and two-sided: the cohorts are different individuals.
                "p_value": stats.mannwhitneyu(control, pasc).pvalue,
            }
        )

endpoint_stats = pd.DataFrame(comparisons)
# BH across the four endpoints, i.e. one family per method x population.
endpoint_stats["p_bh"] = endpoint_stats.groupby(["method", "population"])[
    "p_value"
].transform(lambda p: stats.false_discovery_control(p, method="bh"))
# The primary endpoint is a family of its own: the concentration of the three
# populations under the a priori method, BH across those three. It replaces the
# per-population value on those rows only.
is_primary = (endpoint_stats["method"] == HEADLINE_METHOD) & (
    endpoint_stats["endpoint"] == PRIMARY_ENDPOINT
)
endpoint_stats.loc[is_primary, "p_bh"] = stats.false_discovery_control(
    endpoint_stats.loc[is_primary, "p_value"], method="bh"
)


# %% Secondary analysis -- staining across the four preparation methods
# Every donor is measured under all four methods, so donors are blocks and the
# comparison is within-donor throughout.
blocked = per_donor_staining.pivot_table(
    index=["population", "marker", "statistic", "donor"],
    columns="method",
    values="value",
).reindex(columns=list(utils.METHODS))

omnibus, pairwise = [], []
for (population, marker, statistic), block in blocked.groupby(
    level=["population", "marker", "statistic"]
):
    what = {"population": population, "marker": marker, "statistic": statistic}
    chi2, p_value = stats.friedmanchisquare(*block.to_numpy().T)
    omnibus.append(
        {**what, "n_donors": len(block), "friedman_chi2": chi2, "p_value": p_value}
    )
    for method_a, method_b in itertools.combinations(utils.METHODS, 2):
        a, b = block[method_a].to_numpy(), block[method_b].to_numpy()
        # Paired: the same donor under two methods.
        result = stats.wilcoxon(a, b)
        pairwise.append(
            {
                **what,
                "method_a": method_a,
                "method_b": method_b,
                "median_a": np.median(a),
                "median_b": np.median(b),
                "wilcoxon_stat": result.statistic,
                "p_value": result.pvalue,
            }
        )

# BH across all populations, markers and summary statistics jointly -- one
# family for the omnibus tests, one for the post hoc tests.
friedman_stats = pd.DataFrame(omnibus)
friedman_stats["p_bh"] = stats.false_discovery_control(
    friedman_stats["p_value"], method="bh"
)
pairwise_stats = pd.DataFrame(pairwise)
pairwise_stats["p_bh"] = stats.false_discovery_control(
    pairwise_stats["p_value"], method="bh"
)

# A Friedman test says the methods differ, not which is best. The ranking is
# the platelet signal relative to the RBC internal negative: RBCs express
# neither marker, so a ratio near 1 means the method is reading background,
# however bright it looks in absolute terms.
# The ratio is taken per donor and then summarised as the median over donors,
# as in the paper's sample-preparation table: a ratio of medians is not the
# median of ratios.
donor_separation = (
    per_donor_staining[per_donor_staining["statistic"] == "median"]
    .pivot_table(
        index=["method", "marker", "donor"], columns="population", values="value"
    )
    .reset_index()
)
donor_separation["platelet_over_rbc"] = (
    donor_separation["platelet"] / donor_separation["rbc"]
)
marker_separation = (
    donor_separation.groupby(["method", "marker"])[
        ["platelet", "rbc", "platelet_over_rbc"]
    ]
    .median()
    .reset_index()
)
# Best = the method whose weakest marker separates most.
weakest_marker_ratio = marker_separation.groupby("method")["platelet_over_rbc"].min()
best_method = weakest_marker_ratio.idxmax()
best_method_ratios = marker_separation[
    marker_separation["method"] == best_method
].set_index("marker")
runner_up_ratio = weakest_marker_ratio.drop(best_method).max()

# %% Platelet hyperactivation -- control vs PASC, one method
# Where the secondary analysis asks whether the four methods stain equally, this
# asks whether the cohorts differ under the one method chosen to report on. The
# per-donor summaries are the same ones -- each donor's median and 95th
# percentile marker intensity -- so the donor is again the unit of analysis and
# the test is unpaired and two-sided, the cohorts being different individuals.
activation = per_donor_staining[
    per_donor_staining["method"] == ACTIVATION_METHOD
]

activation_rows = []
for population in ACTIVATION_POPULATIONS:
    for marker in utils.MARKERS:
        for statistic in ACTIVATION_STATISTICS:
            group = activation[
                (activation["population"] == population)
                & (activation["marker"] == marker)
                & (activation["statistic"] == statistic)
            ]
            values = group.set_index("cohort")["value"].dropna()
            control = values.loc[["control"]].to_numpy()
            pasc = values.loc[["pasc"]].to_numpy()
            activation_rows.append(
                {
                    "population": population,
                    "marker": marker,
                    "statistic": statistic,
                    "n_control": control.size,
                    "n_lv": pasc.size,
                    "control_median": np.median(control),
                    "lv_median": np.median(pasc),
                    "cliffs_delta": cliffs_delta(control, pasc),
                    "p_two_sided": stats.mannwhitneyu(control, pasc).pvalue,
                }
            )

activation_stats = pd.DataFrame(activation_rows)
# One family: BH across all populations, markers and summary statistics of this
# method jointly, as for the secondary analysis.
activation_stats["p_bh"] = stats.false_discovery_control(
    activation_stats["p_two_sided"], method="bh"
)
for column, source in (("sig_raw", "p_two_sided"), ("sig_bh", "p_bh")):
    activation_stats[column] = np.where(activation_stats[source] < ALPHA, "*", "ns")

# Kept out of the CSV, whose columns are fixed, but wanted for the report.
activation_magnitude = [
    DELTA_LABELS[np.searchsorted(DELTA_CUTS, abs(delta), side="right")]
    for delta in activation_stats["cliffs_delta"]
]

#----------------------------------Report Writing----------------------------------
# %% Report
outputs = {
    "endpoints_per_donor.csv": per_donor,
    "endpoints_stats.csv": endpoint_stats,
    "staining_per_donor.csv": per_donor_staining,
    "staining_friedman.csv": friedman_stats,
    "staining_pairwise.csv": pairwise_stats,
    "marker_gates.csv": pd.DataFrame(gate_rows),
    "activation_stats.csv": activation_stats,
}
for name, frame in outputs.items():
    frame.to_csv(OUTPUT_DIR / name, index=False)

primary = (
    endpoint_stats[
        (endpoint_stats["method"] == HEADLINE_METHOD)
        & (endpoint_stats["endpoint"] == PRIMARY_ENDPOINT)
    ]
    .set_index("population")
    .reindex(utils.POPULATIONS)
    .reset_index()[
        [
            "population",
            "control_median",
            "pasc_median",
            "cliffs_delta",
            "magnitude",
            "p_bh",
        ]
    ]
    .rename(columns={"control_median": "control", "pasc_median": "PASC"})
)
staining_verdict = (
    f"{best_method} stains best: the platelet signal sits "
    f"{best_method_ratios.loc['PAC1', 'platelet_over_rbc']:.2f}x (PAC1) and "
    f"{best_method_ratios.loc['CD62', 'platelet_over_rbc']:.2f}x (CD62) above the "
    f"RBC internal negative, against at most {runner_up_ratio:.2f}x for the other "
    f"three methods."
)

print(
    f"{len(common)} donors ({cohort_sizes['control']} control, "
    f"{cohort_sizes['pasc']} PASC) x {len(utils.METHODS)} methods "
    f"-> {total_objects:,} segmented objects\n"
)
print(f"Primary endpoint -- cell concentration (obj/uL), {HEADLINE_METHOD}")
print(as_table(primary))
print(f"\nSecondary analysis: {staining_verdict}")

# Everything else: which comparisons are significant, and where to read them.
significant = endpoint_stats[endpoint_stats["p_bh"] < ALPHA]
n_friedman_sig = int((friedman_stats["p_bh"] < ALPHA).sum())
n_pairwise_sig = int((pairwise_stats["p_bh"] < ALPHA).sum())
n_activation_sig = int((activation_stats["p_bh"] < ALPHA).sum())
activation_table = activation_stats.assign(magnitude=activation_magnitude)[
    [
        "population",
        "marker",
        "statistic",
        "control_median",
        "lv_median",
        "cliffs_delta",
        "magnitude",
        "p_bh",
    ]
].rename(columns={"control_median": "control", "lv_median": "PASC"})
control_ids = ", ".join(cohort_of[cohort_of == "control"].index)
pasc_ids = ", ".join(cohort_of[cohort_of == "pasc"].index)

report = f"""\
Platelet hyperactivation -- statistical analysis
{"=" * 64}
What each test is and why it was chosen: see README.md.

Donors imaged under all {len(utils.METHODS)} methods: n = {len(common)}
  control  n = {cohort_sizes["control"]:2d}  {control_ids}
  pasc     n = {cohort_sizes["pasc"]:2d}  {pasc_ids}

Segmented objects: {total_objects:,}
  {n_cells:,} in the three target populations
  {total_objects - n_cells:,} irregular (excluded from every endpoint)


1. COHORT COMPARISON -- control vs PASC
{"=" * 64}
Significant after BH correction (p_BH < {ALPHA}): {len(significant)} of \
{len(endpoint_stats)} comparisons.
A positive Cliff's delta means the endpoint is higher in PASC.

{as_table(significant.drop(columns=["n_control", "n_pasc", "p_value"]))}

Full results, including uncorrected p-values, are in
output/endpoints_stats.csv; the per-donor values they were computed from are
in output/endpoints_per_donor.csv.


2. STAINING ACROSS THE FOUR PREPARATION METHODS
{"=" * 64}
Friedman: {n_friedman_sig} of {len(friedman_stats)} tests significant, largest
p_BH {friedman_stats["p_bh"].max():.2e}. The four methods do not stain equally.

Post hoc: {n_pairwise_sig} of {len(pairwise_stats)} pairwise tests significant
after correction.

The omnibus test establishes that the methods differ, not which is best. The
ranking below is the platelet signal relative to the RBC internal negative, so
a ratio near 1 means the method is reading background, however bright it looks
in absolute terms.

{as_table(marker_separation.round(3))}

{staining_verdict}

Note prp, which is the instructive near-miss: its absolute intensities look
healthy, but its RBCs are equally bright, so that is non-specific staining
rather than signal.

Full results: output/staining_friedman.csv, output/staining_pairwise.csv, and
the per-donor values in output/staining_per_donor.csv. The PAC1/CD62
positivity gates each method's classifier used are in output/marker_gates.csv.


3. PLATELET HYPERACTIVATION -- control vs PASC, {ACTIVATION_METHOD}
{"=" * 64}
Each donor's median and 95th-percentile PAC1 and CD62 intensity, compared
between cohorts. A positive Cliff's delta means the intensity is higher in
PASC.

{as_table(activation_table)}

Significant after BH correction across all {len(activation_stats)} tests: \
{n_activation_sig}.

Platelet PAC1 and CD62 both sit higher in PASC, by a small to medium effect
that {len(cohort_of)} donors are not enough to separate from chance. The RBC
rows are the control on that reading: RBCs express neither marker, so their
deltas are the noise floor this comparison is measured against, and they are
of the same size.

Full results: output/activation_stats.csv; the per-donor values they were
computed from are the {ACTIVATION_METHOD} rows of
output/staining_per_donor.csv.
"""
(OUTPUT_DIR / "output.txt").write_text(report)
print(
    f"\nSummary of everything else in {OUTPUT_DIR.name}/output.txt; "
    f"full results in {len(outputs)} CSVs beside it."
)
