# Platelet hyperactivation — statistical analysis

Reproduces the statistical analysis of the platelet hyperactivation imaging flow
cytometry study, from the published per-object image features.

Two files do the work: `utils.py` (loading and population classification) and
`main.py` (the analysis and its statistics, written as `# %%` cells so it can be
stepped through in a REPL or run top to bottom). It takes about ten seconds.

## Running it

```sh
uv run python main.py
```

Results are printed and written as CSVs to `output/`.

## Data

The feature tables are distributed separately from this repository as `data.tar.gz`. Unpack them with layout:

```
data/<method>/<donor>.parquet
```

with one file per donor under each of the four preparation methods
(`new_method`, `normal_method`, `prp`, `whole_blood`). Each file holds one row
per segmented mask and 83 image features; only the columns each step needs are
read.

Donors imaged under all four methods form the analysis set (n = 19; 11 control,
8 PASC). The secondary analysis blocks on donor, so a donor missing from any
method is excluded from all of them.

`donors.csv` carries the two quantities that are not per-cell image features and
so are absent from the parquet files: each donor's cohort, and the sample volume
imaged under each method. Volume is what turns a count into the primary
endpoint.

## What is computed

**Populations.** Masks below the brightfield noise floor
(`bf_std_intensity <= 8`) are the segmentation model's attempts at empty
background and are not cells. Every remaining mask is called RBC-like,
platelet-like, or other on size, shape, and CD235; masks lying within one pixel
of each other are then grouped into objects. An object of two or more real masks
is an `aggregate`; an object of one real mask takes that mask's population
(`platelet` or `rbc`). Everything else is `irregular` and is excluded from all
endpoints. Thresholds are the ones given in the paper and are collected at the
top of `utils.py`.

PAC1 and CD62 positivity is defined per method against an internal negative
control: RBCs express neither marker, so the 99th percentile of each marker
among that method's RBC-like masks is its non-specific ceiling. The gates
actually used are written to `output/marker_gates.csv`.

**Primary and exploratory endpoints.** Cell concentration (objects/µL) is the
primary endpoint; absolute count, % of total, and median cell area are
exploratory. Each is compared between cohorts with an unpaired, two-sided
Mann–Whitney U test and reported with Cliff's δ, interpreted on the thresholds
of Romano et al. (negligible / small / medium / large). The donor is the unit of
analysis — cell features are collapsed to one number per donor before any test
runs, so cells are never treated as independent observations. p-values are
Benjamini–Hochberg corrected across the four endpoints, i.e. one family per
method × population. A positive δ means the endpoint is higher in PASC.

**Secondary analysis.** Whether the four preparation methods stain equally well.
Median and 95th-percentile PAC1 and CD62 intensities are compared across methods
with a Friedman test blocked on donor, followed by post hoc pairwise two-sided
Wilcoxon signed-rank tests. p-values are BH corrected across populations,
markers, and summary statistics jointly — one family for the omnibus tests, one
for the post hoc tests.

## Outputs

`output.txt` summarises the results in prose and tables; the methodology behind
them is the section above rather than repeated there.

| file | contents |
| --- | --- |
| `endpoints_per_donor.csv` | the four endpoints, one row per method × donor × population |
| `endpoints_stats.csv` | Mann–Whitney p, Cliff's δ and magnitude, BH-corrected p |
| `staining_per_donor.csv` | PAC1/CD62 median and p95, one row per method × donor × population |
| `staining_friedman.csv` | Friedman χ² and BH-corrected p across the four methods |
| `staining_pairwise.csv` | post hoc pairwise Wilcoxon signed-rank tests |
| `marker_gates.csv` | the PAC1/CD62 positivity thresholds used for each method |

## Not included

The fine-tuned segmentation model and its training dataset are not available at
the time of writing. If you are interested in using the model or in
collaborating, contact the authors.
