# Platelet hyperactivation — statistical analysis

Code for the statistical analysis of the platelet hyperactivation imaging flow
cytometry study (control vs PASC donors, four sample-preparation methods). It
accompanies the article *Deep-learning instance segmentation and explainable
rule-based classification to resolve platelets, red blood cells and aggregates
in imaging flow cytometry* (Wild, Venter, Perold, Niesler & Pretorius,
submitted to Scientific Reports).

The code works from the published per-object image features. Each script in
`src/` reproduces one part of the paper. Together they produce every statistic,
table and figure built from the feature tables and the validation labels.

The code and the paper use different names for the four preparation methods:

| code | paper |
| --- | --- |
| `new_method` | high-yield platelet optimization (HPO) |
| `normal_method` | hematocrit |
| `prp` | platelet-rich plasma (PRP) |
| `whole_blood` | whole blood (WB) |

The cohort the code calls `pasc` is the Long COVID cohort of the paper.

## Repository layout

```
.
├── README.md            this file: setup, data, what is computed, outputs
├── AGENTS.md            a short guide to this repository for AI assistants
├── LICENSE              MIT licence for the code
├── CITATION.cff         how to cite this code and the paper
├── pyproject.toml       dependencies (Python ≥ 3.14)
├── uv.lock              exact dependency versions used for the paper
├── donors.csv           cohort and imaged sample volume per donor × method
├── acquisitions.csv     events acquired per HPO sample
├── data_manifest.txt    every file the feature archives unpack to
├── SHA256SUMS.txt       checksums of the four feature archives
├── src/                 all code; see "Scripts" below
├── validation/          human labels of the two validation sets, and their images
├── data/                feature tables, unpacked here (not in git)
└── output/              everything the scripts write (not in git)
```

## Setup

The environment is pinned in `uv.lock`. With [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

With pip, in a Python ≥ 3.14 virtual environment:

```sh
pip install .
```

This installs the dependencies only. The scripts are run in place from `src/`
and are not installed as a package.

## Data

The feature tables are distributed separately from this repository as one
archive per preparation method: `new_method.tar`, `normal_method.tar`,
`prp.tar` and `whole_blood.tar`. They are deposited on Zenodo with `donors.csv`
and `SHA256SUMS.txt` under a CC BY 4.0 licence:
[10.5281/zenodo.22895198](https://doi.org/10.5281/zenodo.22895198). Download
them into the repository root, then verify and unpack them:

```sh
sha256sum -c SHA256SUMS.txt
for f in new_method.tar normal_method.tar prp.tar whole_blood.tar; do tar xf "$f"; done
```

This creates `data/<method>/<donor>.parquet`, one file per donor under each of
the four methods. `data_manifest.txt` lists every file you should end up with.
Each file holds one row per segmented mask and 83 image features. Each step
reads only the columns it needs.

Donors imaged under all four methods form the analysis set (n = 20; 11 control,
9 PASC). The secondary analysis blocks on donor, so a donor missing from any
method is excluded from every analysis. This applies to C63, which has no
`new_method` acquisition.

`donors.csv` holds the two quantities that are not per-cell image features and
are therefore missing from the parquet files: each donor's cohort, and the
sample volume imaged under each method. The volume turns a count into the
primary endpoint. It is copied from the `sample_volume_uL` root attribute of
each acquisition's raw file.

`acquisitions.csv` holds the number of events acquired per HPO sample, copied
from the `n_events` attribute of each raw file. It is reported in the Methods.
The raw IFC files are not distributed, so these two files carry everything the
analysis needs from them.

## Running the analysis

Run from the repository root. `main.py` comes first, because the other scripts
read the CSVs it writes. It takes about ten seconds.

```sh
uv run src/main.py              # all statistics -> output/*.csv, output/output.txt
uv run src/paper_numbers.py     # in-text numbers and the minimum detectable effect
uv run src/endpoints_table.py   # output/endpoints_table.tex
uv run src/separation_table.py  # output/separation_table.tex
uv run src/graphs.py            # output/*.pdf box plots
uv run src/volumes.py           # sample-volume table rows (printed)
uv run src/inter_annotator.py   # classifier vs human annotators -> output/validation_scores.txt
```

With pip, replace `uv run` with `python`. `main.py` is written as `# %%` cells,
so you can also step through it in an editor or REPL.

## Scripts

All code is in `src/`. `utils.py` is the shared library. Every other file is a
script that you run directly. The scripts use only what is published: the
feature tables from Zenodo and the files in this repository.

| script | reproduces |
| --- | --- |
| `utils.py` | loading the feature tables, the population classifier and the per-donor summaries (Methodology, "Data Analysis"); all classifier thresholds are at the top |
| `main.py` | every hypothesis test: primary and exploratory endpoints, staining across methods, hyperactivation |
| `paper_numbers.py` | the in-text numbers no table carries, and the minimum detectable effect (power simulation) |
| `endpoints_table.py` | the endpoint table, as LaTeX, from `endpoints_stats.csv` |
| `separation_table.py` | the sample-preparation table, as LaTeX |
| `graphs.py` | the concentration and activation box plots |
| `volumes.py` | the imaged-volume summary per method and cohort |
| `inter_annotator.py` | the Pipeline Validation results: the classifier against the human annotators on both validation sets, from the labels in `validation/` |

## Validation labels

`validation/` holds the human labels behind the Pipeline Validation section,
and the images they were given. Both sets were sampled from the `new_method`
(HPO) acquisitions, as described in the paper's Methods.

| path | contents |
| --- | --- |
| `labelled_3000/answer_key.csv` | the 3000 sampled objects: the classifier's class, donor, cohort, event and bounding box |
| `labelled_3000/labels.csv` | annotator 1's class for each object; the 40 objects missing were judged not to be a valid event |
| `panel_50/answer_key.csv` | the 50 panel objects and the classifier's class |
| `panel_50/answers/annotator_N.csv` | each of the five annotators' classes |
| `brightfield_images.zip` | the brightfield image of every object above |

Unzip the images with `unzip validation/brightfield_images.zip -d validation/`.
`labelled_3000/NNNN.png` and `panel_50/NN.png` are named by the `number` column
of the answer keys, and `panel_50/collage.png` is the numbered sheet the
annotators labelled (`collage_x6.png` is the same sheet enlarged six times).
Each image is a 72 × 72 px crop of the brightfield channel at native resolution
(1 px = 1 µm), centred on the object and shown on one fixed grey scale (raw
counts 230–850 mapped to 0–255). The images contain no segmentation masks or
outlines.

## What is computed

**Populations.** Masks below the brightfield noise floor
(`bf_std_intensity <= 8`) are the segmentation model's attempts at segmenting
empty background, not cells. Every remaining mask is classed as RBC-like,
platelet-like or other from its size, shape and CD235. Masks lying within one
pixel of each other are then grouped into objects. An object of two or more real
masks is an `aggregate`. An object of one real mask takes that mask's population
(`platelet` or `rbc`). Every other object is `irregular` and is excluded from
all endpoints. The thresholds are those given in the paper and are collected at
the top of `src/utils.py`.

PAC1 and CD62 positivity is defined per method against an internal negative
control. RBCs express neither marker, so the 99th percentile of each marker
among that method's RBC-like masks is its non-specific ceiling. The gates used
are written to `output/marker_gates.csv`.

**Primary and exploratory endpoints.** Cell concentration (objects/µL) is the
primary endpoint. Absolute count, % of total and median cell area are
exploratory. Each is compared between cohorts with an unpaired, two-sided
Mann–Whitney U test and reported with Cliff's δ, which is interpreted using the
thresholds of Romano et al. (negligible / small / medium / large). The donor is
the unit of analysis: cell features are reduced to one number per donor before
any test runs, so cells are never treated as independent observations. The
primary endpoint is a family of its own. It consists of the concentrations of
the three populations under `new_method`, Benjamini–Hochberg corrected across
those three. Every other p-value is corrected across the four endpoints, which
gives one family per method × population. A positive δ means the endpoint is
higher in PASC.

**Secondary analysis.** This tests whether the four preparation methods stain
equally well. Median and 95th-percentile PAC1 and CD62 intensities are compared
across methods with a Friedman test blocked on donor. Post hoc pairwise
comparisons use two-sided Wilcoxon signed-rank tests. p-values are BH corrected
across populations, markers and summary statistics jointly, with one family for
the omnibus tests and one for the post hoc tests.

**Platelet hyperactivation.** This tests whether the cohorts differ in marker
intensity under `new_method`. The endpoint is the per-donor summary also used in
the secondary analysis: each donor's median and 95th-percentile PAC1 and CD62
intensity. It is compared between cohorts with an unpaired, two-sided
Mann–Whitney U test and reported with Cliff's δ. p-values are BH corrected
across populations, markers and summary statistics jointly, giving one family
of twelve. A positive δ means the intensity is higher in PASC.

## Outputs

`output/output.txt` summarises the results in prose and tables. The methods
behind them are described in the section above and are not repeated there.

| file | contents |
| --- | --- |
| `endpoints_per_donor.csv` | the four endpoints, one row per method × donor × population |
| `endpoints_stats.csv` | Mann–Whitney p, Cliff's δ and magnitude, BH-corrected p |
| `staining_per_donor.csv` | PAC1/CD62 median and p95, one row per method × donor × population |
| `staining_friedman.csv` | Friedman χ² and BH-corrected p across the four methods |
| `staining_pairwise.csv` | post hoc pairwise Wilcoxon signed-rank tests |
| `marker_gates.csv` | the PAC1/CD62 positivity thresholds used for each method |
| `validation_scores.txt` | the validation results, as printed by `inter_annotator.py` |
| `activation_stats.csv` | hyperactivation: Mann–Whitney p, Cliff's δ, BH-corrected p |
| `endpoints_table.tex`, `separation_table.tex` | the LaTeX tables |
| `obj_per_uL_*.pdf`, `median_*_platelet_*.pdf` | the box plots |

Areas are in µm². The FlowSight's 20× objective gives a nominal pixel of
1 × 1 µm, so one pixel is 1 µm² (`UM2_PER_PX` in `src/utils.py`). The
classifier's size thresholds are in pixels, as in the paper.

## Not included

The segmentation pipeline and feature extraction are not part of this
repository. The raw IFC files are available from the corresponding author on
reasonable request. The fine-tuned segmentation model is deposited on Zenodo
with restricted access
([10.5281/zenodo.22895731](https://doi.org/10.5281/zenodo.22895731)) and is
available on request for non-commercial research under the PolyForm Strict
License 1.0.0. Its training dataset is not publicly available.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The feature tables on
Zenodo are released under CC BY 4.0. To cite this work, see `CITATION.cff`.
