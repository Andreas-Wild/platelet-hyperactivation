# Guide for AI assistants

This repository is the analysis code of a published study. Its job is to
reproduce the paper's numbers exactly, so read before you change anything.
`README.md` is the full reference. This file is the short route through it.

## Where things are

- `src/utils.py`: loading, the population classifier and per-donor summaries.
  Every classifier threshold is a named constant at the top. Start here to
  understand how a mask becomes a platelet, RBC or aggregate.
- `src/main.py`: every hypothesis test in the paper, as `# %%` cells. It writes
  the CSVs in `output/` that all other scripts read, so it runs first.
- `src/paper_numbers.py`, `src/endpoints_table.py`, `src/separation_table.py`,
  `src/graphs.py`, `src/volumes.py`: in-text numbers, tables and figures, built
  from `main.py`'s CSVs.
- `src/inter_annotator.py`: the validation results, from the labels in
  `validation/`.
- `donors.csv`: cohort and imaged volume per donor and method.
- `acquisitions.csv`: events acquired per HPO sample.
- `data/`: feature tables (parquet), not in git; see README "Data".
- `output/`: generated, not in git. Never edit it by hand; rerun the scripts.

## Names

The code's method names differ from the paper's: `new_method` = HPO,
`normal_method` = hematocrit, `prp` = PRP, `whole_blood` = WB. The cohort
`pasc` is the paper's Long COVID cohort. One pixel is 1 × 1 µm.

## Running

```sh
uv sync
uv run src/main.py      # ~10 s; needs data/ unpacked
```

Run scripts from the repository root, or anywhere: paths are resolved from
`utils.ROOT`.

## Ground rules

- Code may use only what is published: this repository and the feature tables
  on Zenodo. The raw IFC (HDF5) files are not distributed; anything needed from
  them is already in `donors.csv` and `acquisitions.csv`.
- Do not change a threshold, statistical test, correction family or donor set
  without saying so plainly: each of them changes numbers in the published
  paper.
- After any change to `src/utils.py` or `src/main.py`, rerun `main.py` and diff
  `output/*.csv` against the previous run.
- The analysis set is the 20 donors imaged under all four methods. Donor C63 is
  excluded because it has no `new_method` acquisition.
- The donor is the unit of analysis. Cells are never tested as independent
  observations.
