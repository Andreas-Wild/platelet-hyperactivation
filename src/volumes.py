"""Mean and SD of the imaged sample volume per method and cohort, as LaTeX rows.

Prints one `method&cohort&mean&sd` row each, for the volume table of the paper.
Covers the analysis set of `main.py`: the donors imaged under all four methods.

    uv run src/volumes.py
"""

import pandas as pd

import utils

df = pd.read_csv(utils.DONORS_CSV)
donors_per_method = (set(group["donor"]) for _, group in df.groupby("method"))
df = df[df["donor"].isin(set.intersection(*donors_per_method))]

methods = {
"new_method": df[df["method"]=="new_method"],
"normal_method": df[df["method"]=="normal_method"],
"prp": df[df["method"]=="prp"],
"whole_blood": df[df["method"]=="whole_blood"]
}

def calc(df, cohort) -> tuple[float, float]:
     df_cohort = df[df["cohort"]==cohort]
     return (df_cohort["volume_uL"].mean(), df_cohort["volume_uL"].std())

for method, df in methods.items():
    for cohort in ["control", "pasc"]:
        mu, std = calc(df, cohort)
        print(f"{method}&{cohort}&{mu:.3g}&{std:.3g}")
