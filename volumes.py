import pandas as pd

df = pd.read_csv("donors.csv")
# The analysis set of main.py: the donors imaged under all four methods.
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
