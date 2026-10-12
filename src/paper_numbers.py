"""The in-text numbers of the paper that no table carries.

Everything here is descriptive, read from the per-donor CSVs that `main.py`
writes, `acquisitions.csv` and the feature tables, plus one simulation: the
smallest Cliff's delta the Mann-Whitney U test detects with 80% power at the
group sizes of the study. Run `main.py` first.

    uv run src/paper_numbers.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

import utils

METHOD = "new_method"
OUTPUT_DIR = utils.ROOT / "output"
#: Events acquired per HPO sample, read once from the `n_events` attribute of each
#: raw acquisition file, which is not distributed.
ACQUISITIONS_CSV = utils.ROOT / "acquisitions.csv"

#: Power simulation: two-sided alpha, target power, draws per effect size.
ALPHA, POWER, N_SIMULATIONS, SEED = 0.05, 0.80, 4000, 20260922


def common_donors() -> list[str]:
    donors = pd.read_csv(utils.DONORS_CSV)
    per_method = (set(group["donor"]) for _, group in donors.groupby("method"))
    return sorted(set.intersection(*per_method))


def acquisition() -> None:
    """Events per HPO sample and the acquired volumes."""
    acquisitions = pd.read_csv(ACQUISITIONS_CSV).query("method == @METHOD")
    events = acquisitions.set_index("donor")["n_events"].reindex(common_donors())
    volumes = pd.read_csv(utils.DONORS_CSV).query("method == @METHOD")["volume_uL"]
    print(
        f"HPO events per sample: median {np.median(events):.0f}, "
        f"range {events.min()} to {events.max()} (n={events.count()})"
    )
    print(f"HPO acquired volume: {volumes.min():.2f} to {volumes.max():.2f} uL")


def masks() -> None:
    """Mask and object counts, and the solidity percentiles behind the 0.88 rule."""
    donors = common_donors()
    n_masks = 0
    for method in utils.METHODS:
        for donor in donors:
            n_masks += pq.read_metadata(
                utils.DATA_DIR / method / f"{donor}.parquet"
            ).num_rows
    shared = sum(
        pq.read_metadata(f).num_rows
        for f in glob.glob(str(utils.DATA_DIR / "*/*.parquet"))
    )
    print(f"masks over the {len(donors)} participants, all methods: {n_masks:,}")
    print(f"masks in the shared feature files, every donor: {shared:,}")

    solidity = []
    for donor in donors:
        cols = utils._read(METHOD, donor, ("solidity", "bf_std_intensity"))
        solidity.append(cols["solidity"][cols["bf_std_intensity"] > utils.NOISE_BF_STD])
    solidity = np.concatenate(solidity)
    p5, p50 = np.percentile(solidity, [5, 50])
    kept = (solidity >= utils.MIN_RBC_SOLIDITY).mean()
    print(
        f"HPO solidity, masks above the noise floor: P5 {p5:.2f}, P50 {p50:.2f}; "
        f">= {utils.MIN_RBC_SOLIDITY} keeps {kept:.1%}"
    )


def per_donor_summaries() -> None:
    """Median over donors of the per-donor areas and marker intensities."""
    endpoints = pd.read_csv(OUTPUT_DIR / "endpoints_per_donor.csv")
    area = endpoints.groupby(["method", "population"])["area_um2_median"].median()
    print("median object area (um^2), median over donors:")
    print(area.unstack().round(2).to_string())
    staining = pd.read_csv(OUTPUT_DIR / "staining_per_donor.csv").query(
        "statistic == 'median'"
    )
    intensity = staining.groupby(["method", "population", "marker"])["value"].median()
    print("median marker intensity (a.u.), median over donors:")
    print(intensity.unstack().round(1).to_string())


def detectable_delta() -> None:
    """Smallest Cliff's delta with `POWER` power, for a normal location shift.

    For two normals one standard deviation apart by `shift`, the population
    Cliff's delta is 2 * Phi(shift / sqrt(2)) - 1.
    """
    per_donor = pd.read_csv(OUTPUT_DIR / "endpoints_per_donor.csv").query(
        "method == @METHOD and population == 'platelet'"
    )
    n_control = (per_donor["cohort"] == "control").sum()
    n_pasc = (per_donor["cohort"] == "pasc").sum()
    rng = np.random.default_rng(SEED)
    for shift in np.arange(0.5, 3.01, 0.05):
        control = rng.standard_normal((N_SIMULATIONS, n_control))
        pasc = rng.standard_normal((N_SIMULATIONS, n_pasc)) + shift
        p = stats.mannwhitneyu(control, pasc, axis=1).pvalue
        if (p < ALPHA).mean() >= POWER:
            delta = 2 * stats.norm.cdf(shift / np.sqrt(2)) - 1
            print(
                f"{POWER:.0%} power at alpha {ALPHA} with n = {n_control} vs "
                f"{n_pasc}: Cliff's delta >= {delta:.2f} (shift {shift:.2f} SD)"
            )
            return


if __name__ == "__main__":
    acquisition()
    masks()
    per_donor_summaries()
    detectable_delta()
