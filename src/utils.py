"""Loading and population classification for the platelet study.

Implements the Methodology section of the paper: the population classifier and
the per-donor collapse of cell features. The statistics live in `main.py`, next
to the analysis that uses them.

Only the columns each step needs are read from the parquet files; the remaining
79 feature columns stay on disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: The repository root; this file lives in `src/`.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DONORS_CSV = ROOT / "donors.csv"

#: The four physical sample-preparation methods.
METHODS = ("new_method", "normal_method", "prp", "whole_blood")

#: The three target populations. A fourth, `irregular`, holds everything that is
#: not one of these and is excluded from every endpoint.
POPULATIONS = ("platelet", "rbc", "aggregate")
_POPULATION_LEVELS = ("irregular", *POPULATIONS)

#: Platelet-activation markers, and the raw feature each is measured on.
MARKERS = {"PAC1": "ch2_median_intensity", "CD62": "ch3_median_intensity"}

#: The FlowSight's 20x objective gives a nominal 1 x 1 um pixel, so 1 px = 1 um^2.
UM2_PER_PX = 1.0

# --- Classifier thresholds (Methodology, "Data Analysis") --------------------
NOISE_BF_STD = 8.0  # masks at or below this are background, not cells
# Size cuts lie between the median platelet mask (~5 px equivalent diameter) and
# the median RBC mask (~9 px); brightfield masks are larger than the cells.
PLATELET_MAX_DIAMETER = 8.0  # px
PLATELET_MAX_MAJOR_AXIS = 7.0  # px; catches edge-on RBCs that look small by area
MIN_RBC_SOLIDITY = 0.88
MAX_RBC_ECCENTRICITY = 0.85
MIN_RBC_CD235 = 1000.0  # raw median CD235 (ch7) marking the RBC membrane
FLUOR_GATE_PERCENTILE = 99.0  # marker positivity, against the RBC internal negative

_SHAPE_COLUMNS = (
    "equivalent_diameter",
    "major_axis_length",
    "bf_std_intensity",
    "solidity",
    "eccentricity",
    "ch7_median_intensity",
    *MARKERS.values(),
)
_OBJECT_COLUMNS = (
    *_SHAPE_COLUMNS,
    "event_id",
    "instance_number",
    "connected_to",
    "area",
)


# --- Reading ----------------------------------------------------------------


def _read(method: str, donor: str, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Read only `columns` for one donor under one method.

    `connected_to` is a list column; it is returned decomposed into the
    per-mask neighbour counts and the flat neighbour values.
    """
    table = pq.read_table(DATA_DIR / method / f"{donor}.parquet", columns=list(columns))
    out: dict[str, np.ndarray] = {}
    for name in columns:
        column = table.column(name).combine_chunks()
        if name == "connected_to":
            lengths = pc.fill_null(pc.list_value_length(column), 0)
            out["neighbour_counts"] = lengths.to_numpy(zero_copy_only=False)
            out["neighbours"] = column.flatten().to_numpy(zero_copy_only=False)
        else:
            out[name] = column.to_numpy(zero_copy_only=False)
    return out


# --- Population classification ----------------------------------------------


def _classify_masks(
    cols: dict[str, np.ndarray], pac1_gate: float, cd62_gate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mask `real`, `rbc_like` and `platelet_like` boolean arrays.

    RBC-like: cell-sized on either size measure *or* carrying a strong CD235
    membrane signal, and compact and round. Platelet-like: small on both size
    measures, CD235-negative, not already RBC-like, and either round or carrying
    genuine PAC1/CD62 positivity. That last clause only ever rescues a platelet,
    so resting (marker-negative) platelets are kept. Infinite gates switch the
    rescue off, which is how `marker_gates` finds RBCs before any gate exists.
    """
    diameter = cols["equivalent_diameter"]
    major_axis = cols["major_axis_length"]
    # Masks below the brightfield noise floor are the model's attempts at
    # segmenting empty background, not cells.
    real = cols["bf_std_intensity"] > NOISE_BF_STD
    cd235_high = cols["ch7_median_intensity"] >= MIN_RBC_CD235
    cell_sized = (diameter >= PLATELET_MAX_DIAMETER) | (
        major_axis >= PLATELET_MAX_MAJOR_AXIS
    )

    rbc_like = (
        real
        & (cell_sized | cd235_high)
        & (cols["solidity"] >= MIN_RBC_SOLIDITY)
        & (cols["eccentricity"] <= MAX_RBC_ECCENTRICITY)
    )
    marker_positive = (cols[MARKERS["PAC1"]] >= pac1_gate) | (
        cols[MARKERS["CD62"]] >= cd62_gate
    )
    platelet_like = (
        real
        & (major_axis < PLATELET_MAX_MAJOR_AXIS)
        & (diameter < PLATELET_MAX_DIAMETER)
        & ~cd235_high
        & ~rbc_like
        & ((cols["eccentricity"] <= MAX_RBC_ECCENTRICITY) | marker_positive)
    )
    return real, rbc_like, platelet_like


def marker_gates(method: str, donors: Sequence[str]) -> dict[str, float]:
    """PAC1/CD62 positivity thresholds for one method.

    RBCs express neither marker, so the 99th percentile of each marker among a
    method's RBC-like masks is its non-specific ceiling. The gate is a property
    of the method, so it is pooled over all of that method's donors.
    """
    rbc_intensities: dict[str, list[np.ndarray]] = {name: [] for name in MARKERS}
    for donor in donors:
        cols = _read(method, donor, _SHAPE_COLUMNS)
        _, rbc_like, _ = _classify_masks(cols, np.inf, np.inf)
        for name, column in MARKERS.items():
            rbc_intensities[name].append(cols[column][rbc_like])
    return {
        name: float(np.percentile(np.concatenate(values), FLUOR_GATE_PERCENTILE))
        for name, values in rbc_intensities.items()
    }


def object_labels(cols: dict[str, np.ndarray]) -> np.ndarray:
    """Each mask's object label; masks at most one pixel apart share an object.

    Resolves `connected_to` into a graph whose connected components are the
    objects. A mask is addressed by (event, instance); packing that into one
    sortable key lets neighbour references resolve to row positions with a
    binary search.
    """
    event_id = cols["event_id"]
    instance_number = cols["instance_number"]
    n_masks = event_id.size
    stride = int(instance_number.max()) + 1
    keys = event_id.astype(np.int64) * stride + instance_number
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    sources = np.repeat(np.arange(n_masks), cols["neighbour_counts"])
    target_keys = event_id[sources].astype(np.int64) * stride + cols["neighbours"]
    positions = np.searchsorted(sorted_keys, target_keys)
    if not np.array_equal(sorted_keys[positions], target_keys):
        raise ValueError("connected_to references a mask that is not in the file")

    graph = coo_matrix(
        (np.ones(sources.size, dtype=np.int8), (sources, order[positions])),
        shape=(n_masks, n_masks),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels


def object_populations(
    cols: dict[str, np.ndarray], gates: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Each mask's object label, and each object's population code.

    An object of two or more real masks is an aggregate; an object of one real
    mask takes that mask's population. Codes index `_POPULATION_LEVELS`.
    """
    real, rbc_like, platelet_like = _classify_masks(cols, gates["PAC1"], gates["CD62"])
    labels = object_labels(cols)
    n_objects = int(labels.max()) + 1

    def total(weights: np.ndarray) -> np.ndarray:
        return np.bincount(labels, weights=weights, minlength=n_objects)

    n_real, n_rbc, n_platelet = total(real), total(rbc_like), total(platelet_like)

    codes = np.zeros(n_objects, dtype=np.int8)  # 0 = irregular
    single = n_real == 1
    codes[single & (n_platelet == 1)] = 1
    codes[single & (n_rbc == 1)] = 2
    codes[n_real >= 2] = 3
    return labels, codes


def object_table(method: str, donor: str, gates: dict[str, float]) -> pd.DataFrame:
    """One row per segmented object for a donor, with its population.

    Masks lying at most one pixel apart are grouped into a single object. An
    object of two or more real masks is an aggregate; an object of one real mask
    takes that mask's population. An object's area is the summed footprint of
    its masks (an aggregate's true size); its marker intensity is the mean over
    its masks, so a many-cell clump is not over-weighted by its own mask count.
    """
    cols = _read(method, donor, _OBJECT_COLUMNS)
    labels, codes = object_populations(cols, gates)
    n_objects = codes.size

    def total(weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(labels, weights=weights, minlength=n_objects)

    masks_per_object = total()

    return pd.DataFrame(
        {
            "population": pd.Categorical.from_codes(codes, _POPULATION_LEVELS),
            "area_um2": total(cols["area"]) * UM2_PER_PX,
            **{
                name: total(cols[column]) / masks_per_object
                for name, column in MARKERS.items()
            },
        }
    )


# --- Collapsing to one number per donor -------------------------------------


def donor_endpoints(objects: pd.DataFrame, volume_uL: float) -> pd.DataFrame:
    """The primary and three exploratory endpoints, per population.

    The donor is the unit of analysis, so each population's objects collapse to
    a single number per endpoint. Percentages are of all cells, i.e. excluding
    the irregular population.
    """
    cells = objects[objects["population"] != "irregular"]
    n_cells = len(cells)
    grouped = cells.groupby("population", observed=False)
    counts = grouped.size().reindex(POPULATIONS).to_numpy()
    median_areas = grouped["area_um2"].median().reindex(POPULATIONS).to_numpy()
    return pd.DataFrame(
        {
            "population": POPULATIONS,
            "obj_per_uL": counts / volume_uL,
            "count": counts,
            "pct_total": 100.0 * counts / n_cells,
            "area_um2_median": median_areas,
        }
    )


def donor_staining(objects: pd.DataFrame) -> pd.DataFrame:
    """Per-donor marker summaries, one row per population x marker x statistic."""
    rows = []
    for population, group in objects.groupby("population", observed=True):
        if population == "irregular":
            continue
        for marker in MARKERS:
            values = group[marker].to_numpy()
            for statistic, value in (
                ("median", np.median(values)),
                ("p95", np.percentile(values, 95)),
            ):
                rows.append(
                    {
                        "population": population,
                        "marker": marker,
                        "statistic": statistic,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)
