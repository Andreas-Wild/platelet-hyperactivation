"""Representative events of each population, for the paper's example figures.

Two steps, both driven from the command line:

    uv run representative_cells.py candidates   # contact sheets to choose from
    uv run representative_cells.py figure       # the three 5 x 5 figures
    uv run representative_cells.py example      # the paper's single-event example

`candidates` scores every classified object under `METHOD` for how clearly it
shows its class, keeps the best `POOL` per class, and deals `N_CANDIDATES` of
them out over donors and cohorts. Each candidate is drawn as one strip of the
five panels the paper shows -- brightfield, segmentation, PAC-1, CD62P and
CD235a -- and labelled with an id `<donor>:<event>:<instances>`.

`figure` draws the ids listed in `PICKS` as one figure per class: five events
as rows, the five panels as columns. Swap ids in `PICKS` and re-run to change a
row. Classes left empty in `PICKS` fall back to the first five candidates.

Every tile of a channel shares one fixed display window, so a bright tile is a
bright event, not a different contrast stretch. Fluorescence is shown on an
arcsinh scale, so background and dim cells stay visible beside bright ones.
The windows are taken over the candidate pool and written next to the sheets.

The choice is deliberately for clarity, not a random sample: the captions must
say the events were selected as representative, as the validation set is the
random one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.patches import Rectangle

import utils
from validation_sample import MASK_RUN, find_h5

METHOD = "new_method"
SEED = 20260922

#: Best-scoring objects per class kept before dealing out candidates.
POOL = 600
N_CANDIDATES = 48

#: Native tile side per class, in pixels. Each class is cropped as tight as
#: its largest members allow, so a platelet is not lost in empty field; every
#: figure carries its own scale bar. Events are 45 px tall, the upper bound.
TILE = {"platelet": 24, "rbc": 28, "aggregate": 40}
#: Objects must sit this many pixels clear of the event border, so no cell is
#: cut off by the camera frame.
EDGE_MARGIN = 2

#: Channel order inside `events/<event_id>`.
BF, PAC1, CD62, SSC, CD235 = range(5)
PANELS = ("Brightfield", "Segmentation", "PAC-1", "CD62P", "CD235a")
FLUOR = {"PAC-1": PAC1, "CD62P": CD62, "CD235a": CD235}
#: One colour map per marker, so the channels are told apart at a glance.
FLUOR_CMAP = {"PAC-1": "viridis", "CD62P": "magma", "CD235a": "inferno"}

#: 0.5 um per pixel, so 10 px is 5 um.
SCALE_BAR_PX = 10

PICKS: dict[str, list[str]] = {
    "platelet": [
        "C54:59087:1",
        "C55:62396:1",
        "C57:170709:1",
        "LV976:40366:1",
        "LV971:20397:1",
    ],
    "rbc": [
        "C58:178665:1",
        "LV972:137605:1",
        "C57:208339:1",
        "C60:161450:1",
        "LV977:221811:1",
    ],
    "aggregate": [
        "LV972:266290:1+2",
        "C65:190832:1+2+3",
        "C57:124215:1+2",
        "LV971:178326:1+2+3",
        "LV977:274689:1+2+3",
    ],
}

OUTPUT_DIR = utils.ROOT / "output" / "representative"
FIGURE_DIR = Path.home() / "Documents/platelet_paper/report/figures/images"

_EXTRA = (
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "ch2_max_intensity",
    "ch3_max_intensity",
    "ch7_max_intensity",
)


# --- Scoring -----------------------------------------------------------------


def donor_objects(donor: str, gates: dict[str, float]) -> pd.DataFrame:
    """Every object of one donor, with the features the scores need."""
    cols = utils._read(METHOD, donor, (*utils._OBJECT_COLUMNS, *_EXTRA))
    labels, codes = utils.object_populations(cols, gates)
    n = codes.size
    real = cols["bf_std_intensity"] > utils.NOISE_BF_STD

    def agg(ufunc, values, fill):
        out = np.full(n, fill, dtype=np.float64)
        ufunc.at(out, labels, values)
        return out

    x0 = agg(np.minimum, cols["bbox_x"], np.inf)
    y0 = agg(np.minimum, cols["bbox_y"], np.inf)
    x1 = agg(np.maximum, cols["bbox_x"] + cols["bbox_w"], 0)
    y1 = agg(np.maximum, cols["bbox_y"] + cols["bbox_h"], 0)
    event_id = np.zeros(n, dtype=np.int64)
    event_id[labels] = cols["event_id"]

    order = np.argsort(labels, kind="stable")
    instances = (
        pd.Series(cols["instance_number"][order])
        .groupby(labels[order])
        .apply(lambda g: " ".join(str(int(v)) for v in sorted(g)))
    )
    n_real = np.bincount(labels, weights=real, minlength=n)
    # Objects that are cells; the event is clean when it holds only this one.
    cells_per_event = pd.Series(n_real > 0).groupby(event_id).transform("sum")

    frame = pd.DataFrame(
        {
            "donor": donor,
            "population": pd.Categorical.from_codes(codes, utils._POPULATION_LEVELS),
            "event_id": event_id,
            "instances": instances.to_numpy(),
            "n_masks": np.bincount(labels, minlength=n),
            "n_real": n_real,
            "cells_in_event": cells_per_event.to_numpy(),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "area": np.bincount(labels, weights=cols["area"], minlength=n),
            "bf_std": agg(np.maximum, cols["bf_std_intensity"], 0),
            "eccentricity": agg(np.maximum, cols["eccentricity"], 0),
            "solidity": agg(np.minimum, cols["solidity"], np.inf),
            "diameter": agg(np.maximum, cols["equivalent_diameter"], 0),
            "pac1": agg(np.maximum, cols["ch2_median_intensity"], -np.inf),
            "cd62": agg(np.maximum, cols["ch3_median_intensity"], -np.inf),
            "cd235": agg(np.maximum, cols["ch7_median_intensity"], -np.inf),
            "cd235_min": agg(np.minimum, cols["ch7_median_intensity"], np.inf),
        }
    )
    return frame[frame["population"] != "irregular"]


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True)


def score(objects: pd.DataFrame, gates: dict[str, float]) -> pd.DataFrame:
    """Keep clean, well-framed objects and score each for clarity within its class."""
    frame_h, frame_w = 45, 60
    clean = (
        (objects["cells_in_event"] == 1)
        & (objects["x0"] >= EDGE_MARGIN)
        & (objects["y0"] >= EDGE_MARGIN)
        & (objects["x1"] <= frame_w - EDGE_MARGIN)
        & (objects["y1"] <= frame_h - EDGE_MARGIN)
    )
    room = objects["population"].map(TILE).astype(float) - 6
    clean &= (objects["x1"] - objects["x0"] <= room) & (
        objects["y1"] - objects["y0"] <= room
    )
    objects = objects[clean].copy()
    marker = np.maximum(
        objects["pac1"] / gates["PAC1"], objects["cd62"] / gates["CD62"]
    )
    objects["marker"] = marker

    scored = []
    for population, group in objects.groupby("population", observed=True):
        if population == "platelet":
            # Big enough to see, and visibly marker-positive.
            group = group[(group["diameter"] >= 4) & (group["marker"] >= 1)]
            s = _rank(group["marker"]) + _rank(group["bf_std"]) + _rank(group["area"])
        elif population == "rbc":
            # Face-on discs with a strong membrane stain and no activation signal.
            group = group[
                (group["eccentricity"] <= 0.6)
                & (group["solidity"] >= 0.93)
                & (group["cd235"] >= utils.MIN_RBC_CD235)
            ]
            s = (
                _rank(group["cd235"])
                + _rank(group["bf_std"])
                - _rank(group["eccentricity"])
                - _rank(group["marker"])
            )
        else:
            # Compact clumps of a few cells, ideally a platelet on an RBC, so
            # the channels show which member is which.
            group = group[group["n_real"].between(2, 6)]
            mixed = (group["cd235"] >= utils.MIN_RBC_CD235) & (group["marker"] >= 1)
            s = _rank(group["marker"]) + _rank(group["bf_std"]) + mixed.astype(float)
        scored.append(group.assign(score=s))
    return pd.concat(scored)


def candidates(objects: pd.DataFrame, cohorts: pd.Series) -> pd.DataFrame:
    """Top `POOL` per class, dealt round-robin over cohort-interleaved donors."""
    rng = np.random.default_rng(SEED)
    chosen = []
    for _, group in objects.groupby("population", observed=True):
        pool = group.nlargest(POOL, "score")
        pools = {
            donor: list(rng.permutation(g.index.to_numpy()))
            for donor, g in pool.groupby("donor")
        }
        order = sorted(pools, key=lambda d: (cohorts[d], rng.uniform()))
        # Interleave cohorts so any prefix of the list is balanced.
        by_cohort = [
            [d for d in order if cohorts[d] == c] for c in sorted(set(cohorts[order]))
        ]
        order = [d for pair in zip(*by_cohort, strict=False) for d in pair] + [
            d for group_ in by_cohort for d in group_[min(map(len, by_cohort)) :]
        ]
        picked = []
        while len(picked) < N_CANDIDATES and any(pools.values()):
            for donor in order:
                if pools[donor] and len(picked) < N_CANDIDATES:
                    picked.append(pools[donor].pop())
        chosen.append(pool.loc[picked].sort_values("score", ascending=False))
    out = pd.concat(chosen)
    out["cohort"] = out["donor"].map(cohorts)
    out["id"] = (
        out["donor"]
        + ":"
        + out["event_id"].astype(str)
        + ":"
        + out["instances"].str.replace(" ", "+")
    )
    return out


# --- Cutting tiles -----------------------------------------------------------


def cut(row: pd.Series, events: h5py.Group, masks: h5py.Group) -> dict[str, np.ndarray]:
    """Square crops of every channel and of the mask labels, object centred.

    The side is `TILE` of the object's class.
    """
    tile_side = TILE[row["population"]]
    key = str(int(row["event_id"]))
    stack = np.asarray(events[key], dtype=np.float32)
    labelled = np.asarray(masks[key])
    cy = int(round((row["y0"] + row["y1"]) / 2 - tile_side / 2))
    cx = int(round((row["x0"] + row["x1"]) / 2 - tile_side / 2))
    h, w = labelled.shape
    sy0, sx0, sy1, sx1 = (
        max(cy, 0),
        max(cx, 0),
        min(cy + tile_side, h),
        min(cx + tile_side, w),
    )
    dy, dx = sy0 - cy, sx0 - cx

    tiles = {}
    for name, channel in (
        ("bf", BF),
        ("PAC-1", PAC1),
        ("CD62P", CD62),
        ("CD235a", CD235),
    ):
        plane = stack[channel]
        # Pad with the event's own background so the tile reads as one field.
        tile = np.full((tile_side, tile_side), np.median(plane), dtype=np.float32)
        tile[dy : dy + sy1 - sy0, dx : dx + sx1 - sx0] = plane[sy0:sy1, sx0:sx1]
        tiles[name] = tile
    label_tile = np.zeros((tile_side, tile_side), dtype=labelled.dtype)
    label_tile[dy : dy + sy1 - sy0, dx : dx + sx1 - sx0] = labelled[sy0:sy1, sx0:sx1]
    tiles["labels"] = label_tile
    return tiles


def cut_all(rows: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for donor, group in rows.groupby("donor"):
        with h5py.File(find_h5(donor, METHOD), "r") as handle:
            for _, row in group.iterrows():
                out[row["id"]] = cut(row, handle["events"], handle[f"masks/{MASK_RUN}"])
    return out


def display_windows(tiles: dict[str, dict[str, np.ndarray]]) -> dict[str, list[float]]:
    """One fixed display range per channel, shared by every tile and figure.

    Brightfield is linear, `[low, high]`. Fluorescence is `[low, high, cofactor]`
    on an arcsinh scale: linear up to about `cofactor` above `low`, logarithmic
    beyond. Most tile pixels are background, so the median sets the cofactor and
    the background lands as a faint texture rather than black, while the
    brightest cells still fit in the range.
    """
    windows = {}
    for name in ("bf", *FLUOR):
        pixels = np.concatenate([t[name].ravel() for t in tiles.values()])
        if name == "bf":
            low, high = np.percentile(pixels, [1, 99.8])
            windows[name] = [float(low), float(high)]
        else:
            low, background, high = np.percentile(pixels, [0.5, 50, 99.8])
            windows[name] = [float(low), float(high), float(background - low)]
    return windows


# --- Drawing -----------------------------------------------------------------


def _norm(tile: np.ndarray, window: list[float]) -> np.ndarray:
    """Map raw counts to 0-1: linear for `[low, high]`, arcsinh with a cofactor."""
    if len(window) == 2:
        low, high = window
        return np.clip((tile - low) / (high - low), 0, 1)
    low, high, cofactor = window
    shifted = np.clip(tile - low, 0, None)
    return np.clip(
        np.arcsinh(shifted / cofactor) / np.arcsinh((high - low) / cofactor), 0, 1
    )


def panel_images(tiles: dict[str, np.ndarray], windows: dict) -> list[np.ndarray]:
    """The five RGB panels of one event."""
    grey = _norm(tiles["bf"], windows["bf"])
    bf = np.repeat(grey[..., None], 3, axis=2)

    # Each mask instance gets its own colour, blended over the brightfield, with
    # a solid one-pixel boundary so touching instances stay distinguishable.
    seg = bf.copy()
    palette = colormaps["tab10"].colors
    labels = tiles["labels"]
    for i, value in enumerate(v for v in np.unique(labels) if v):
        member = labels == value
        colour = np.array(palette[i % len(palette)])
        seg[member] = 0.55 * seg[member] + 0.45 * colour
        p = np.pad(member, 1)
        edge = member & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
        seg[edge] = colour

    fluor = [
        colormaps[FLUOR_CMAP[name]](_norm(tiles[name], windows[name]))[..., :3]
        for name in FLUOR
    ]
    return [bf, seg, *fluor]


def _scale_bar(ax, side: int, colour="black") -> None:
    ax.add_patch(
        Rectangle(
            (side - 2 - SCALE_BAR_PX, side - 2.5),
            SCALE_BAR_PX,
            side / 30,
            color=colour,
            linewidth=0,
        )
    )


def contact_sheet(rows: pd.DataFrame, tiles: dict, windows: dict, path: Path) -> None:
    """All candidates of one class, three strips of five panels per row."""
    per_row = 3
    n_rows = int(np.ceil(len(rows) / per_row))
    fig, axes = plt.subplots(
        n_rows,
        per_row * 5,
        figsize=(per_row * 5 * 1.0, n_rows * 1.12),
        gridspec_kw={"wspace": 0.04, "hspace": 0.32},
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (_, row) in enumerate(rows.iterrows()):
        r, c = divmod(k, per_row)
        for j, image in enumerate(panel_images(tiles[row["id"]], windows)):
            ax = axes[r, c * 5 + j]
            ax.imshow(image, interpolation="nearest")
            if j == 0:
                ax.set_title(f"{k + 1}  {row['id']}", fontsize=6, loc="left")
    fig.suptitle(
        f"{rows['population'].iloc[0]} candidates -- columns per strip: "
        + ", ".join(PANELS),
        fontsize=9,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def class_figure(ids: list[str], tiles: dict, windows: dict, path: Path) -> None:
    """Five events as rows, the five panels as columns."""
    fig, axes = plt.subplots(
        len(ids),
        5,
        figsize=(5 * 1.1, len(ids) * 1.1),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
    )
    for r, event in enumerate(ids):
        for j, image in enumerate(panel_images(tiles[event], windows)):
            ax = axes[r, j]
            ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(PANELS[j], fontsize=8)
            if j == 0:
                _scale_bar(ax, image.shape[0])
    for suffix in (".png", ".pdf"):
        fig.savefig(
            path.with_suffix(suffix), dpi=600, bbox_inches="tight", pad_inches=0.02
        )
    plt.close(fig)


# --- Running -----------------------------------------------------------------


def build_candidates() -> None:
    donors = pd.read_csv(utils.DONORS_CSV)
    cohorts = donors.drop_duplicates("donor").set_index("donor")["cohort"]
    common = sorted(set(donors.loc[donors["method"] == METHOD, "donor"]))
    gates = utils.marker_gates(METHOD, common)

    objects = pd.concat((donor_objects(d, gates) for d in common), ignore_index=True)
    pool = candidates(score(objects, gates), cohorts)
    tiles = cut_all(pool)
    windows = display_windows(tiles)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUTPUT_DIR / "candidates.csv", index=False)
    (OUTPUT_DIR / "windows.json").write_text(json.dumps(windows, indent=2))
    for population, rows in pool.groupby("population", observed=True):
        contact_sheet(rows, tiles, windows, OUTPUT_DIR / f"candidates_{population}.png")
    print(pool.groupby(["population", "cohort"], observed=True).size().unstack())
    print(f"written to {OUTPUT_DIR}")


def build_figures() -> None:
    pool = pd.read_csv(OUTPUT_DIR / "candidates.csv", dtype={"instances": str})
    windows = json.loads((OUTPUT_DIR / "windows.json").read_text())
    picks = {
        population: PICKS[population]
        or pool.loc[pool["population"] == population, "id"].head(5).tolist()
        for population in utils.POPULATIONS
    }
    wanted = pool[pool["id"].isin([i for ids in picks.values() for i in ids])]
    missing = {i for ids in picks.values() for i in ids} - set(wanted["id"])
    if missing:
        raise KeyError(f"not in candidates.csv: {sorted(missing)}")
    tiles = cut_all(wanted)
    for population, ids in picks.items():
        class_figure(ids, tiles, windows, FIGURE_DIR / f"representative_{population}")
        print(population, ids)
    print(f"written to {FIGURE_DIR}")


#: The single event of the paper's fluorescence example figure, `fig:fluor_examples`.
EXAMPLE_DONOR, EXAMPLE_EVENT = "C55", 115162
EXAMPLE_FILES = {"PAC-1": "pac", "CD62P": "cd62", "CD235a": "cd235"}


def build_example() -> None:
    """Redraw the fluorescence panels of the example event in the `FLUOR_CMAP` maps.

    Unlike the representative figures, each panel is stretched linearly over its
    own min-max range, as the original panels were, for the look of a single
    event. Panels stay at native resolution with the same 5 um scale bar, rows
    47-48 and columns 47-56, so the brightfield panel beside them is unchanged.
    """
    with h5py.File(find_h5(EXAMPLE_DONOR, METHOD), "r") as handle:
        stack = np.asarray(handle["events"][str(EXAMPLE_EVENT)], dtype=np.float32)
    for name, channel in FLUOR.items():
        plane = stack[channel]
        window = [float(plane.min()), float(plane.max())]
        rgb = colormaps[FLUOR_CMAP[name]](_norm(plane, window))[..., :3]
        rgb[47:49, 47 : 47 + SCALE_BAR_PX] = 1.0
        path = (
            FIGURE_DIR / f"{EXAMPLE_FILES[name]}_rbc_and_platelet_data_c55_115162.png"
        )
        plt.imsave(path, rgb)
        print(name, FLUOR_CMAP[name], window, path.name)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "candidates"
    {
        "candidates": build_candidates,
        "figure": build_figures,
        "example": build_example,
    }[step]()
