"""Draw a blinded validation set of brightfield crops for human classification.

The classifier in `utils.py` assigns every segmented object to platelet, rbc,
aggregate or irregular from its shape and marker features. This script samples
objects it has already classified, cuts each one out of the raw brightfield
channel at native resolution, and lays them out as a numbered collage that
fellow students can label by eye. Comparing their labels with the model's is
the human validation reported in the paper.

Sampling (all under `METHOD`, the method the paper reports on):

  * stratified by population -- `QUOTAS` objects of each class, so aggregates
    (rare in a natural sample) are represented well enough to score;
  * spread over donors and cohorts -- donors are visited in a cohort-interleaved
    random order, at most one object per donor per pass, so no donor or cohort
    dominates a class;
  * blinded -- the 50 tiles are shuffled before numbering, so tile order carries
    no information about the model's answer.

Everything is driven by `SEED`, so the same 50 objects come back on a re-run.

    uv run validation_sample.py

Writes to `output/validation/`:

    collage.png            the 50 numbered tiles, native resolution
    collage_x6.png         the same, nearest-neighbour x6 for on-screen work
    collage_outlined.png   native resolution, object outlined in red
    collage_outlined_x6.png
    tiles/NN.png           each tile on its own, native resolution
    replaced.csv           draws discarded for image quality, and their class
    response_sheet.csv     hand this out with the collage
    answer_key.csv         the model's answer -- do NOT hand this out
    README.txt             instructions for the students
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import utils

# --- What to sample ----------------------------------------------------------

#: The method the paper reports on. The marker gates are a property of the
#: method, so a validation set mixing methods would be validating four
#: different classifiers at once.
METHOD = "new_method"

#: How many objects of each class. Balanced rather than natural: aggregates are
#: the rarest call and a natural sample would contain too few to score.
QUOTAS = {"platelet": 17, "rbc": 17, "aggregate": 16}

SEED = 20260824

#: Tiles rejected on sight for image quality, replaced by the next draw of the
#: same class from the same pool. Replacements keep the rejected tile's number,
#: so labelling already done against an earlier collage stays valid, and a
#: number may appear more than once to redraw the same slot again. Every
#: discard is written to `replaced.csv`: these are faint objects, which is where
#: the classifier is likeliest to be wrong, so dropping them flatters the
#: agreement score and the paper should report how many were dropped.
REPLACED = (4, 15, 17, 24, 29, 17, 24, 29, 17)

# --- Where the images live ---------------------------------------------------

#: The raw acquisitions sit beside this repository, one directory per method.
IMAGE_ROOT = utils.ROOT.parent
METHOD_DIRS = {
    "new_method": "data",
    "normal_method": "old_data",
    "prp": "prp",
    "whole_blood": "whole_blood",
}
MASK_RUN = "cyto3_best"
#: Channel order inside `events/<event_id>`, from experiment.json: bf first.
BRIGHTFIELD = 0

# --- How the tiles look ------------------------------------------------------

#: Native pixels. Covers the 95th percentile aggregate (~22 x 33 px) with room
#: to spare; events are 60 px wide, so every tile is padded sideways.
TILE = 72

#: Display window, in raw brightfield counts, fixed for every tile. Nothing is
#: levelled between donors or events: an acquisition whose background sat
#: brighter renders brighter here, exactly as it came off the instrument. The
#: window is wide enough to hold every donor's background and the darkest
#: aggregate without clipping either end.
DISPLAY_LOW = 230.0
DISPLAY_HIGH = 850.0

COLUMNS, ROWS = 10, 5
GUTTER = 4
LABEL_HEIGHT = 14

PAPER = (255, 255, 255)
INK = (0, 0, 0)
OUTLINE = (220, 30, 30)

OUTPUT_DIR = utils.ROOT / "output" / "validation"

_BBOX_COLUMNS = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
_SAMPLE_COLUMNS = (*utils._OBJECT_COLUMNS, *_BBOX_COLUMNS)


# --- Finding the objects -----------------------------------------------------


def find_h5(donor: str, method: str = METHOD) -> Path:
    """The single acquisition file for one donor under one method."""
    sample_dir = IMAGE_ROOT / METHOD_DIRS[method] / donor
    candidates = sorted(sample_dir.glob("*.h5"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one *.h5 in {sample_dir}")
    return candidates[0]


def donor_objects(donor: str, gates: dict[str, float]) -> pd.DataFrame:
    """Every classified object for one donor, with the masks it is made of.

    One row per object: its population, its event, the mask instances inside it
    and the bounding box of their union -- enough to cut it out of the raw image.
    """
    cols = utils._read(METHOD, donor, _SAMPLE_COLUMNS)
    labels, codes = utils.object_populations(cols, gates)
    n_objects = codes.size

    # An object never spans events, so any member mask's event identifies it.
    event_id = np.zeros(n_objects, dtype=np.int64)
    event_id[labels] = cols["event_id"]

    # Union of the member masks' boxes.
    x0 = np.full(n_objects, np.iinfo(np.int64).max)
    y0 = np.full(n_objects, np.iinfo(np.int64).max)
    x1 = np.zeros(n_objects, dtype=np.int64)
    y1 = np.zeros(n_objects, dtype=np.int64)
    np.minimum.at(x0, labels, cols["bbox_x"])
    np.minimum.at(y0, labels, cols["bbox_y"])
    np.maximum.at(x1, labels, cols["bbox_x"] + cols["bbox_w"])
    np.maximum.at(y1, labels, cols["bbox_y"] + cols["bbox_h"])

    order = np.argsort(labels, kind="stable")
    members = pd.Series(cols["instance_number"][order]).groupby(labels[order])
    instances = members.apply(
        lambda group: " ".join(str(int(value)) for value in sorted(group))
    )

    return pd.DataFrame(
        {
            "population": pd.Categorical.from_codes(codes, utils._POPULATION_LEVELS),
            "event_id": event_id,
            "instances": instances.to_numpy(),
            "n_masks": np.bincount(labels, minlength=n_objects),
            "bbox_x": x0,
            "bbox_y": y0,
            "bbox_w": x1 - x0,
            "bbox_h": y1 - y0,
        }
    )


def cohort_interleaved(donors: pd.Series, rng: np.random.Generator) -> list[str]:
    """Donor visiting order in which every prefix is roughly cohort-balanced.

    Each donor is placed at its rank within its own cohort, jittered, so the
    two cohorts are dealt out proportionally however far down the order the
    quota happens to reach.
    """
    positions = {}
    for _, group in donors.groupby(donors):
        members = rng.permutation(group.index.to_numpy())
        for rank, donor in enumerate(members):
            positions[donor] = (rank + rng.uniform()) / len(members)
    return sorted(positions, key=positions.get)


def _round_robin(
    pools: dict[str, list[int]], order: list[str]
) -> Iterator[tuple[str, int]]:
    """Deal donors out one object at a time, cycling until every pool is empty.

    One pass takes at most one object from each donor, so a donor with many
    objects cannot crowd the others out.
    """
    while any(pools.values()):
        for donor in order:
            if pools[donor]:
                yield donor, pools[donor].pop()


def draw_sample(
    objects: dict[str, pd.DataFrame], cohorts: pd.Series, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`QUOTAS` objects of each population, spread over donors and cohorts.

    Returns the sample and the rows `REPLACED` discarded.
    """
    def row_for(population: str) -> dict:
        donor, index = next(streams[population])
        return {
            "donor": donor,
            "cohort": cohorts[donor],
            **objects[donor].loc[index].to_dict(),
        }

    streams = {}
    picked = []
    for population, quota in QUOTAS.items():
        pools = {
            donor: list(rng.permutation(frame.index[frame["population"] == population]))
            for donor, frame in objects.items()
        }
        streams[population] = _round_robin(pools, cohort_interleaved(cohorts, rng))
        try:
            picked.extend(row_for(population) for _ in range(quota))
        except StopIteration as exhausted:
            raise ValueError(f"too few {population} objects to sample") from exhausted

    sample = pd.DataFrame(picked)
    # Blinding: presentation order must not track the model's answer.
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.insert(0, "number", np.arange(1, len(sample) + 1))

    # Replacements come after numbering, so the tiles that were kept do not move.
    # A number repeated in `REPLACED` redraws that slot again, so the discards
    # are collected in the order they were rejected rather than by tile.
    numbered = sample.set_index("number", drop=False)
    discards = []
    for number in REPLACED:
        discards.append(numbered.loc[number].to_dict())
        numbered.loc[number] = {
            "number": number,
            **row_for(str(numbered.at[number, "population"])),
        }
    sample = numbered.reset_index(drop=True)
    return sample, pd.DataFrame(discards)


# --- Cutting the objects out of the raw images -------------------------------


def extract_tile(
    row: pd.Series, events: h5py.Group, masks: h5py.Group
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """One `TILE` x `TILE` native-resolution crop centred on the object.

    Returns the 8-bit greyscale tile, a boolean outline of the object's mask
    union, whether the object was larger than the tile and so clipped, and the
    event's own background level.
    """
    key = str(int(row["event_id"]))
    frame = np.asarray(events[key][BRIGHTFIELD], dtype=np.float32)
    labelled = np.asarray(masks[key])
    member = np.isin(labelled, [int(value) for value in row["instances"].split()])

    # Window the raw counts, then centre the object's bounding box in the tile.
    # Cells occupy a small fraction of an event, so its median is its own
    # background; padding with that keeps a tile at one brightness rather than
    # framing it in white.
    background = float(np.median(frame))
    grey = np.clip((frame - DISPLAY_LOW) / (DISPLAY_HIGH - DISPLAY_LOW), 0.0, 1.0)
    canvas = np.full(
        (TILE, TILE),
        (background - DISPLAY_LOW) / (DISPLAY_HIGH - DISPLAY_LOW),
        dtype=np.float32,
    )
    canvas_mask = np.zeros((TILE, TILE), dtype=bool)

    top = int(row["bbox_y"] + row["bbox_h"] / 2 - TILE / 2)
    left = int(row["bbox_x"] + row["bbox_w"] / 2 - TILE / 2)
    src_y0, src_x0 = max(top, 0), max(left, 0)
    src_y1 = min(top + TILE, frame.shape[0])
    src_x1 = min(left + TILE, frame.shape[1])
    dst_y0, dst_x0 = src_y0 - top, src_x0 - left
    height, width = src_y1 - src_y0, src_x1 - src_x0
    canvas[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = grey[
        src_y0:src_y1, src_x0:src_x1
    ]
    canvas_mask[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = member[
        src_y0:src_y1, src_x0:src_x1
    ]

    # A one-pixel boundary: inside the object, next to something outside it. It
    # shows the object's extent without revealing how many masks it is made of,
    # which is what separates an aggregate from a single cell.
    padded = np.pad(canvas_mask, 1)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    outline = canvas_mask & ~interior
    clipped = bool(row["bbox_w"] > TILE or row["bbox_h"] > TILE)
    return (canvas * 255).round().astype(np.uint8), outline, clipped, background


# --- Laying the tiles out ----------------------------------------------------


def label_font() -> ImageFont.FreeTypeFont:
    """A bitmap-crisp numeral font, from the copy matplotlib ships."""
    face = font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    return ImageFont.truetype(font_manager.findfont(face), 11)


def build_collage(
    tiles: list[np.ndarray], outlines: list[np.ndarray] | None
) -> Image.Image:
    """The numbered grid, at native resolution.

    The number sits in a strip above its tile rather than on top of it, so no
    cell pixel is ever hidden by the branding.
    """
    cell_height = LABEL_HEIGHT + TILE
    width = COLUMNS * (TILE + GUTTER) + GUTTER
    height = ROWS * (cell_height + GUTTER) + GUTTER
    collage = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(collage)
    font = label_font()

    for index, tile in enumerate(tiles):
        column, row = index % COLUMNS, index // COLUMNS
        x = GUTTER + column * (TILE + GUTTER)
        y = GUTTER + row * (cell_height + GUTTER)

        rgb = np.repeat(tile[:, :, None], 3, axis=2)
        if outlines is not None:
            rgb[outlines[index]] = OUTLINE
        collage.paste(Image.fromarray(rgb), (x, y + LABEL_HEIGHT))
        draw.text((x + 1, y + 1), str(index + 1), font=font, fill=INK)
        # A hairline box so the tile's edge is visible against white paper.
        draw.rectangle(
            [x - 1, y + LABEL_HEIGHT - 1, x + TILE, y + LABEL_HEIGHT + TILE],
            outline=(180, 180, 180),
        )
    return collage


# --- Running -----------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(SEED)

    donors = pd.read_csv(utils.DONORS_CSV)
    cohorts = donors.drop_duplicates("donor").set_index("donor")["cohort"]
    donors_per_method = (set(group["donor"]) for _, group in donors.groupby("method"))
    common = sorted(set.intersection(*donors_per_method))
    cohort_of = cohorts.reindex(common)

    gates = utils.marker_gates(METHOD, common)
    objects = {donor: donor_objects(donor, gates) for donor in common}
    sample, rejected = draw_sample(objects, cohort_of, rng)

    # One open file handle per donor, tiles cut in donor order to keep it that
    # way, then restored to the blinded presentation order.
    tiles: dict[int, np.ndarray] = {}
    outlines: dict[int, np.ndarray] = {}
    clipped: dict[int, bool] = {}
    backgrounds: dict[int, float] = {}
    for donor, group in sample.groupby("donor"):
        with h5py.File(find_h5(donor), "r") as handle:
            events = handle["events"]
            masks = handle[f"masks/{MASK_RUN}"]
            for _, row in group.iterrows():
                tile, outline, was_clipped, background = extract_tile(
                    row, events, masks
                )
                tiles[row["number"]] = tile
                outlines[row["number"]] = outline
                clipped[row["number"]] = was_clipped
                backgrounds[row["number"]] = background

    ordered = [tiles[number] for number in sample["number"]]
    ordered_outlines = [outlines[number] for number in sample["number"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_dir = OUTPUT_DIR / "tiles"
    tile_dir.mkdir(exist_ok=True)
    for number, tile in tiles.items():
        Image.fromarray(tile).save(tile_dir / f"{number:02d}.png")

    for name, marks in (("collage", None), ("collage_outlined", ordered_outlines)):
        collage = build_collage(ordered, marks)
        collage.save(OUTPUT_DIR / f"{name}.png")
        # Nearest neighbour: bigger on screen, not one pixel of new information.
        collage.resize(
            (collage.width * 6, collage.height * 6), Image.Resampling.NEAREST
        ).save(OUTPUT_DIR / f"{name}_x6.png")

    key = sample.assign(
        method=METHOD,
        model_class=sample["population"].astype(str),
        clipped=[clipped[number] for number in sample["number"]],
        background=[backgrounds[number] for number in sample["number"]],
    )[
        [
            "number",
            "model_class",
            "method",
            "donor",
            "cohort",
            "event_id",
            "instances",
            "n_masks",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "clipped",
            "background",
        ]
    ]
    key.to_csv(OUTPUT_DIR / "answer_key.csv", index=False)
    rejected.assign(model_class=rejected["population"].astype(str))[
        ["number", "model_class", "donor", "cohort", "event_id", "instances"]
    ].to_csv(OUTPUT_DIR / "replaced.csv", index=False)

    with (OUTPUT_DIR / "response_sheet.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rater_name", "number", "class", "confidence_1_to_5", "notes"])
        for number in sample["number"]:
            writer.writerow(["", number, "", "", ""])

    (OUTPUT_DIR / "README.txt").write_text(INSTRUCTIONS)

    counts = key.groupby(["model_class", "cohort"]).size().unstack(fill_value=0)
    print(f"{len(sample)} objects, {METHOD}, seed {SEED}")
    print(counts.to_string())
    print(f"\ndonors represented: {sample['donor'].nunique()} of {len(common)}")
    print(f"clipped by the {TILE}px tile: {sum(clipped.values())}")
    swapped = ", ".join(str(number) for number in REPLACED)
    print(f"replaced for image quality: {len(rejected)} ({swapped})")
    print(f"\nwritten to {OUTPUT_DIR}")


INSTRUCTIONS = """\
Platelet / RBC / aggregate -- human validation set
==================================================

You are looking at 50 numbered brightfield crops of objects segmented from
imaging flow cytometry of human blood. Each crop is shown at the microscope's
native resolution; collage_x6.png is the same picture enlarged six times with
no smoothing, which is easier to read on screen.

For each number, classify the object in the CENTRE of the tile as one of:

    platelet    a single small cell, roughly 1-4 um across
    rbc         a single red blood cell, roughly 6-8 um across
    aggregate   two or more cells stuck together as one clump

The pixels are the raw brightfield counts on a single fixed grey scale, so
tiles are not all equally bright -- each carries the background its own
acquisition had. Platelets are genuinely faint in brightfield; some are barely
darker than the background around them. That is the data, not a fault in the
picture.

Other cells may drift into the edges of a tile. Judge only the central object.
If you would like the object's outline marked for you, use
collage_outlined.png -- the red line is the boundary the segmentation drew. It
does not tell you how many cells are inside it.

Record your answers in response_sheet.csv: put your name in the rater_name
column, one of the three class names in the class column, and optionally how
sure you were from 1 (guessing) to 5 (certain). Please answer every number,
guessing if you must, and please do not confer with the other raters.
"""


if __name__ == "__main__":
    main()
