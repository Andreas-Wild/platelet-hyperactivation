"""Label single cells by hand, one image at a time, to validate the classifier.

Draws `TOTAL` objects the classifier in `utils.py` has already called, cuts each
one out of the raw brightfield channel at native resolution, and then shows them
to you one at a time in a window: press 1, 2 or 3 for the class and Enter to
commit it and move on. Unlike `validation_sample.py`, nothing is laid out as a
collage -- each object is judged on its own, filling the screen.

Sampling is balanced over the six strata (three classes x two cohorts), so
aggregates and PASC donors are represented well enough to score even though
neither is common in a natural sample. Within a stratum donors are dealt out
round robin, at most one object per donor per pass, so no donor dominates.
Presentation order is shuffled, so the order of the images carries no hint of
the model's answer, and the model's answer is never shown on screen.

    uv run hand_label.py                # sample once (if needed), then label
    uv run hand_label.py --sample       # only cut the first batch of tiles
    uv run hand_label.py --extend 1500  # add a further batch, keeping the old one
    uv run hand_label.py --score        # your labels against the model's
    uv run hand_label.py --review       # walk from the top, answers loaded

Sampling happens in batches: `--extend` draws a further set, excluding every
object drawn before, numbering it after the last batch and leaving the tiles
already labelled untouched. Each batch is seeded from `SEED` and its own index,
so any batch can be redrawn exactly.
Labelling can be interrupted at any time: `labels.csv` is rewritten after every
answer and the next run resumes at the first image still unlabelled.

Keys, while labelling:

    1 / 2 / 3   platelet / rbc / aggregate
    enter       commit the pending answer and go to the next image
    backspace   go back one image
    o           toggle the segmentation outline
    ?           toggle the key reminder
    escape      save and quit

Writes to `output/hand_labels/`:

    tiles/NNNN.png       each object, native resolution
    tiles/NNNN_o.png     the same with its mask boundary in red
    answer_key.csv       what the model called each one -- do not read while labelling
    labels.csv           your answers, one row per image
    disagreements.csv    where you and the model differ, written by --score
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import h5py
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.registry import backend_registry
from PIL import Image

import utils
import validation_sample as vs

#: The method the paper reports on. The marker gates are a property of the
#: method, so a validation set mixing methods validates four classifiers at once.
METHOD = vs.METHOD

#: How many objects in total, split as evenly as possible over the six
#: population x cohort strata (500 = 84 + 83 + 83 + 83 + 84 + 83).
TOTAL = 500

COHORTS = ("control", "pasc")

SEED = 20260825

#: Keys 1..3, in the order they are pressed.
CLASSES = ("platelet", "rbc", "aggregate")

OUTPUT_DIR = utils.ROOT / "output" / "hand_labels"
TILE_DIR = OUTPUT_DIR / "tiles"
ANSWER_KEY = OUTPUT_DIR / "answer_key.csv"
LABELS = OUTPUT_DIR / "labels.csv"
DISAGREEMENTS = OUTPUT_DIR / "disagreements.csv"


# --- Drawing the sample ------------------------------------------------------


def stratum_quotas(total: int) -> dict[tuple[str, str], int]:
    """`total` split as evenly as possible over population x cohort."""
    strata = [(population, cohort) for population in CLASSES for cohort in COHORTS]
    base, extra = divmod(total, len(strata))
    return {stratum: base + (i < extra) for i, stratum in enumerate(strata)}


def draw_sample(
    objects: dict[str, pd.DataFrame],
    cohorts: pd.Series,
    rng: np.random.Generator,
    total: int = TOTAL,
    taken: set[tuple[str, int, str]] = frozenset(),
    first_number: int = 1,
) -> pd.DataFrame:
    """`total` objects, equally split over class x cohort, spread over donors.

    `taken` holds the (donor, event_id, instances) of objects an earlier batch
    already drew; they are left out of the pools so a later batch never shows
    the same object twice. Instance numbers run per event, so the event has to
    be part of the key -- on the instance alone it would exclude every object
    that happens to be the first mask of its own event.
    """
    # One pass per donor to drop what earlier batches took, before the strata
    # start carving the frames up by population.
    fresh = {}
    for donor, frame in objects.items():
        keep = [
            (donor, int(event), instances) not in taken
            for event, instances in zip(
                frame["event_id"], frame["instances"], strict=True
            )
        ]
        fresh[donor] = frame[np.asarray(keep)]

    picked = []
    for (population, cohort), quota in stratum_quotas(total).items():
        donors = [donor for donor in objects if cohorts[donor] == cohort]
        pools = {}
        for donor in donors:
            frame = fresh[donor]
            available = frame.index[frame["population"] == population]
            pools[donor] = list(rng.permutation(available))
        stream = vs._round_robin(pools, list(rng.permutation(donors)))
        try:
            for _ in range(quota):
                donor, index = next(stream)
                picked.append(
                    {
                        "donor": donor,
                        "cohort": cohort,
                        **objects[donor].loc[index].to_dict(),
                    }
                )
        except StopIteration as exhausted:
            raise ValueError(
                f"too few {population} objects among {cohort} donors to sample {quota}"
            ) from exhausted

    # Blinding: presentation order must not track the model's answer.
    sample = pd.DataFrame(picked)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.insert(0, "number", np.arange(first_number, first_number + len(sample)))
    return sample


def cut_tiles(sample: pd.DataFrame) -> pd.DataFrame:
    """Write one PNG per object, plain and outlined; return the answer key.

    One open file handle per donor: objects are cut in donor order and the
    numbering keeps them in the blinded presentation order.
    """
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    clipped: dict[int, bool] = {}
    for donor, group in sample.groupby("donor"):
        with h5py.File(vs.find_h5(donor, METHOD), "r") as handle:
            events = handle["events"]
            masks = handle[f"masks/{vs.MASK_RUN}"]
            for _, row in group.iterrows():
                tile, outline, was_clipped, _ = vs.extract_tile(row, events, masks)
                number = int(row["number"])
                clipped[number] = was_clipped
                Image.fromarray(tile).save(TILE_DIR / f"{number:04d}.png")
                marked = np.repeat(tile[:, :, None], 3, axis=2)
                marked[outline] = vs.OUTLINE
                Image.fromarray(marked).save(TILE_DIR / f"{number:04d}_o.png")
        print(f"  {donor}: {len(group)} tiles")

    return sample.assign(
        method=METHOD,
        model_class=sample["population"].astype(str),
        clipped=[clipped[number] for number in sample["number"]],
    )[
        [
            "number",
            "model_class",
            "method",
            "batch",
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
        ]
    ]


def read_key() -> pd.DataFrame:
    """The answer key, with a `batch` column even if it was written before them."""
    key = pd.read_csv(ANSWER_KEY)
    if "batch" not in key.columns:
        key.insert(3, "batch", 1)
    return key


def load_objects() -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Every classified object for every donor imaged under all four methods."""
    donors = pd.read_csv(utils.DONORS_CSV)
    cohorts = donors.drop_duplicates("donor").set_index("donor")["cohort"]
    donors_per_method = (set(group["donor"]) for _, group in donors.groupby("method"))
    common = sorted(set.intersection(*donors_per_method))

    gates = utils.marker_gates(METHOD, common)
    print(f"classifying {len(common)} donors under {METHOD} ...")
    objects = {donor: vs.donor_objects(donor, gates) for donor in common}
    return objects, cohorts.reindex(common)


def build_sample(total: int = TOTAL) -> pd.DataFrame:
    """Draw `total` objects and cut them out, adding to any batch already drawn.

    Batches are appended, never redrawn: the numbering continues from the last
    batch and each batch has its own seed, so tiles already labelled keep their
    number and their image. Objects drawn before are excluded, so the batches
    together are still one sample without replacement.
    """
    existing = read_key() if ANSWER_KEY.exists() else None
    batch = 1 if existing is None else int(existing["batch"].max()) + 1
    taken = (
        set()
        if existing is None
        else set(
            zip(
                existing["donor"],
                existing["event_id"],
                existing["instances"],
                strict=True,
            )
        )
    )
    first_number = 1 if existing is None else int(existing["number"].max()) + 1

    objects, cohorts = load_objects()
    rng = np.random.default_rng([SEED, batch])
    sample = draw_sample(objects, cohorts, rng, total, taken, first_number)
    sample = sample.assign(batch=batch)

    print(f"cutting {len(sample)} tiles (batch {batch}) ...")
    key = cut_tiles(sample)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        key = pd.concat([existing, key], ignore_index=True)
    key.to_csv(ANSWER_KEY, index=False)

    fresh = key[key["batch"] == batch]
    counts = fresh.groupby(["model_class", "cohort"]).size().unstack(fill_value=0)
    print(
        f"\nbatch {batch}: {len(fresh)} objects, numbers "
        f"{first_number}-{int(fresh['number'].max())}, seed [{SEED}, {batch}]"
    )
    print(counts.to_string())
    print(f"donors represented: {fresh['donor'].nunique()} of {len(objects)}")
    print(f"clipped by the {vs.TILE}px tile: {int(fresh['clipped'].sum())}")
    print(f"{len(key)} objects in total -> {OUTPUT_DIR}")
    return key


# --- Scoring ------------------------------------------------------------------


def score() -> None:
    """Compare the labels on file with the model's, and write the comparison.

    Your labels are treated as the reference, so recall is the share of a real
    class the model finds and precision the share of its calls that are right.
    """
    key = read_key()
    if not LABELS.exists():
        raise FileNotFoundError(f"no labels yet at {LABELS}")
    labels = pd.read_csv(LABELS)
    scored = key.merge(labels[["number", "label"]], on="number", how="inner")
    scored = scored[scored["label"].isin(CLASSES)]
    if scored.empty:
        raise ValueError("no labelled images to score")

    agree = scored["label"] == scored["model_class"]
    matrix = pd.crosstab(scored["label"], scored["model_class"], dropna=False).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )

    # Cohen's kappa: agreement above what the two label distributions would
    # give by chance alone.
    n = len(scored)
    observed = agree.mean()
    expected = sum((matrix.loc[c].sum() / n) * (matrix[c].sum() / n) for c in CLASSES)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")

    per_class = pd.DataFrame(
        {
            "n_yours": matrix.sum(axis=1),
            "n_model": matrix.sum(axis=0),
            "recall": [
                matrix.loc[c, c] / matrix.loc[c].sum()
                if matrix.loc[c].sum()
                else np.nan
                for c in CLASSES
            ],
            "precision": [
                matrix.loc[c, c] / matrix[c].sum() if matrix[c].sum() else np.nan
                for c in CLASSES
            ],
        }
    )
    per_class["f1"] = (
        2
        * per_class["precision"]
        * per_class["recall"]
        / (per_class["precision"] + per_class["recall"])
    )

    print(f"{n} images labelled of {len(key)} drawn ({key['batch'].nunique()} batches)")
    print(f"\nagreement {observed:.1%}   Cohen's kappa {kappa:.3f}")
    print("\nrows: your label, columns: the model's")
    print(matrix.to_string())
    print("\n" + per_class.round(3).to_string())

    by = scored.assign(agree=agree).groupby("cohort")["agree"].agg(["size", "mean"])
    print("\nby cohort")
    print(by.rename(columns={"size": "n", "mean": "agreement"}).round(3).to_string())
    by_batch = (
        scored.assign(agree=agree).groupby("batch")["agree"].agg(["size", "mean"])
    )
    print("\nby batch")
    print(
        by_batch.rename(columns={"size": "n", "mean": "agreement"}).round(3).to_string()
    )

    disagreements = scored[~agree][
        ["number", "label", "model_class", "donor", "cohort", "n_masks", "clipped"]
    ].rename(columns={"label": "your_class"})
    disagreements.to_csv(DISAGREEMENTS, index=False)
    print(f"\n{len(disagreements)} disagreements -> {DISAGREEMENTS}")


# --- Labelling ---------------------------------------------------------------

HELP = (
    "1 platelet   2 rbc   3 aggregate   |   "
    "enter: next   backspace: back   o: outline   esc: quit"
)


class Session:
    """One pass over the tiles: the window, the key handling, and the answers.

    The answer file is the state -- it is rewritten after every commit, so a
    crash or a closed window costs at most the image on screen.
    """

    def __init__(
        self,
        numbers: list[int],
        answers: dict[int, str],
        start: int,
        times: dict[int, str] | None = None,
    ) -> None:
        self.numbers = numbers
        self.answers = answers
        #: When each answer was committed. Answers carried over from an earlier
        #: run keep the time they were first given.
        self.times = dict(times or {})
        self.index = start
        self.pending: str | None = None
        self.outlined = False
        self.show_help = True

        self.figure, self.axes = plt.subplots(figsize=(7.5, 8))
        self.figure.canvas.manager.set_window_title("hand labelling")
        self.axes.set_axis_off()
        self.image = self.axes.imshow(
            self._tile(), cmap="gray", vmin=0, vmax=255, interpolation="nearest"
        )
        self.caption = self.figure.text(
            0.5, 0.03, "", ha="center", va="bottom", fontsize=9, color="0.35"
        )
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)

        # Matplotlib's own shortcuts sit on s, q, o, f, ... -- keys this window
        # needs for itself.
        for key in list(plt.rcParams):
            if key.startswith("keymap."):
                plt.rcParams[key] = []
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", lambda event: self.save())

    # -- state --

    @property
    def number(self) -> int:
        return self.numbers[self.index]

    def _tile(self) -> np.ndarray:
        suffix = "_o" if getattr(self, "outlined", False) else ""
        return np.asarray(Image.open(TILE_DIR / f"{self.number:04d}{suffix}.png"))

    def save(self) -> None:
        rows = [
            {
                "number": number,
                "label": self.answers[number],
                "labelled_at": self.times.get(number, ""),
            }
            for number in self.numbers
            if number in self.answers
        ]
        frame = pd.DataFrame(rows, columns=["number", "label", "labelled_at"])
        frame.insert(2, "method", METHOD)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(LABELS, index=False)

    # -- drawing --

    def refresh(self) -> None:
        self.image.set_data(self._tile())
        done = len(self.answers)
        committed = self.answers.get(self.number)
        choice = self.pending or committed
        state = choice if choice else "-"
        if self.pending and committed and self.pending != committed:
            state = f"{self.pending} (was {committed})"
        elif committed and not self.pending:
            state = f"{committed} (committed)"
        self.axes.set_title(
            f"image {self.index + 1} of {len(self.numbers)}"
            f"   ({done} labelled)\n{state}",
            fontsize=13,
        )
        self.caption.set_text(HELP if self.show_help else "")
        self.figure.canvas.draw_idle()

    # -- keys --

    def on_key(self, event) -> None:
        key = event.key
        if key in ("1", "2", "3"):
            self.pending = CLASSES[int(key) - 1]
        elif key == "enter":
            self.commit()
        elif key in ("backspace", "left"):
            self.step(-1)
        elif key == "right":
            self.step(1)
        elif key == "o":
            self.outlined = not self.outlined
        elif key == "?":
            self.show_help = not self.show_help
        elif key == "escape":
            self.save()
            plt.close(self.figure)
            return
        else:
            return
        self.refresh()

    def commit(self) -> None:
        """Record the pending answer and advance; Enter alone keeps the old one."""
        if self.pending is not None:
            self.answers[self.number] = self.pending
            self.times[self.number] = datetime.now(UTC).isoformat(timespec="seconds")
            self.save()
        elif self.number not in self.answers:
            # Nothing chosen and nothing on file: refuse to skip silently.
            return
        self.step(1)

    def step(self, delta: int) -> None:
        self.index = min(max(self.index + delta, 0), len(self.numbers) - 1)
        self.pending = None


def load_answers(numbers: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    """The answers on file, and when each was given."""
    if not LABELS.exists():
        return {}, {}
    frame = pd.read_csv(LABELS)
    known = set(numbers)
    rows = [
        row
        for row in frame.itertuples()
        if int(row.number) in known and isinstance(row.label, str)
    ]
    answers = {int(row.number): str(row.label) for row in rows}
    times = {
        int(row.number): str(row.labelled_at)
        for row in rows
        if isinstance(getattr(row, "labelled_at", None), str)
    }
    return answers, times


def label(numbers: list[int], review: bool = False) -> None:
    """Open the window at the first image still unlabelled, or at the top."""
    # Without a display matplotlib falls back to Agg, where plt.show() returns
    # at once: the run would look like it worked and label nothing. Over ssh
    # this means X11 forwarding (ssh -Y, XQuartz on a Mac) is not up.
    if backend_registry.resolve_backend(None)[1] is None:
        raise SystemExit(
            f"no interactive display: matplotlib fell back to "
            f"{matplotlib.get_backend()!r}.\n"
            "Over ssh, connect with -Y and an X server running on your machine, "
            "or copy output/hand_labels/ to a machine with a screen."
        )

    answers, times = load_answers(numbers)
    unlabelled = [i for i, number in enumerate(numbers) if number not in answers]
    start = 0 if review or not unlabelled else unlabelled[0]
    print(
        f"{len(answers)} of {len(numbers)} already labelled; "
        f"opening at image {start + 1}"
    )
    print(HELP)

    session = Session(numbers, answers, start, times)
    session.refresh()
    plt.show()
    session.save()
    print(f"\n{len(session.answers)} of {len(numbers)} labelled -> {LABELS}")


# --- Running -----------------------------------------------------------------


def main(argv: list[str]) -> None:
    """`--sample`/`--extend N` draw tiles, `--score` compares, otherwise label."""
    if "--score" in argv:
        score()
        return

    extend = "--extend" in argv
    only_sample = "--sample" in argv or extend
    total = TOTAL
    if extend:
        position = argv.index("--extend") + 1
        if position >= len(argv) or not argv[position].isdigit():
            raise SystemExit("--extend needs a number of images, e.g. --extend 1500")
        total = int(argv[position])

    if only_sample or not ANSWER_KEY.exists():
        build_sample(total)
        if only_sample:
            return
        print()

    numbers = [int(number) for number in read_key()["number"]]
    missing = [n for n in numbers if not (TILE_DIR / f"{n:04d}.png").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} tiles missing from {TILE_DIR}; re-run with --sample"
        )
    label(numbers, review="--review" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
