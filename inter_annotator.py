"""Score the classifier against the human annotators, as reported in the paper.

Two validation sets, both drawn from `new_method`:

  * the 3000-image set labelled by annotator 1 alone: agreement with its Wilson
    interval, Cohen's kappa and macro-averaged precision, recall and F1;
  * the 50-image panel labelled independently by five annotators: agreement and
    Cohen's kappa for every pair of raters, Fleiss' kappa with and without the
    classifier as a sixth rater, and the classifier against the human majority
    consensus.

    uv run inter_annotator.py

Reads the labels in `validation/`. Prints the scores and writes them to
`output/validation_scores.txt`.
"""

from __future__ import annotations

import io
import itertools
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy import stats

import utils

CLASSES = ("platelet", "rbc", "aggregate")
#: A tile a rater left blank is scored as an error: this label matches no rater.
BLANK = "unlabelled"
CATEGORIES = (*CLASSES, BLANK)

VALIDATION_DIR = utils.ROOT / "validation"
HAND_LABELS = VALIDATION_DIR / "labelled_3000"
PANEL_DIR = VALIDATION_DIR / "panel_50"
#: Annotator N's answers are in `answers/annotator_N.csv`. Annotator 1 also
#: labelled the 3000-image set.
ANNOTATORS = tuple(f"annotator_{i}" for i in range(1, 6))


# --- Statistics --------------------------------------------------------------


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion."""
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa of two raters over all tiles."""
    observed = (a == b).mean()
    expected = sum((a == c).mean() * (b == c).mean() for c in CATEGORIES)
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: pd.DataFrame) -> float:
    """Fleiss' kappa over all tiles (rows) and raters (columns)."""
    counts = np.stack([(ratings == c).sum(axis=1).to_numpy() for c in CATEGORIES], 1)
    n_raters = ratings.shape[1]
    per_tile = ((counts * (counts - 1)).sum(axis=1)) / (n_raters * (n_raters - 1))
    p_class = counts.sum(axis=0) / counts.sum()
    observed, expected = per_tile.mean(), (p_class**2).sum()
    return (observed - expected) / (1 - expected)


def per_class(truth: pd.Series, rater: pd.Series) -> pd.DataFrame:
    """Recall, precision and F1 of `rater` against `truth`, one row per class."""
    rows = {}
    for c in CLASSES:
        tp = ((rater == c) & (truth == c)).sum()
        recall = tp / (truth == c).sum()
        precision = tp / (rater == c).sum()
        rows[c] = {
            "n_truth": int((truth == c).sum()),
            "n_rater": int((rater == c).sum()),
            "recall": recall,
            "precision": precision,
            "f1": 2 * recall * precision / (recall + precision),
        }
    return pd.DataFrame(rows).T


# --- Loading -----------------------------------------------------------------


def panel() -> pd.DataFrame:
    """One row per tile: every annotator's class and the classifier's."""
    key = pd.read_csv(PANEL_DIR / "answer_key.csv").set_index("number")
    table = pd.DataFrame({"model": key["model_class"]})
    for i, name in enumerate(ANNOTATORS, start=1):
        answers = pd.read_csv(
            PANEL_DIR / "answers" / f"annotator_{i}.csv", skipinitialspace=True
        ).set_index("image_number")["class"]
        answers = answers.str.strip().str.lower().where(lambda s: s.isin(CLASSES))
        table[name] = answers.reindex(table.index).fillna(BLANK)
    return table[[*ANNOTATORS, "model"]]


def consensus(humans: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Majority class of the human annotators per tile, and its vote count."""
    counts = pd.DataFrame({c: (humans == c).sum(axis=1) for c in CLASSES})
    return counts.idxmax(axis=1), counts.max(axis=1)


# --- Report ------------------------------------------------------------------


def report() -> None:
    # The 3000-image set: agreement of annotator 1 with the classifier.
    key = pd.read_csv(HAND_LABELS / "answer_key.csv").set_index("number")
    labels = pd.read_csv(HAND_LABELS / "labels.csv").set_index("number")["label"]
    model = key["model_class"].reindex(labels.index)
    agree, n = int((labels == model).sum()), len(labels)
    low, high = wilson(agree, n)
    print("=== 3000-image set, annotator 1 vs classifier ===")
    print(f"labelled {n} of {len(key)} drawn; {len(key) - n} judged not a valid event")
    print(f"agreement {agree}/{n} = {agree / n:.1%}", end="  ")
    print(f"95% Wilson CI {low:.1%} to {high:.1%}")
    print(f"Cohen's kappa {cohen_kappa(labels, model):.3f}")
    macro = per_class(labels, model)[["precision", "recall", "f1"]].mean()
    print(
        f"macro-averaged precision {macro['precision']:.2f}, "
        f"recall {macro['recall']:.2f}, F1 {macro['f1']:.2f}"
    )
    print(
        f"agreement over all {len(key)}, unlabelled counted as errors: "
        f"{agree / len(key):.1%}\n"
    )

    table = panel()
    humans = table[list(ANNOTATORS)]
    majority, votes = consensus(humans)
    print("=== 50-image panel ===")
    unanimous = (votes == len(ANNOTATORS)).sum()
    print(f"all {len(ANNOTATORS)} annotators agree: {unanimous} of {len(table)} tiles")
    split = votes.value_counts().sort_index(ascending=False).to_dict()
    print(f"votes for the consensus: {split}\n")

    raters = [*ANNOTATORS, "model"]
    pairs = pd.DataFrame(index=raters, columns=raters, dtype=float)
    for a, b in itertools.combinations(raters, 2):
        pairs.loc[a, b] = (table[a] == table[b]).mean()
        pairs.loc[b, a] = cohen_kappa(table[a], table[b])
    print("pairwise agreement (upper) / Cohen's kappa (lower)")
    print(pairs.round(3).fillna(1.0).to_string(), "\n")

    human_pairs = list(itertools.combinations(ANNOTATORS, 2))
    hh_agree = np.mean([pairs.loc[a, b] for a, b in human_pairs])
    hh_kappa = np.mean([pairs.loc[b, a] for a, b in human_pairs])
    hm_agree = np.mean([pairs.loc[a, "model"] for a in ANNOTATORS])
    hm_kappa = np.mean([pairs.loc["model", a] for a in ANNOTATORS])
    hh = [pairs.loc[a, b] for a, b in human_pairs]
    hk = [pairs.loc[b, a] for a, b in human_pairs]
    print(f"mean pairwise human-human: agreement {hh_agree:.3f}, kappa {hh_kappa:.3f}")
    print(
        f"  range: agreement {min(hh):.3f} to {max(hh):.3f}, "
        f"kappa {min(hk):.3f} to {max(hk):.3f}"
    )
    print(f"mean pairwise human-model: agreement {hm_agree:.3f}, kappa {hm_kappa:.3f}")
    complete = len(humans)
    print(
        f"Fleiss' kappa, 5 humans:         {fleiss_kappa(humans):.3f}  (n={complete})"
    )
    print(
        f"Fleiss' kappa, 5 humans + model: {fleiss_kappa(table):.3f}  (n={complete})\n"
    )

    model_agree = int((table["model"] == majority).sum())
    low, high = wilson(model_agree, len(table))
    print("=== classifier vs the human consensus ===")
    print(
        f"agreement {model_agree}/{len(table)} = {model_agree / len(table):.1%}  "
        f"95% Wilson CI {low:.1%} to {high:.1%}"
    )
    print(f"Cohen's kappa {cohen_kappa(majority, table['model']):.3f}")
    print(per_class(majority, table["model"]).round(3).to_string())
    differ = table.index[table["model"] != majority].tolist()
    print(f"tiles where the classifier differs from the consensus: {differ}")


if __name__ == "__main__":
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report()
    text = buffer.getvalue()
    print(text, end="")
    (utils.ROOT / "output").mkdir(exist_ok=True)
    (utils.ROOT / "output" / "validation_scores.txt").write_text(text)
