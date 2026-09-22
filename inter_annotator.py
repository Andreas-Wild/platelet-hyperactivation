"""Score the classifier against the human annotators, as reported in the paper.

Two validation sets, both drawn from `new_method`:

  * the 3000-image set labelled by annotator 1 alone (`hand_label.py`), scored
    here only for the Wilson interval of its agreement -- `hand_label.py --score`
    gives the rest;
  * the 50-image panel labelled independently by five annotators
    (`validation_sample.py`), scored in full: agreement and Cohen's kappa for
    every pair of raters, Fleiss' kappa with and without the classifier as a
    sixth rater, and the classifier against the human majority consensus.

A rater who left a tile blank is dropped from the pairs involving that tile, and
Fleiss' kappa uses only the tiles every rater answered.

    uv run inter_annotator.py

Prints the scores and writes them to `output/hand_labels/validation/scores.txt`.
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

HAND_LABELS = utils.ROOT / "output" / "hand_labels"
PANEL_DIR = HAND_LABELS / "validation"
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
    """Cohen's kappa of two raters over the tiles both answered."""
    both = a.notna() & b.notna()
    a, b = a[both], b[both]
    observed = (a == b).mean()
    expected = sum((a == c).mean() * (b == c).mean() for c in CLASSES)
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: pd.DataFrame) -> float:
    """Fleiss' kappa over the tiles (rows) every rater (column) answered."""
    ratings = ratings.dropna()
    counts = np.stack([(ratings == c).sum(axis=1).to_numpy() for c in CLASSES], 1)
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
        table[name] = answers.reindex(table.index)
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
        both = table[a].notna() & table[b].notna()
        pairs.loc[a, b] = (table.loc[both, a] == table.loc[both, b]).mean()
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
    complete = humans.dropna().shape[0]
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
    (PANEL_DIR / "scores.txt").write_text(text)
