"""Combine the detectors into one verdict per site, and score against the truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import DETECTORS, validate
from .simulate import TrialConfig, simulate_trial


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries, controlling the false discovery rate."""
    p = np.asarray(pvalues, dtype=float).ravel()
    order = np.argsort(p)
    thresholds = alpha * np.arange(1, len(p) + 1) / len(p)
    passed = p[order] <= thresholds
    mask = np.zeros(len(p), dtype=bool)
    if passed.any():
        mask[order[: np.max(np.nonzero(passed)[0]) + 1]] = True
    return mask.reshape(np.shape(pvalues))


def run_all(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run every detector. One row per site, most suspicious first.

    Columns: one p-value per detector, ``suspicion`` (the largest
    -log10 p-value), ``flagged`` and ``reasons`` (which detectors fired after
    correcting for the number of sites x detectors tested).
    """
    validate(df)
    pvals = pd.DataFrame({name: fn(df) for name, fn in DETECTORS.items()}).sort_index()
    pvals = pvals.clip(lower=1e-20)  # tail approximations are not trustworthy beyond this
    significant = benjamini_hochberg(pvals.to_numpy(), alpha)

    result = pvals.copy()
    result["suspicion"] = (-np.log10(pvals)).max(axis=1)
    result["flagged"] = significant.any(axis=1)
    result["reasons"] = [
        ", ".join(pvals.columns[row]) for row in significant
    ]
    result.index.name = "site_id"
    return result.sort_values("suspicion", ascending=False)


def evaluate(result: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Compare flags with the known answer."""
    truth = truth.set_index("site_id")
    flagged = result["flagged"].reindex(truth.index)
    fraud = truth["is_fraud"]
    tp = int((flagged & fraud).sum())
    fp = int((flagged & ~fraud).sum())
    fn = int((~flagged & fraud).sum())
    tn = int((~flagged & ~fraud).sum())
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
    }


def benchmark(n_trials: int = 100, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Simulate many trials and report how often each kind of site is flagged."""
    rows = []
    for i in range(n_trials):
        df, truth = simulate_trial(TrialConfig(seed=seed + i))
        result = run_all(df, alpha)
        merged = truth.set_index("site_id").join(result[["flagged"]])
        merged["fraud_type"] = merged["fraud_type"].replace("", "honest")
        rows.append(merged[["fraud_type", "flagged"]])
    everything = pd.concat(rows)
    summary = everything.groupby("fraud_type")["flagged"].agg(sites="count", flagged="sum")
    summary["flag_rate"] = summary["flagged"] / summary["sites"]
    return summary
