"""Simulate a multi-site blood-pressure trial in which some sites fabricate data.

The trial: patients with high blood pressure are randomised to a drug or a
placebo and measured at weeks 0, 4, 8 and 12. Honest sites produce realistic
data (correlated measurements, natural noise, a little rounding). Each
fraudulent site cheats in one specific way, described in ``FRAUD_TYPES``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VISIT_WEEKS = (0, 4, 8, 12)
MEASURES = ("sbp", "dbp", "hr")  # systolic BP, diastolic BP, heart rate

FRAUD_TYPES = {
    "digit_preference": "Numbers were invented by a person, who over-uses endings like 0 and 5.",
    "too_tidy": "Values are unrealistically close to the average, with too little noise.",
    "copy_paste": "Some patients are copies of other patients with small edits.",
    "no_correlation": "Each number was invented separately, so natural relationships are missing.",
}

# How often a person inventing numbers picks each last digit (0-9).
_HUMAN_DIGIT_WEIGHTS = np.array([0.22, 0.04, 0.12, 0.07, 0.08, 0.20, 0.07, 0.06, 0.10, 0.04])

# Average fall in blood pressure by week 12: (drug, placebo)
_SBP_DROP = {"active": 12.0, "placebo": 4.0}
_DBP_DROP = {"active": 7.0, "placebo": 2.0}


@dataclass(frozen=True)
class TrialConfig:
    n_sites: int = 20
    patients_per_site: tuple[int, int] = (40, 80)
    fraud_types: tuple[str, ...] = tuple(FRAUD_TYPES)
    seed: int = 42


def _site_measurements(rng, arm, site_shift, between_scale=1.0, noise_scale=1.0):
    """Realistic (n_patients x n_visits) arrays for one site."""
    n = len(arm)
    progress = np.arange(len(VISIT_WEEKS)) / (len(VISIT_WEEKS) - 1)  # 0 .. 1
    response = rng.normal(1.0, 0.3, n)  # how strongly each patient responds
    sbp_drop = np.array([_SBP_DROP[a] for a in arm]) * response
    dbp_drop = np.array([_DBP_DROP[a] for a in arm]) * response

    sbp_dev = rng.normal(0, 12 * between_scale, n)
    sbp_base = 155 + site_shift + sbp_dev
    dbp_base = 95 + 0.45 * sbp_dev + rng.normal(0, 5 * between_scale, n)
    hr_base = rng.normal(72, 8 * between_scale, n)

    shape = (n, len(VISIT_WEEKS))
    sbp = sbp_base[:, None] - sbp_drop[:, None] * progress + rng.normal(0, 6 * noise_scale, shape)
    dbp = dbp_base[:, None] - dbp_drop[:, None] * progress + rng.normal(0, 4 * noise_scale, shape)
    hr = hr_base[:, None] + rng.normal(0, 4 * noise_scale, shape)
    return {"sbp": sbp, "dbp": dbp, "hr": hr}


def _independent_measurements(rng, arm):
    """Every number drawn on its own: averages look right, relationships do not."""
    n = len(arm)
    progress = np.arange(len(VISIT_WEEKS)) / (len(VISIT_WEEKS) - 1)
    sbp_drop = np.array([_SBP_DROP[a] for a in arm])
    dbp_drop = np.array([_DBP_DROP[a] for a in arm])
    shape = (n, len(VISIT_WEEKS))
    return {
        "sbp": 155 - sbp_drop[:, None] * progress + rng.normal(0, 13.5, shape),
        "dbp": 95 - dbp_drop[:, None] * progress + rng.normal(0, 8.5, shape),
        "hr": rng.normal(72, 9, shape),
    }


def _apply_copy_paste(rng, data, fraction=0.35):
    n = data["sbp"].shape[0]
    n_copies = max(2, round(fraction * n))
    targets = rng.choice(n, n_copies, replace=False)
    originals = np.setdiff1d(np.arange(n), targets)
    for t in targets:
        src = rng.choice(originals)
        for m in MEASURES:
            edit = rng.integers(-1, 2, data[m].shape[1]) * (rng.random(data[m].shape[1]) < 0.3)
            data[m][t] = data[m][src] + edit


def _apply_digit_preference(rng, data):
    for m in MEASURES:
        values = data[m]
        digits = rng.choice(10, size=values.shape, p=_HUMAN_DIGIT_WEIGHTS)
        tens = np.round((values - digits) / 10)  # nearest number ending in that digit
        data[m] = (tens * 10 + digits).astype(int)


def simulate_trial(config: TrialConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (data, truth).

    ``data`` has one row per patient visit and is all a detector may see.
    ``truth`` has one row per site and says which sites cheated and how.
    """
    config = config or TrialConfig()
    unknown = set(config.fraud_types) - set(FRAUD_TYPES)
    if unknown:
        raise ValueError(f"Unknown fraud types: {sorted(unknown)}")
    if len(config.fraud_types) > config.n_sites:
        raise ValueError("More fraud types than sites")

    rng = np.random.default_rng(config.seed)
    site_ids = [f"S{i + 1:02d}" for i in range(config.n_sites)]
    fraud_sites = rng.choice(config.n_sites, len(config.fraud_types), replace=False)
    fraud_by_site = {site_ids[s]: f for s, f in zip(fraud_sites, config.fraud_types)}

    frames = []
    lo, hi = config.patients_per_site
    for site in site_ids:
        fraud = fraud_by_site.get(site)
        n = int(rng.integers(lo, hi + 1))
        arm = rng.permutation(np.where(np.arange(n) % 2 == 0, "active", "placebo"))
        site_shift = rng.normal(0, 2)

        if fraud == "no_correlation":
            data = _independent_measurements(rng, arm)
        elif fraud == "too_tidy":
            data = _site_measurements(rng, arm, site_shift, between_scale=0.5, noise_scale=0.35)
        else:
            data = _site_measurements(rng, arm, site_shift)

        # Staff record whole numbers and occasionally round to the nearest 10.
        rounding_habit = rng.uniform(0, 0.08)
        for m in MEASURES:
            whole = np.round(data[m])
            rounded = rng.random(whole.shape) < rounding_habit
            data[m] = np.where(rounded, np.round(whole / 10) * 10, whole).astype(int)

        if fraud == "copy_paste":
            _apply_copy_paste(rng, data)
        elif fraud == "digit_preference":
            _apply_digit_preference(rng, data)

        patient_ids = [f"{site}-{i + 1:03d}" for i in range(n)]
        age = np.clip(np.round(rng.normal(58, 10, n)), 30, 85).astype(int)
        sex = rng.choice(["F", "M"], n)
        for k, week in enumerate(VISIT_WEEKS):
            frames.append(
                pd.DataFrame(
                    {
                        "site_id": site,
                        "patient_id": patient_ids,
                        "age": age,
                        "sex": sex,
                        "arm": arm,
                        "visit_week": week,
                        "sbp": data["sbp"][:, k],
                        "dbp": data["dbp"][:, k],
                        "hr": data["hr"][:, k],
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["patient_id", "visit_week"], ignore_index=True)
    truth = pd.DataFrame(
        {
            "site_id": site_ids,
            "is_fraud": [s in fraud_by_site for s in site_ids],
            "fraud_type": [fraud_by_site.get(s, "") for s in site_ids],
        }
    )
    return df, truth
