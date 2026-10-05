import numpy as np
import pytest

from trial_detective import TrialConfig, evaluate, run_all, simulate_trial
from trial_detective.scoring import benjamini_hochberg

EXPECTED_DETECTOR = {
    "digit_preference": "digit_preference",
    "too_tidy": "variability",
    "copy_paste": "duplicates",
    "no_correlation": "correlation",
}


@pytest.mark.parametrize("fraud,detector", EXPECTED_DETECTOR.items())
def test_each_fraud_is_caught_by_its_detector(fraud, detector):
    df, truth = simulate_trial(TrialConfig(seed=11, fraud_types=(fraud,)))
    result = run_all(df)
    site = truth.loc[truth["is_fraud"], "site_id"].iloc[0]
    assert result.loc[site, "flagged"]
    assert detector in result.loc[site, "reasons"]


def test_honest_trials_are_rarely_flagged():
    flagged = 0
    for seed in range(10):
        df, _ = simulate_trial(TrialConfig(seed=seed, fraud_types=()))
        flagged += run_all(df)["flagged"].sum()
    assert flagged <= 10  # under 5% of 200 honest sites


def test_full_trial_scores_well():
    df, truth = simulate_trial(TrialConfig(seed=42))
    score = evaluate(run_all(df), truth)
    assert score["recall"] == 1.0
    assert score["false_positives"] <= 1


def test_benjamini_hochberg_known_answer():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False]


def test_missing_columns_rejected():
    df, _ = simulate_trial(TrialConfig(seed=1))
    with pytest.raises(ValueError):
        run_all(df.drop(columns="hr"))
