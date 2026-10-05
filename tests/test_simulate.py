import pandas as pd
import pytest

from trial_detective import FRAUD_TYPES, TrialConfig, simulate_trial


def test_shape_and_truth():
    df, truth = simulate_trial(TrialConfig(seed=1))
    assert df["site_id"].nunique() == 20
    assert (df.groupby("patient_id").size() == 4).all()
    assert truth["is_fraud"].sum() == len(FRAUD_TYPES)
    assert set(truth["fraud_type"]) - {""} == set(FRAUD_TYPES)


def test_same_seed_same_data():
    a, _ = simulate_trial(TrialConfig(seed=7))
    b, _ = simulate_trial(TrialConfig(seed=7))
    pd.testing.assert_frame_equal(a, b)


def test_drug_lowers_blood_pressure():
    df, _ = simulate_trial(TrialConfig(seed=3, fraud_types=()))
    last = df[df["visit_week"] == 12].groupby("arm")["sbp"].mean()
    assert last["placebo"] - last["active"] > 4


def test_unknown_fraud_type_rejected():
    with pytest.raises(ValueError):
        simulate_trial(TrialConfig(fraud_types=("magic",)))
