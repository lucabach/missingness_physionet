import numpy as np

from missingness.baselines.features import build_features, feature_names
from missingness.data.records import PatientRecord

NAN = np.nan


def _record():
    return PatientRecord(
        patient_id="p1",
        source="hospital1",
        hours=np.array([1.0, 2.0, 3.0, 4.0]),
        values=np.array([[NAN, 5.0], [7.0, NAN], [NAN, NAN], [9.0, 6.0]]),
        static=np.array([60.0]),
        labels=np.array([0, 0, 1, 1]),
    )


def test_pattern_features_are_causal_and_correct():
    out = build_features([_record()], "pattern", "hourly", np.zeros(2), np.zeros(1))
    X = out["X"]
    assert X.shape == (4, 6)
    assert X[:, 0].tolist() == [0, 1, 1, 2]  # count of var 0 so far
    assert X[:, 2].tolist() == [1, 0, 1, 0]  # hours since var 0: never yet -> elapsed + 1
    assert X[:, 3].tolist() == [0, 1, 2, 0]  # hours since var 1
    assert X[:, 4].tolist() == [0, 1, 0, 1]  # measured now


def test_values_use_training_mean_before_first_measurement():
    out = build_features([_record()], "values", "hourly", np.array([3.0, 4.0]), np.zeros(1))
    assert out["X"][:, 0].tolist() == [3, 7, 7, 9]
    assert out["X"][:, 2].tolist() == [60] * 4


def test_stay_task_keeps_last_step_only():
    out = build_features([_record()], "all", "stay", np.zeros(2), np.zeros(1))
    assert out["X"].shape == (1, len(feature_names(["a", "b"], ["s"], "all")))
    assert out["y"].tolist() == [1]
