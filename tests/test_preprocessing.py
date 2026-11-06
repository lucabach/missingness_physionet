import numpy as np

from missingness.data.preprocessing import Normalizer, compute_delta, compute_mask, forward_fill

NAN = np.nan


def test_mask_and_delta_follow_grud_definition():
    values = np.array([[1.0], [NAN], [NAN], [4.0], [5.0]])
    hours = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = compute_mask(values)
    assert m[:, 0].tolist() == [1, 0, 0, 1, 1]
    # delta: 0 at start, then grows while missing, resets after an observation
    assert compute_delta(m, hours)[:, 0].tolist() == [0, 1, 2, 3, 1]


def test_delta_uses_real_time_gaps():
    values = np.array([[1.0], [NAN], [2.0]])
    hours = np.array([0.0, 0.5, 2.0])
    assert compute_delta(compute_mask(values), hours)[:, 0].tolist() == [0, 0.5, 2.0]


def test_forward_fill_uses_fill_value_before_first_observation():
    values = np.array([[NAN, 1.0], [2.0, NAN], [NAN, NAN]])
    out = forward_fill(values, fill_value=np.array([9.0, 9.0]))
    assert out.tolist() == [[9, 1], [2, 1], [2, 1]]


def test_normalizer_ignores_missing(records):
    norm = Normalizer.fit(records)
    assert np.all(np.isfinite(norm.mean)) and np.all(norm.std > 0)
    again = Normalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(again.mean, norm.mean)
