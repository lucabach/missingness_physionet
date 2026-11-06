import pytest

from missingness.data.synthetic import make_synthetic


@pytest.fixture
def records():
    return make_synthetic(n_patients=40, seed=0)


@pytest.fixture
def stay_records():
    return make_synthetic(n_patients=40, task="stay", seed=1)
