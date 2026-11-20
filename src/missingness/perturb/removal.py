"""Test-time removal of existing observations (sensitivity to measurement changes).

Two ways that remove the *same number* of values per patient (see the proposal):
* state-dependent:  the extra measurements in hours where the patient's measurement rate
                    rises above their own baseline (D5);
* random:           the same number of values *of each variable* as state-dependent removal
                    took, from random hours (D16). Without this matching, state-dependent
                    removal hits mostly labs and random removal mostly vital signs, so the two
                    would differ in which variables they remove, not only in which hours.

Removing values always removes physiological information as well as changing the pattern.
That is why sensitivity is measured as the *extra* loss of state-dependent over random
removal, compared with the values-only model (see CLAUDE.md, "Claims and wording").
"""

from __future__ import annotations

import numpy as np

from missingness.data.records import PatientRecord


def _remove(values: np.ndarray, flat_idx: np.ndarray) -> np.ndarray:
    out = values.copy()
    out.reshape(-1)[flat_idx] = np.nan
    return out


def random_removal(record: PatientRecord, n_remove: int, rng: np.random.Generator) -> PatientRecord:
    """Remove ``n_remove`` observed values chosen uniformly at random."""
    observed = np.flatnonzero(~np.isnan(record.values))
    n_remove = min(n_remove, len(observed))
    idx = rng.choice(observed, size=n_remove, replace=False)
    return record.with_values(_remove(record.values, idx))


def matched_random_removal(
    record: PatientRecord, per_variable: np.ndarray, rng: np.random.Generator
) -> PatientRecord:
    """Remove ``per_variable[d]`` observed values of each variable d, at random hours (D16)."""
    obs = ~np.isnan(record.values)
    D = record.values.shape[1]
    idx = []
    for d in np.flatnonzero(per_variable):
        hours = np.flatnonzero(obs[:, d])
        pick = rng.choice(hours, size=min(int(per_variable[d]), len(hours)), replace=False)
        idx.extend(pick * D + d)
    return record.with_values(_remove(record.values, np.asarray(idx, dtype=int)))


def state_dependent_candidates(record: PatientRecord, rng: np.random.Generator) -> np.ndarray:
    """Flat indices of the 'extra' measurements, in random order within each hour.

    First version (to be reviewed by the week-2 owner, see docs/decisions.md D5):
    baseline = the patient's median number of measurements per hour; in each hour with more
    than that, the observations above the baseline are candidates.
    """
    obs = ~np.isnan(record.values)
    counts = obs.sum(axis=1)
    baseline = int(np.floor(np.median(counts)))
    D = record.values.shape[1]
    cands = []
    for t in np.flatnonzero(counts > baseline):
        cols = np.flatnonzero(obs[t])
        extra = rng.choice(cols, size=counts[t] - baseline, replace=False)
        cands.extend(t * D + extra)
    return np.asarray(cands, dtype=int)


def state_dependent_removal(
    record: PatientRecord, n_remove: int, rng: np.random.Generator
) -> PatientRecord:
    cands = state_dependent_candidates(record, rng)
    idx = rng.choice(cands, size=min(n_remove, len(cands)), replace=False)
    return record.with_values(_remove(record.values, idx))


def paired_removal(
    records: list[PatientRecord], p: float, seed: int = 0, match_variables: bool = True
) -> tuple[list[PatientRecord], list[PatientRecord], float]:
    """Remove a fraction ``p`` of observations both ways, same count per patient.

    If a patient has fewer state-dependent candidates than ``p`` asks for, both methods remove
    only that many. With ``match_variables`` (default, D16) random removal also takes the same
    count per variable; ``False`` gives the first version (uniform over all values).
    Returns (random_records, state_records, achieved_fraction).
    """
    rng = np.random.default_rng(seed)
    rand_out, state_out, removed, total = [], [], 0, 0
    for r in records:
        cands = state_dependent_candidates(r, rng)
        n = min(int(round(p * r.n_observed)), len(cands))
        idx = rng.choice(cands, size=n, replace=False).astype(int)
        state_out.append(r.with_values(_remove(r.values, idx)))
        if match_variables:
            per_var = np.bincount(idx % r.values.shape[1], minlength=r.values.shape[1])
            rand_out.append(matched_random_removal(r, per_var, rng))
        else:
            rand_out.append(random_removal(r, n, rng))
        removed += n
        total += r.n_observed
    return rand_out, state_out, removed / max(total, 1)
