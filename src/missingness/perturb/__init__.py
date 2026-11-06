"""Change the measurement process of real records by *removing* observations only.

Every function here returns new PatientRecords with some values set to NaN. Nothing is ever
added or altered, patients and labels stay the same. ``check_only_removed`` asserts this and
should be called in every experiment script.
"""

from missingness.perturb.invariants import check_only_removed
from missingness.perturb.removal import paired_removal, random_removal, state_dependent_removal
from missingness.perturb.thinning import measurement_rates, thin_to_rates

__all__ = [
    "check_only_removed",
    "measurement_rates",
    "paired_removal",
    "random_removal",
    "state_dependent_removal",
    "thin_to_rates",
]
