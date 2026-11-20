"""Small helpers shared across the package."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str:
    """Current commit of the repo, with "-dirty" if files changed since; "unknown" outside git.

    Saved in every run's config.yaml (plan, rule 3) so a result can be traced to its code.
    """
    import subprocess

    from missingness.paths import REPO_ROOT

    try:
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        commit = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
