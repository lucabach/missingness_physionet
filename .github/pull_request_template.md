## What this PR does

<!-- One or two sentences. Link the issue or week deliverable it belongs to. -->

## How to run / check it

<!-- The exact command(s). Code without a run command does not count as done. -->

## Checklist

- [ ] `ruff check . && ruff format --check .` passes
- [ ] `pytest` passes (new logic has a test on synthetic data)
- [ ] No data, checkpoints or patient-level outputs committed
- [ ] Config changes are in `configs/`, not hard-coded
- [ ] Docs updated if behaviour changed (README, CLAUDE.md, `docs/`)
- [ ] Terminology follows CLAUDE.md ("sensitivity to measurement changes", "hospital 1/2")
