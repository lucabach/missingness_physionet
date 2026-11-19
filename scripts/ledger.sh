#!/usr/bin/env bash
# Show the work-ledger lines of every branch (local and origin), newest first, no duplicates.
#
#   bash scripts/ledger.sh            # all lines
#   bash scripts/ledger.sh thinning   # only lines that mention "thinning"
#
# Rules for the ledger: docs/log/work_ledger.md
set -euo pipefail

git fetch --all --quiet 2>/dev/null || echo "(could not fetch, showing what is already here)" >&2

git for-each-ref --format='%(refname)' refs/heads refs/remotes |
  while read -r ref; do
    # docs/log/work_ledger.md since 19 Nov; branches older than that still have docs/ledger.md
    { git show "$ref:docs/log/work_ledger.md" 2>/dev/null ||
      git show "$ref:docs/ledger.md" 2>/dev/null; } | grep '^| 20' || true
  done |
  sort -u -r | { grep -i -- "${1:-}" || true; }
