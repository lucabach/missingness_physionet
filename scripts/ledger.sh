#!/usr/bin/env bash
# Show the work-ledger lines of every branch (local and origin), newest first, no duplicates.
#
#   bash scripts/ledger.sh            # all lines
#   bash scripts/ledger.sh thinning   # only lines that mention "thinning"
#
# Rules for the ledger: docs/ledger.md
set -euo pipefail

git fetch --all --quiet 2>/dev/null || echo "(could not fetch, showing what is already here)" >&2

git for-each-ref --format='%(refname)' refs/heads refs/remotes |
  while read -r ref; do
    git show "$ref:docs/ledger.md" 2>/dev/null | grep '^| 20' || true
  done |
  sort -u -r | { grep -i -- "${1:-}" || true; }
