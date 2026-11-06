#!/usr/bin/env bash
# Download the two open-access PhysioNet datasets into data/raw/ (about 1 GB unpacked).
# Both are open access: no PhysioNet credentialing needed (eICU, our fallback, does need it).
#
#   bash scripts/download_data.sh            # both datasets
#   bash scripts/download_data.sh 2019       # only sepsis
#   bash scripts/download_data.sh 2012       # only mortality
#
# Uses the wget command recommended on the PhysioNet pages. If a URL changed, check
#   https://physionet.org/content/challenge-2019/1.0.0/
#   https://physionet.org/content/challenge-2012/1.0.0/
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${MISSINGNESS_DATA_DIR:-data}/raw"
mkdir -p "$RAW"
which="${1:-all}"

if [[ "$which" == "all" || "$which" == "2019" ]]; then
  echo ">> PhysioNet 2019 (sepsis), ~40k small .psv files, takes a while"
  wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" -P "$RAW/physionet2019" \
    https://physionet.org/files/challenge-2019/1.0.0/training/
fi

if [[ "$which" == "all" || "$which" == "2012" ]]; then
  echo ">> PhysioNet 2012 (mortality)"
  mkdir -p "$RAW/physionet2012"
  for f in set-a.tar.gz set-b.tar.gz set-c.tar.gz Outcomes-a.txt Outcomes-b.txt Outcomes-c.txt; do
    wget -N -c -P "$RAW/physionet2012" "https://physionet.org/files/challenge-2012/1.0.0/$f"
  done
  for s in a b c; do tar -xzf "$RAW/physionet2012/set-$s.tar.gz" -C "$RAW/physionet2012"; done
fi

echo "Done. Next: python scripts/prepare_data.py physionet2019 physionet2012"
