#!/usr/bin/env bash
# Run the whole pipeline for one condition:
#   scrub.py -> extract_labels.py -> serialize.py -> check_text_leaks.py
#
# Usage:
#   serialization/run_condition.sh <run-dir> <codes-file> [serialize.py options]
#
# Example:
#   serialization/run_condition.sh data/pop10000-seed20260916 configs/heart_failure.txt
#
# If configs/drop/<codes-file-stem>.txt exists, it is passed to serialize.py as
# --drop. If configs/leak_terms/<codes-file-stem>.txt exists, it is passed to the
# leak check as --terms. Outputs go to data/serialized/<run>-<stem>.jsonl (+ .stats.json,
# .leaks.md). Existing scrub/label/training outputs are replaced.
set -euo pipefail

RUN_DIR=${1:?usage: run_condition.sh <run-dir> <codes-file> [serialize options]}
CODES=${2:?usage: run_condition.sh <run-dir> <codes-file> [serialize options]}
shift 2

RUN_DIR=${RUN_DIR%/}
RUN=$(basename "$RUN_DIR")
DATA=$(dirname "$RUN_DIR")
STEM=$(basename "${CODES%.*}")
# Same code-set hash as scrub.py: sha1 of the sorted unique codes, first 6 chars.
SHA6=$(python3 - "$CODES" <<'PY'
import hashlib, sys
codes = set()
for line in open(sys.argv[1]):
    code = line.split("#", 1)[0].strip()
    if code:
        codes.add(code)
print(hashlib.sha1("\n".join(sorted(codes)).encode()).hexdigest()[:6])
PY
)

SCRUB="$DATA/${RUN}__scrub-${STEM}-${SHA6}"
TRAIN="$DATA/${RUN}__train-${SHA6}"
LABELS="$DATA/${RUN}__labels-${SHA6}.json"
OUT="$DATA/serialized/${RUN}-${STEM}.jsonl"
TERMS="configs/leak_terms/${STEM}.txt"

echo "== 1/4 scrub ($STEM, $SHA6)"
python3 scripts/scrub.py "$RUN_DIR" --codes "$CODES" --out "$SCRUB" --force | grep -E "Patients|WARN" || true

echo "== 2/4 labels"
python3 scripts/extract_labels.py "$SCRUB" --labels "$LABELS" --train-dir "$TRAIN" --force | grep -E "hits|ERROR" || true

echo "== 3/4 serialize"
DROP="configs/drop/${STEM}.txt"
DROP_ARG=()
[[ -f "$DROP" ]] && DROP_ARG=(--drop "$DROP")
python3 serialization/serialize.py "$TRAIN" "$LABELS" --out "$OUT" "${DROP_ARG[@]+"${DROP_ARG[@]}"}" "$@"

echo "== 4/4 text leak check"
TERMS_ARG=()
[[ -f "$TERMS" ]] && TERMS_ARG=(--terms "$TERMS")
python3 serialization/check_text_leaks.py "$OUT" "$TRAIN" "${TERMS_ARG[@]+"${TERMS_ARG[@]}"}"
