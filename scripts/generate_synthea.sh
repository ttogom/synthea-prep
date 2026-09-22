#!/usr/bin/env bash
# generate_synthea.sh — Reproducible Synthea patient-data generator.
#
# Usage:
#   ./scripts/generate_synthea.sh [--single-thread] [population] [seed] [age_range]
#
#   --single-thread  force Synthea to use one thread (slower but skips the
#                    canonicalize step; useful for debugging)
#   population       number of patients to generate   (default: 10000)
#   seed             controls patient + clinician RNG  (default: 20260916)
#   age_range        optional min-max filter, e.g. 30-40 (default: no filter)
#
# Override the output directory name:
#   SYNTHEA_RUN_NAME=my-name ./scripts/generate_synthea.sh 50 42
#
# Property names verified against synthea.properties at the pinned commit.
# exporter.symptoms.mode=0 (default): exports symptoms within exporter.years_of_history window.
# Any non-zero value exports all symptoms across the full patient history.
set -euo pipefail

# ── Pinned Synthea commit (master, 2026-09-21) ─────────────────────────────────
SYNTHEA_COMMIT="d9d07a6eef91ee5144293b42ab64224d84d124f8"

# ── Pinned reference date (simulation "present") ───────────────────────────────
# Without -r, Synthea uses System.currentTimeMillis() which varies per run and
# shifts all encounter timestamps, breaking byte-identical reproducibility.
REFERENCE_DATE="20260921"

# ── Parse --single-thread flag ─────────────────────────────────────────────────
SINGLE_THREAD=0
if [ "${1:-}" = "--single-thread" ]; then
    SINGLE_THREAD=1
    shift
fi

# ── Arguments ──────────────────────────────────────────────────────────────────
POPULATION=${1:-10000}
SEED=${2:-20260916}
AGE_RANGE=${3:-""}

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYNTHEA_DIR="$REPO_ROOT/.synthea"

# ── Java 17+ check ─────────────────────────────────────────────────────────────
if ! command -v java &>/dev/null; then
    echo "ERROR: java not found. Install Java 17 or later." >&2
    exit 1
fi
_JAVA_LINE=$(java -version 2>&1 | head -1)
_JAVA_MAJOR=$(echo "$_JAVA_LINE" | grep -oE '"[0-9]+' | head -1 | tr -d '"')
# Handle legacy 1.x strings (Java 8: "1.8.0_xxx")
if [ "${_JAVA_MAJOR:-0}" = "1" ]; then
    _JAVA_MAJOR=$(echo "$_JAVA_LINE" | grep -oE '"1\.[0-9]+' | head -1 | cut -d. -f2)
fi
if [ -z "${_JAVA_MAJOR:-}" ] || [ "${_JAVA_MAJOR}" -lt 17 ] 2>/dev/null; then
    echo "ERROR: Java 17+ required. Found: $_JAVA_LINE" >&2
    exit 1
fi

# ── Clone / checkout Synthea ───────────────────────────────────────────────────
if [ ! -d "$SYNTHEA_DIR/.git" ]; then
    echo "Cloning Synthea into $SYNTHEA_DIR ..."
    git clone https://github.com/synthetichealth/synthea "$SYNTHEA_DIR"
fi
echo "Checking out pinned commit $SYNTHEA_COMMIT ..."
git -C "$SYNTHEA_DIR" fetch --quiet origin 2>/dev/null || true
git -C "$SYNTHEA_DIR" checkout --quiet "$SYNTHEA_COMMIT"

# ── Output directory ───────────────────────────────────────────────────────────
AGE_SUFFIX=""
[ -n "$AGE_RANGE" ] && AGE_SUFFIX="-age${AGE_RANGE}"
RUN_NAME="${SYNTHEA_RUN_NAME:-"pop${POPULATION}-seed${SEED}${AGE_SUFFIX}"}"
OUTPUT_DIR="$REPO_ROOT/data/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ── Build command ──────────────────────────────────────────────────────────────
CMD=(
    ./run_synthea
    -p "$POPULATION"
    -s "$SEED"
    -cs "$SEED"
    -r "$REFERENCE_DATE"
    "--exporter.baseDirectory=$OUTPUT_DIR/"
    "--exporter.csv.export=true"
    "--exporter.symptoms.csv.export=true"
    "--exporter.symptoms.text.export=true"
    "--exporter.clinical_note.export=true"
    "--exporter.fhir.export=false"
    # Extend the default exclusion list (patient_expenses.csv) with claims files.
    # Separator is comma; verified in CSVFileManager.java propStringToList().
    "--exporter.csv.excluded_files=patient_expenses.csv,claims.csv,claims_transactions.csv,payers.csv"
)
[ -n "$AGE_RANGE" ] && CMD+=(-a "$AGE_RANGE")
[ "$SINGLE_THREAD" = "1" ] && CMD+=("--generate.thread_pool_size=1")

echo "Run name : $RUN_NAME"
echo "Output   : $OUTPUT_DIR"
echo "Threads  : $([ "$SINGLE_THREAD" = "1" ] && echo "1 (--single-thread)" || echo "default (multithreaded)")"
echo "Command  : ${CMD[*]}"
echo ""

(cd "$SYNTHEA_DIR" && "${CMD[@]}")

# ── Verify symptoms output ─────────────────────────────────────────────────────
SYMPTOMS_CSV="$OUTPUT_DIR/symptoms/csv/symptoms.csv"
if [ ! -f "$SYMPTOMS_CSV" ]; then
    echo "" >&2
    echo "ERROR: symptoms CSV not produced at:" >&2
    echo "       $SYMPTOMS_CSV" >&2
    echo "       Verify exporter.symptoms.csv.export=true was accepted." >&2
    exit 1
fi

# ── Canonicalize CSVs ─────────────────────────────────────────────────────────
# Sort every CSV by all columns so row order is deterministic regardless of
# threading. This is a no-op in practice with single-threaded generation but
# is cheap and makes the guarantee explicit.
echo ""
echo "Canonicalizing CSVs ..."
python3 "$SCRIPT_DIR/canonicalize_csvs.py" "$OUTPUT_DIR" --reference-date "$REFERENCE_DATE"

# ── Write manifest.json ────────────────────────────────────────────────────────
FULL_CMD="${CMD[*]}"
JAVA_VERSION="$_JAVA_LINE"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

export OUTPUT_DIR SYNTHEA_COMMIT REFERENCE_DATE POPULATION SEED AGE_RANGE RUN_NAME FULL_CMD JAVA_VERSION TIMESTAMP SINGLE_THREAD

python3 << 'PYEOF'
import json, os
od = os.environ["OUTPUT_DIR"]
manifest = {
    "synthea_commit": os.environ["SYNTHEA_COMMIT"],
    "reference_date": os.environ["REFERENCE_DATE"],
    "population": int(os.environ["POPULATION"]),
    "seed": int(os.environ["SEED"]),
    "clinician_seed": int(os.environ["SEED"]),
    "age_range": os.environ["AGE_RANGE"] or None,
    "run_name": os.environ["RUN_NAME"],
    "output_dir": od,
    "command": os.environ["FULL_CMD"],
    "single_thread": os.environ["SINGLE_THREAD"] == "1",
    "java_version": os.environ["JAVA_VERSION"],
    "timestamp": os.environ["TIMESTAMP"],
}
with open(f"{od}/manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"Manifest : {od}/manifest.json")
PYEOF

# ── Summarize ─────────────────────────────────────────────────────────────────
echo ""
echo "Summarizing ..."
python3 "$SCRIPT_DIR/summarize_dataset.py" "$OUTPUT_DIR"

echo "Done."
