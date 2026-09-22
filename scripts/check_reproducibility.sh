#!/usr/bin/env bash
# check_reproducibility.sh — Generate pop=200 twice, same seed, multithreaded,
# canonicalize, then compare every CSV and every note file byte-for-byte.
# Prints PASS or lists differing files with a few differing lines each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POP=200
SEED=20260916

DATA_A="$REPO_ROOT/data/repro-a"
DATA_B="$REPO_ROOT/data/repro-b"

echo "=== Reproducibility Check ==="
echo "Population: $POP   Seed: $SEED   Mode: multithreaded + canonicalize"
echo ""

echo "--- Run A ---"
SYNTHEA_RUN_NAME="repro-a" "$SCRIPT_DIR/generate_synthea.sh" "$POP" "$SEED"

echo ""
echo "--- Run B ---"
SYNTHEA_RUN_NAME="repro-b" "$SCRIPT_DIR/generate_synthea.sh" "$POP" "$SEED"

echo ""
echo "--- Comparing ---"

DIFFS=()
CONTENT_DIFFS=()   # files where values differ (not just order)

# ── CSVs ───────────────────────────────────────────────────────────────────────
for csv_a in "$DATA_A/csv/"*.csv "$DATA_A/symptoms/csv/"*.csv; do
    [ -f "$csv_a" ] || continue
    # Build relative path from run root
    rel="${csv_a#"$DATA_A/"}"
    csv_b="$DATA_B/$rel"
    if [ ! -f "$csv_b" ]; then
        DIFFS+=("$rel  (missing in run B)")
    elif ! diff -q "$csv_a" "$csv_b" &>/dev/null; then
        DIFFS+=("$rel")
        # Detect whether row counts match (order-only) or values truly differ
        if [ "$(wc -l < "$csv_a")" = "$(wc -l < "$csv_b")" ]; then
            CONTENT_DIFFS+=("$rel  (same row count — may be value difference)")
        else
            CONTENT_DIFFS+=("$rel  (different row counts)")
        fi
    fi
done

# ── Notes directory ────────────────────────────────────────────────────────────
notes_a="$DATA_A/notes"
notes_b="$DATA_B/notes"

if [ -d "$notes_a" ] && [ -d "$notes_b" ]; then
    # Check for files present in A but not B
    while IFS= read -r -d '' note_a; do
        rel_note="${note_a#"$notes_a/"}"
        note_b="$notes_b/$rel_note"
        if [ ! -f "$note_b" ]; then
            DIFFS+=("notes/$rel_note  (missing in run B)")
        elif ! diff -q "$note_a" "$note_b" &>/dev/null; then
            DIFFS+=("notes/$rel_note")
            CONTENT_DIFFS+=("notes/$rel_note")
        fi
    done < <(find "$notes_a" -type f -print0 | sort -z)

    # Check for files in B not in A
    while IFS= read -r -d '' note_b; do
        rel_note="${note_b#"$notes_b/"}"
        note_a="$notes_a/$rel_note"
        if [ ! -f "$note_a" ]; then
            DIFFS+=("notes/$rel_note  (missing in run A)")
        fi
    done < <(find "$notes_b" -type f -print0 | sort -z)

elif [ -d "$notes_a" ] && [ ! -d "$notes_b" ]; then
    DIFFS+=("notes/  (directory missing in run B)")
elif [ ! -d "$notes_a" ] && [ -d "$notes_b" ]; then
    DIFFS+=("notes/  (directory missing in run A)")
fi

# ── Report ─────────────────────────────────────────────────────────────────────
echo ""
if [ "${#DIFFS[@]}" -eq 0 ]; then
    echo "PASS: All CSVs and notes are byte-identical across both runs."
    exit 0
fi

echo "FAIL: ${#DIFFS[@]} file(s) differ:"
for f in "${DIFFS[@]}"; do
    echo "  $f"
done

if [ "${#CONTENT_DIFFS[@]}" -gt 0 ]; then
    echo ""
    echo "Files with value differences (showing up to 6 diff lines each):"
    for f in "${CONTENT_DIFFS[@]}"; do
        # Strip any annotation after the filename
        base="${f%% *}"
        file_a="$DATA_A/$base"
        file_b="$DATA_B/$base"
        if [ -f "$file_a" ] && [ -f "$file_b" ]; then
            echo ""
            echo "  --- $base ---"
            diff "$file_a" "$file_b" | grep '^[<>]' | head -6 | sed 's/^/    /'
        fi
    done
fi

exit 1
