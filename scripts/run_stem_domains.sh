#!/bin/bash
# =============================================================================
# Domain Stemmer launcher
#
# Usage:
#   ./scripts/run_stem_domains.sh                       # read output/domains/domains_list.txt
#   ./scripts/run_stem_domains.sh --test                # first 50 domains
#   ./scripts/run_stem_domains.sh --limit 200           # first 200 domains
#   ./scripts/run_stem_domains.sh --file output/full_pipeline/
#   ./scripts/run_stem_domains.sh --file output/wikidata/grounded_domains.json
# =============================================================================

set -eo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_FILE="${OUTPUT_FILE:-output/domains/stem_results.json}"

# ── Argument parsing ────────────────────────────────────────────────────────

EXTRA_ARGS=()
INPUT_FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            EXTRA_ARGS+=("--limit" "50" "--verbose")
            shift
            ;;
        --file|--input-file)
            INPUT_FILES+=("$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Output directory ──────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_DIR}/$(dirname "$OUTPUT_FILE")"
mkdir -p "${REPO_DIR}/logs"

# ── Summary ─────────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════"
echo "  Domain Stemmer"
echo "══════════════════════════════════════════════════"
if [[ ${#INPUT_FILES[@]} -gt 0 ]]; then
    echo "  Source       : file (${INPUT_FILES[*]})"
else
    INPUT_FILES=("output/domains/domains_list.txt")
    echo "  Source       : file (${INPUT_FILES[*]})"
fi
echo "  Output       : $OUTPUT_FILE"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  Extra args   : ${EXTRA_ARGS[*]}"
fi
echo "══════════════════════════════════════════════════"
echo ""

# ── Launch ──────────────────────────────────────────────────────────────────

LOG_FILE="${REPO_DIR}/logs/stem_domains_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG_FILE"
echo ""

python3 "${REPO_DIR}/analyzer/stem_domains.py" \
    --input-file "${INPUT_FILES[@]}" \
    --output     "$OUTPUT_FILE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo "Log saved to:     $LOG_FILE"
