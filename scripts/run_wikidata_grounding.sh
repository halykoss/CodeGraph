#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONCEPT_TYPE="generic"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --concept-type)
            CONCEPT_TYPE="$2"
            shift 2
            ;;
        --input-file)
            INPUT_FILE_OVERRIDE="$2"
            shift 2
            ;;
        --test)
            EXTRA_ARGS+=("--limit" "20" "--verbose")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$CONCEPT_TYPE" in
    domain|domains)
        DEFAULT_OUTPUT="output/wikidata/grounded_domains.json"
        DEFAULT_CACHE="cache/wikidata_domains"
        DEFAULT_INPUT="output/domains/domains_list.txt"
        ;;
    algorithm|algorithms)
        DEFAULT_OUTPUT="output/wikidata/grounded_algorithms.json"
        DEFAULT_CACHE="cache/wikidata_algorithms"
        DEFAULT_INPUT="output/algorithms/algorithms_list.txt"
        ;;
    paradigm|paradigms)
        DEFAULT_OUTPUT="output/wikidata/grounded_paradigms.json"
        DEFAULT_CACHE="cache/wikidata_paradigms"
        DEFAULT_INPUT="output/paradigms/paradigms_list.txt"
        ;;
    design-pattern|design-patterns|pattern|patterns)
        DEFAULT_OUTPUT="output/wikidata/grounded_design_patterns.json"
        DEFAULT_CACHE="cache/wikidata_design_patterns"
        DEFAULT_INPUT="output/design_patterns/design_patterns_list.txt"
        ;;
    generic|concept|concepts)
        DEFAULT_OUTPUT="output/wikidata/grounded_concepts.json"
        DEFAULT_CACHE="cache/wikidata_concepts"
        DEFAULT_INPUT="output/concepts/concepts_list.txt"
        ;;
    *)
        echo "ERROR: unknown --concept-type '$CONCEPT_TYPE'" >&2
        exit 2
        ;;
esac

BACKEND="${BACKEND:-openrouter}"
MODEL="${MODEL:-google/gemini-3-flash-preview}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
WORKERS="${WORKERS:-4}"
WIKIDATA_DELAY="${WIKIDATA_DELAY:-2.0}"
OUTPUT_FILE="${OUTPUT_FILE:-$DEFAULT_OUTPUT}"
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
INPUT_FILE="${INPUT_FILE_OVERRIDE:-$DEFAULT_INPUT}"

if [[ "$BACKEND" == "openrouter" && -z "$OPENROUTER_API_KEY" ]]; then
    echo "ERROR: set OPENROUTER_API_KEY or use BACKEND=vllm." >&2
    exit 1
fi

mkdir -p "${REPO_DIR}/$(dirname "$OUTPUT_FILE")" "${REPO_DIR}/${CACHE_DIR}" "${REPO_DIR}/logs"

LOG_FILE="${REPO_DIR}/logs/wikidata_grounding_${CONCEPT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "Grounding Wikidata"
echo "  concept type : $CONCEPT_TYPE"
echo "  backend      : $BACKEND"
echo "  model        : $MODEL"
echo "  output       : $OUTPUT_FILE"
echo "  cache        : $CACHE_DIR"
echo "  input        : $INPUT_FILE"
echo "  log          : $LOG_FILE"

python3 "${REPO_DIR}/analyzer/wikidata_grounding.py" \
    --input-file "$INPUT_FILE" \
    --backend "$BACKEND" \
    --openrouter-key "$OPENROUTER_API_KEY" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --cache-dir "$CACHE_DIR" \
    --workers "$WORKERS" \
    --wikidata-delay "$WIKIDATA_DELAY" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
