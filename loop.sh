#!/bin/bash
# Ralph Loop - Playbook-compliant autonomous coding loop
#
# Usage: ./loop.sh <max_iterations> [mode]
# Examples:
#   ./loop.sh 20           # Build mode, max 20 iterations
#   ./loop.sh 5 plan       # Plan mode, max 5 iterations
#   ./loop.sh 10 research  # Research mode, max 10 iterations
#
# max_iterations is REQUIRED to prevent runaway loops

set -euo pipefail

# Exit cleanly on Ctrl+C
trap 'echo -e "\n\nInterrupted. Completed $ITERATION iterations."; exit 130' INT

# --- Argument validation ---
usage() {
    echo "Usage: $0 <max_iterations> [mode]" >&2
    echo "" >&2
    echo "Arguments:" >&2
    echo "  max_iterations  Required. Positive integer (safety limit)" >&2
    echo "  mode            Optional. One of: plan, research (default: build)" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 20           # Build mode, max 20 iterations" >&2
    echo "  $0 5 plan       # Plan mode, max 5 iterations" >&2
    echo "  $0 10 research  # Research mode, max 10 iterations" >&2
    exit 1
}

# Check we have at least 1 argument
if [ $# -lt 1 ]; then
    echo "Error: max_iterations is required" >&2
    usage
fi

MAX_ITERATIONS="$1"

# Validate max_iterations is a positive integer
if ! [[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: max_iterations must be a positive integer, got: '$MAX_ITERATIONS'" >&2
    usage
fi

# Parse optional mode argument
case "${2:-}" in
    plan)
        MODE="plan"
        PROMPT_FILE="PROMPT_plan.md"
        ;;
    research)
        MODE="research"
        PROMPT_FILE="PROMPT_research.md"
        ;;
    *)
        MODE="build"
        PROMPT_FILE="PROMPT_build.md"
        ;;
esac

ITERATION=0
CURRENT_BRANCH=$(git branch --show-current)

# --- Logging setup ---
LOG_DIR="logs/claude-sessions"
LOG_FILE="$LOG_DIR/${MODE}_$(date +%Y%m%d_%H%M%S).jsonl"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode:   $MODE"
echo "Prompt: $PROMPT_FILE"
echo "Branch: $CURRENT_BRANCH"
echo "Max:    $MAX_ITERATIONS iterations"
echo "Log:    $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify required files exist
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

if [ ! -f "AGENTS.md" ]; then
    echo "Error: AGENTS.md not found"
    exit 1
fi

# ======================================================= #
# RALPH LOOP
# ======================================================= #

# Create log file so wc -l works on first iteration
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

# Main loop
while [ "$ITERATION" -lt "$MAX_ITERATIONS" ]; do

    # Track log file position before this iteration
    LOG_LINES_BEFORE=$(wc -l < "$LOG_FILE")

    # THE CORE PATTERN: Concatenate AGENTS.md + PROMPT.md and feed to Claude
    # Output: JSONL appended to file, human-readable text to stdout
    {
        cat AGENTS.md
        echo ""
        echo "---"
        echo ""
        cat "$PROMPT_FILE"
    } | claude -p \
        --dangerously-skip-permissions \
        --model opus \
        --verbose \
        --output-format stream-json \
        2>&1 | tee -a "$LOG_FILE" \
        | jq '{type, message: .message.content[]? | {type, text, content, name, input, command, description} | del(.[] | nulls)}' 2>/dev/null

    ITERATION=$((ITERATION + 1))
    echo -e "\n\n======================== LOOP $ITERATION/$MAX_ITERATIONS ========================\n"

    # Check for exit signals in THIS iteration's JSONL output only
    NEW_LINES=$(tail -n +"$((LOG_LINES_BEFORE + 1))" "$LOG_FILE")

    # Check for completion signal
    if echo "$NEW_LINES" | grep -q '<promise>COMPLETE</promise>'; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Completion signal received after $ITERATION iterations"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit 0
    fi

    # Check for blocked signal
    if echo "$NEW_LINES" | grep -q '<promise>BLOCKED'; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚫 Blocked signal received after $ITERATION iterations"
        echo "$NEW_LINES" | grep '<promise>BLOCKED' || true
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit 1
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  Max iterations ($MAX_ITERATIONS) reached without completion signal"

# Check for blockers file as fallback diagnostic
if [ -f "blockers.md" ]; then
    echo "📋 Blockers documented in blockers.md:"
    head -20 blockers.md
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
