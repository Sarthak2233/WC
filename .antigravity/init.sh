#!/bin/bash
# .antigravity/init.sh - Harness Bootstrap

echo "--- Initializing World Cup Oracle Harness ---"

# 1. Environment Check
echo "[1/3] Checking dependencies..."
for cmd in ruff mypy pytest python3; do
    if ! command -v $cmd &> /dev/null; then
        echo "WARNING: $cmd is not installed."
    else
        echo "Found $cmd: $($cmd --version | head -n 1)"
    fi
done

# 2. State & Integrity Check
echo "[2/3] Validating TASK_STATE.json..."
if [ -f ".antigravity/TASK_STATE.json" ]; then
    python3 -c "import json; json.load(open('.antigravity/TASK_STATE.json'))" && echo "TASK_STATE.json is valid."
else
    echo "ERROR: TASK_STATE.json missing."
fi

# 3. Situational Awareness
echo "[3/3] Grounding Mission..."
echo "Current Goal: $(python3 -c "import json; d=json.load(open('.antigravity/TASK_STATE.json')); print(next(m['title'] for m in d['milestones'] if m['status'] == 'NOT_STARTED'))")"

echo "--- Harness Ready. Read .antigravity/CORE_PROBLEM_TO_SOLVE.md to begin. ---"
