#!/bin/bash
# .antigravity/verify.sh - Harness Feedback Sensor

echo "--- Starting Harness Verification Suite ---"

# 1. Syntax & Formatting (Ruff)
if command -v ruff &> /dev/null; then
    echo "[1/3] Running Ruff (Lint/Format)..."
    ruff check src/ tests/
else
    echo "[SKIP] Ruff not found."
fi

# 2. Type Checking (Mypy)
if command -v mypy &> /dev/null; then
    echo "[2/3] Running Mypy (Type Check)..."
    mypy src/
else
    echo "[SKIP] Mypy not found."
fi

# 3. Unit Tests (Pytest)
if command -v pytest &> /dev/null; then
    echo "[3/3] Running Pytest (Logic Check)..."
    pytest tests/
else
    echo "[SKIP] Pytest not found."
fi

echo "--- Verification Complete ---"
