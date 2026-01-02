#!/usr/bin/env bash
# Bootstrap script: create a venv with Python 3.11 (if available) and install dev dependencies
set -euo pipefail
PYTHON=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON=python3.11
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "No suitable Python interpreter found (need >=3.11). Install Python 3.11 and retry." >&2
  exit 1
fi
# Check version
VER=$($PYTHON -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
if [[ "$VER" < "3.11" ]]; then
  echo "Python $VER detected. Project requires Python >= 3.11. Use a Python 3.11 interpreter." >&2
  exit 1
fi
# Create venv
$PYTHON -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install editable package with dev extras
pip install -e '.[dev]'
# Install coverage too (if not included)
pip install coverage pytest-cov || true

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
