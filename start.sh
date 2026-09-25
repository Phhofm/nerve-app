#!/usr/bin/env bash
# Quick run for users who already have Python. Creates a venv, installs deps, launches the app.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
pip install -q onnxruntime-gpu 2>/dev/null || true   # NVIDIA if available
python app.py
