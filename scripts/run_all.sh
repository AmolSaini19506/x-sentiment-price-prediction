#!/usr/bin/env bash
# Convenience wrapper: full pipeline + reports.
set -e
cd "$(dirname "$0")/.."
python -m src.run_pipeline
python -m src.make_report
