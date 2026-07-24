#!/bin/sh
# API entrypoint: produce missing artifacts on first start, then serve.
# The reviewer never runs training by hand (SPEC M7).
set -e

if [ ! -f artifacts/signal_report.json ]; then
    echo "First start: no signal report found - running verify-signal (~15-20 min)..."
    python -m rwsat.cli verify-signal
fi

if [ ! -f artifacts/model.joblib ]; then
    echo "First start: no model artefact found - training (~15-20 min)..."
    python -m rwsat.cli train
fi

exec uvicorn rwsat.api:app --host 0.0.0.0 --port 8000
