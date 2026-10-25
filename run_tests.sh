#!/usr/bin/env bash
# One command runs all tests (unit, integration, end-to-end) with no API keys.
# Mock mode is forced by tests/conftest.py, so no keys are required.
set -euo pipefail
cd "$(dirname "$0")"
exec python -m pytest -q "$@"
