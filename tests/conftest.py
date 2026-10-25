"""Test bootstrap: force mock mode and an isolated data dir BEFORE the app loads.

This is what lets `pytest` run with no API keys and no network
(Integration Standard point 1: one command runs all tests with no API keys).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="broll-tests-")
