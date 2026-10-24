"""All file locations for ABOT. Import paths from here; never hard-code them.

Set the environment variable ABOT_DATA=fixtures to work with the small SYNTHETIC
dataset in tests/fixtures/ (for development only). In that mode results are written
to results/_synthetic/, which is git-ignored, so synthetic numbers never reach the deck.

Set ABOT_RAW_DIR to point at the shared-drive copy of the raw downloads if they are
not in data/raw/.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

USING_FIXTURES = os.environ.get("ABOT_DATA", "").strip().lower() == "fixtures"

CONFIG = ROOT / "config.yaml"
RAW = Path(os.environ.get("ABOT_RAW_DIR", str(ROOT / "data" / "raw")))
FIXTURES = ROOT / "tests" / "fixtures"
PROCESSED = FIXTURES if USING_FIXTURES else ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "_synthetic" if USING_FIXTURES else ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"


def folder(kind: str) -> Path:
    """Return the folder for a contract file kind: 'processed' or 'results'."""
    if kind == "processed":
        return PROCESSED
    if kind == "results":
        return RESULTS
    raise ValueError(f"Unknown folder kind: {kind!r}")
