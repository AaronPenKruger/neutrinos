"""Pytest configuration.

This repository uses the "src" layout. To make running tests convenient without
an editable install, we add the src/ directory to sys.path.

Users can still install the package normally (recommended for real use).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
