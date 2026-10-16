# -*- coding: utf-8 -*-
"""Pytest configuration for sc15 unit and regression tests."""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PIPELINE_DIR = ROOT_DIR / "data_pipeline"

for p in (ROOT_DIR, DATA_PIPELINE_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
