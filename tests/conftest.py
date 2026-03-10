from __future__ import annotations

import gc
import shutil
import time
import uuid
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parents[1] / "test-tmp" / "cases"


@pytest.fixture
def tmp_path() -> Path:
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / str(uuid.uuid4())
    path.mkdir()
    try:
        yield path
    finally:
        gc.collect()
        time.sleep(0.05)
        shutil.rmtree(path, ignore_errors=True)
