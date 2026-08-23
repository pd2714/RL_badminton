from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_writable_matplotlib_config() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = Path(tempfile.gettempdir()) / "badminton-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
