from __future__ import annotations

import os
from datetime import datetime


def make_run_dir(results_dir: str, run_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(results_dir, f"{run_name}-{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

