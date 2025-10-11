"""
CSV logging utilities for experiment tracking

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import csv
import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, Optional


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    """Append a row to CSV file, creating it with headers if it doesn't exist.

    Args:
        csv_path: Path to CSV file
        row: Dictionary of values to write
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def flatten_cfg(**kwargs) -> Dict[str, Any]:
    """Flatten nested config objects into CSV-able scalars.

    Converts dicts, lists, tuples to JSON strings for CSV compatibility.

    Returns:
        Flattened dictionary
    """
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (dict, list, tuple)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def get_env_info(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect environment info for reproducibility logs.

    Returns keys like torch_version, transformers_version, datasets_version,
    python_version, platform, git_commit.
    """
    info: Dict[str, Any] = {}
    # Library versions
    try:
        import torch
        info["torch_version"] = torch.__version__
    except Exception:
        info["torch_version"] = None
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = None
    try:
        import datasets
        info["datasets_version"] = datasets.__version__
    except Exception:
        info["datasets_version"] = None

    # Python and platform
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()

    # Git commit (best-effort)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = None
    info["git_commit"] = commit

    if extra:
        info.update(extra)
    return info
