"""
CSV logging utilities for experiment tracking

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""
import csv
import os
import json
import time
from typing import Dict, Any


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    """
    Append a row to CSV file, creating it with headers if it doesn't exist.
    
    Args:
        csv_path: Path to CSV file
        row: Dictionary of values to write
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), **row)
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def flatten_cfg(**kwargs) -> Dict[str, Any]:
    """
    Flatten nested config objects into CSV-able scalars.
    
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

