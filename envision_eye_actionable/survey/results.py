"""Read the survey results JSONL (kept dependency-free for the Excel step)."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(path: Path) -> dict[str, dict]:
    """Last JSONL line per record id (later lines win, so reruns replace rows)."""
    out: dict[str, dict] = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a half-written line from a killed run
            if row.get("record_id"):
                out[str(row["record_id"])] = row
    return out
