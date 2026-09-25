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


# Last-line statuses that do not finish a record: processing started (and
# may have died), or the triage pass asked for a deep pass that has not run.
UNFINISHED_STATUSES = ("started", "awaiting_deep")


def is_final(row: dict | None, retry_statuses=()) -> bool:
    """True when ``row`` (a record's last results line) finishes the record
    and its status is not one to retry."""
    if not row:
        return False
    st = row.get("status")
    return st not in UNFINISHED_STATUSES and st not in set(retry_statuses or ())


def load_statuses(path: Path) -> dict[str, dict]:
    """As load_results, but each record keeps only its status (the fetch
    role reads a results file of 30,000 large rows at start)."""
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
                continue
            if row.get("record_id"):
                out[str(row["record_id"])] = {"status": row.get("status")}
    return out
