"""Spool directory shared by the fetch (producer) and process (consumer) roles.

Layout under the spool root (claimed with the marker file
``.envision_survey_spool``; the survey refuses a non-empty directory
without it, and deletes only under ``records/``)::

    records/<id>/                 the producer writes the record's files here
    records/<id>/manifest.json    written atomically, then
    records/<id>/READY            the triage fetch is complete
    records/<id>/AWAITING_DEEP    the consumer asked for a deep pass
    records/<id>/deep/            the deep pass files, then
    records/<id>/manifest_deep.json and DEEP_READY
    records/<id>/.attempts        consumer attempts (crash loop guard)
    records/<id>/RETURNED         the consumer sent the record back (refetch)
    requests/<id>.json            {"kind": "deep" | "refetch"} for the producer

States (``Spool.state``): absent, fetching (dir without READY: the producer
is writing it, or died writing it), ready, awaiting_deep, deep_ready.

Ownership: the producer creates and fills a record dir and writes the
markers; the consumer reads ready dirs and removes a record dir only after
the record's result row is durably written. Local originals (the discovery
downloads dir) are never copied into the spool: manifests reference them
by absolute path, and the spool refuses to sit inside, contain or equal the
downloads dir.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

SPOOL_MARKER = ".envision_survey_spool"
RECORDS = "records"
REQUESTS = "requests"
READY = "READY"
AWAITING_DEEP = "AWAITING_DEEP"
DEEP_READY = "DEEP_READY"
RETURNED = "RETURNED"
MANIFEST = "manifest.json"
MANIFEST_DEEP = "manifest_deep.json"
LISTINGS = "listings.json"
ATTEMPTS = ".attempts"
DEEP_DIR = "deep"
STATES = ("absent", "fetching", "ready", "awaiting_deep", "deep_ready")


def _fsync_write(path: Path, text: str):
    """Write ``text`` to ``path`` atomically and durably (tmp, fsync, replace)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        fp.write(text)
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(path)


def valid_record_id(rid: str) -> bool:
    return isinstance(rid, str) and rid.isdigit() and 0 < len(rid) <= 20


class Spool:
    def __init__(self, root: Path, downloads_dir: Path | None = None, out_dir: Path | None = None):
        self.root = Path(root).resolve()
        if downloads_dir is not None:
            dl = Path(downloads_dir).resolve()
            if self.root == dl or dl.is_relative_to(self.root) or self.root.is_relative_to(dl):
                raise ValueError(f"spool dir {self.root} overlaps downloads dir {dl}")
        if out_dir is not None:
            out = Path(out_dir).resolve()
            if out == self.root or out.is_relative_to(self.root) or self.root.is_relative_to(out):
                raise ValueError(f"spool dir {self.root} overlaps output dir {out}")
        self._claim()
        self.records = self.root / RECORDS
        self.requests_dir = self.root / REQUESTS
        self.records.mkdir(exist_ok=True)
        self.requests_dir.mkdir(exist_ok=True)

    def _claim(self):
        marker = self.root / SPOOL_MARKER
        if self.root.exists():
            if not self.root.is_dir():
                raise ValueError(f"spool dir {self.root} is not a directory")
            if not marker.exists() and any(self.root.iterdir()):
                raise ValueError(f"spool dir {self.root} exists, is not empty and was not created by the survey "
                                 f"(no {SPOOL_MARKER} file); pass a new or empty directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text("envision-survey spool; the survey deletes files under records/ only\n",
                              encoding="utf-8")

    # ------------------------------------------------------------- records
    def record_dir(self, rid: str) -> Path:
        if not valid_record_id(rid):
            raise ValueError(f"not a Zenodo record id: {rid!r}")
        return self.records / rid

    def state(self, rid: str) -> str:
        d = self.record_dir(rid)
        if not d.is_dir():
            return "absent"
        if (d / DEEP_READY).exists():
            return "deep_ready"
        if (d / AWAITING_DEEP).exists():
            return "awaiting_deep"
        if (d / READY).exists():
            return "ready"
        return "fetching"

    def record_ids(self) -> list[str]:
        try:
            return sorted(p.name for p in self.records.iterdir() if p.is_dir() and valid_record_id(p.name))
        except OSError:
            return []

    def states(self) -> dict[str, str]:
        return {rid: self.state(rid) for rid in self.record_ids()}

    def ready_records(self) -> list[str]:
        """Records the consumer can process (ready or deep_ready), oldest
        marker first."""
        out = []
        for rid in self.record_ids():
            st = self.state(rid)
            if st in ("ready", "deep_ready"):
                marker = self.record_dir(rid) / (DEEP_READY if st == "deep_ready" else READY)
                try:
                    out.append((marker.stat().st_mtime, rid))
                except OSError:
                    continue
        return [rid for _, rid in sorted(out)]

    def write_manifest(self, rid: str, manifest: dict, deep: bool = False):
        """Manifest first (durable), then the ready marker: a READY file
        always has a complete manifest next to it."""
        d = self.record_dir(rid)
        target = d / DEEP_DIR if deep else d
        target.mkdir(parents=True, exist_ok=True)
        _fsync_write(d / (MANIFEST_DEEP if deep else MANIFEST), json.dumps(manifest, default=str))
        _fsync_write(d / (DEEP_READY if deep else READY), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")

    def read_manifest(self, rid: str, deep: bool = False) -> dict:
        p = self.record_dir(rid) / (MANIFEST_DEEP if deep else MANIFEST)
        return json.loads(p.read_text(encoding="utf-8"))

    def write_listings(self, rid: str, listings: list[dict]):
        _fsync_write(self.record_dir(rid) / LISTINGS, json.dumps(listings))

    def read_listings(self, rid: str) -> list[dict] | None:
        p = self.record_dir(rid) / LISTINGS
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def unmark(self, rid: str):
        """Send a published record back to the producer: a RETURNED marker
        first (a dir the consumer returned, not one a producer left half
        written), then the ready markers, READY first and DEEP_READY last, so
        the record is never seen as awaiting_deep on the way. The state is
        then 'fetching'; the producer removes and refetches the dir."""
        d = self.record_dir(rid)
        _fsync_write(d / RETURNED, "sent back to the producer\n")
        for name in (READY, AWAITING_DEEP, DEEP_READY):
            (d / name).unlink(missing_ok=True)

    def returned(self, rid: str) -> bool:
        return (self.record_dir(rid) / RETURNED).exists()

    def mark_awaiting_deep(self, rid: str):
        _fsync_write(self.record_dir(rid) / AWAITING_DEEP, "deep pass requested\n")

    def bump_attempts(self, rid: str) -> int:
        p = self.record_dir(rid) / ATTEMPTS
        try:
            n = int(p.read_text(encoding="ascii").strip() or 0)
        except (OSError, ValueError):
            n = 0
        _fsync_write(p, f"{n + 1}\n")
        return n + 1

    def attempts(self, rid: str) -> int:
        try:
            return int((self.record_dir(rid) / ATTEMPTS).read_text(encoding="ascii").strip() or 0)
        except (OSError, ValueError):
            return 0

    def _safe_target(self, path: Path) -> Path:
        """``path`` itself (not resolved through a symlink), after checking it
        is a real directory directly under records/ (or its deep/ subdir)."""
        path = Path(path)
        if path.is_symlink():
            raise ValueError(f"refusing to delete a symlink in the spool: {path}")
        parent = path.parent.resolve()
        records = self.records.resolve()
        ok = (parent == records and valid_record_id(path.name)) or (
            path.name == DEEP_DIR and parent.parent == records and valid_record_id(parent.name))
        if not ok:
            raise ValueError(f"refusing to delete {path}: not a spool record dir")
        return path

    def remove(self, rid: str):
        """Delete a record's spool dir (and nothing else)."""
        d = self._safe_target(self.record_dir(rid))
        if d.exists():
            shutil.rmtree(d)

    def remove_deep(self, rid: str):
        """Delete a partly written deep pass (dir, manifest, marker)."""
        d = self.record_dir(rid)
        deep = d / DEEP_DIR
        if deep.exists():
            shutil.rmtree(self._safe_target(deep))
        for name in (MANIFEST_DEEP, DEEP_READY):
            (d / name).unlink(missing_ok=True)

    # ------------------------------------------------------------ requests
    def request(self, rid: str, kind: str, detail: dict | None = None):
        if kind not in ("deep", "refetch"):
            raise ValueError(kind)
        self.record_dir(rid)          # validates the id
        _fsync_write(self.requests_dir / f"{rid}.json", json.dumps({"record_id": rid, "kind": kind,
                                                                    **(detail or {})}))

    def pending_requests(self) -> list[dict]:
        out = []
        try:
            paths = sorted(self.requests_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return out
        for p in paths:
            if not valid_record_id(p.stem):
                continue
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(doc)
        return out

    def clear_request(self, rid: str):
        (self.requests_dir / f"{rid}.json").unlink(missing_ok=True)

    # --------------------------------------------------------------- sizes
    def bytes_used(self) -> int:
        """Bytes under records/ (symlinks count as links, not targets)."""
        total = 0
        for dirpath, _dirs, files in os.walk(self.records):
            for fn in files:
                try:
                    total += os.lstat(os.path.join(dirpath, fn)).st_size
                except OSError:
                    pass
        return total

    def validate(self, rid: str, manifest: dict) -> list[str]:
        """Problems with a ready record's files: every spool item must exist
        inside the record dir with its recorded size, every local original
        must exist. Empty when the record can be processed."""
        d = self.record_dir(rid)
        problems = []
        for item in manifest.get("items") or []:
            rel, size = item.get("path"), item.get("size")
            if item.get("local"):
                p = Path(rel)
                if not p.is_file():
                    problems.append(f"local original missing: {rel}")
                continue
            p = (d / rel)
            if ".." in Path(rel).parts or Path(rel).is_absolute():
                problems.append(f"item path outside the record dir: {rel}")
                continue
            try:
                st = p.stat()
            except OSError:
                problems.append(f"missing: {rel}")
                continue
            if size is not None and st.st_size != size:
                problems.append(f"size changed: {rel} ({st.st_size} != {size})")
        return problems
