"""Retention of the fetched files of eye-positive records (the keep dir).

The process role deletes a record's spool dir once its result row is
written. With ``--keep-dir`` a record whose row holds at least one image
classified as an eye class (thresholded label, MASK excluded: the n_CFP,
n_IR, n_PSC, n_FAF, n_OCT and n_OCTA counts) has its fetched files MOVED
into ``<keep>/<record id>/`` instead, relative paths preserved:

* ``dl/...``: files downloaded whole (archives, top-level images);
* ``remote/...``: zip members and nested archives read remotely;
* ``deep/...``: the same for the deep pass;
* ``manifest.json``, ``manifest_deep.json``, ``listings.json``: which
  Zenodo file (key) each path is.

Left out (deleted with the spool dir as before): the spool markers (READY,
AWAITING_DEEP, DEEP_READY, RETURNED, .attempts), temporary files, and the
scratch dirs ``work/`` (archive extractions, joined split sets) and
``.fetchwalk/``. Extractions are copies of members of archives that are
kept themselves, so keeping the archive is enough (and the classifier
never needs more than the archive plus the member path in the predictions
file). Local originals (the discovery downloads) are never moved or
copied: their paths go into the KEPT.json marker.

Moves are renames on one filesystem (the keep dir must be on the spool's
filesystem; checked at start), so no byte is copied and a file is at any
time either in the spool or in the keep dir, never in both and never lost.
Order: the row (kept = true) is written first, then each file is renamed,
then ``<keep>/<id>/KEPT.json`` is written, then the spool dir is deleted.
A kill anywhere in between leaves the spool dir with its READY marker; the
consumer finds the record's row newer than the marker, sees kept = true,
moves what is left and writes the marker again. A file moved again onto an
existing path replaces it (same bytes, same name): nothing is duplicated.

Guards (a record that is not kept gets kept = false and kept_reason; its
files are deleted as before; kept records are never deleted by the
survey):

* ``--keep-max-gb``: the keep dir holds at most this much;
* the disk: kept files stop being freed, so a record is kept only while
  the free space, counted as if the spool were empty, stays at least
  ``--disk-floor-gb`` + ``--spool-max-gb``. The producer's disk floor
  (fetch.RoomGate) then always has a full spool budget above the floor:
  keeping never turns records into skipped_disk rows.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .constants import EYE_CLASSES
from .spool import (ATTEMPTS, AWAITING_DEEP, DEEP_READY, LISTINGS, MANIFEST, MANIFEST_DEEP, READY, RETURNED,
                    _fsync_write, valid_record_id)

KEEP_MARKER = ".envision_survey_keep"
KEPT = "KEPT.json"
STATUS_NAME = "keep_status.json"
MARKER_FILES = {READY, AWAITING_DEEP, DEEP_READY, RETURNED, ATTEMPTS, KEPT}
SCRATCH_DIRS = {"work", ".fetchwalk"}
META_FILES = {MANIFEST, MANIFEST_DEEP, LISTINGS}
MAX_LOCAL_IN_ROW = 20


def _rename(src: Path, dst: Path):
    """One file into the keep dir: a rename (atomic, never a copy)."""
    os.replace(src, dst)


def _st_dev(path: Path) -> int:
    return os.stat(path).st_dev


def n_eye_images(row: dict) -> int:
    """Images of the row classified as an eye class (thresholded, MASK
    excluded)."""
    total = 0
    for c in EYE_CLASSES:
        try:
            total += int(row.get(f"n_{c}") or 0)
        except (TypeError, ValueError):
            continue
    return total


def keep_plan(src: Path) -> list[str]:
    """Relative paths (posix) of the files of spool record dir ``src`` that
    go to the keep dir: every regular file except markers, temporary files
    and the scratch dirs. Symlinks are never followed nor moved."""
    out = []
    src = Path(src)
    if not src.is_dir():
        return out
    for dirpath, dirs, files in os.walk(src, followlinks=False):
        rel_dir = Path(dirpath).relative_to(src)
        if rel_dir == Path("."):
            dirs[:] = [d for d in dirs if d not in SCRATCH_DIRS]
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(dirpath, d))]
        for fn in files:
            p = Path(dirpath) / fn
            if p.is_symlink() or not p.is_file():
                continue
            if rel_dir == Path(".") and fn in MARKER_FILES:
                continue
            if fn.endswith(".tmp") or fn.endswith(".part"):
                continue
            out.append((rel_dir / fn).as_posix())
    return sorted(out)


def _is_meta(rel: str) -> bool:
    return "/" not in rel and rel in META_FILES


def local_originals(*manifests: dict | None) -> list[str]:
    out = []
    for m in manifests:
        for it in (m or {}).get("items") or []:
            if it.get("local") and it.get("path") and it["path"] not in out:
                out.append(str(it["path"]))
    return out


class Keeper:
    """The keep dir of one process role. ``spool`` is the pipeline's
    Spool; the keep dir must not overlap it, the downloads dir or the output
    dir, and must be on the spool's filesystem."""

    def __init__(self, keep_dir: Path, spool, downloads_dir: Path | None = None, out_dir: Path | None = None,
                 max_bytes: float | None = None, floor_bytes: float = 0, spool_max_bytes: float = 0,
                 status_path: Path | None = None, log_event=None):
        self.root = Path(keep_dir).resolve()
        self.spool = spool
        for name, other in (("spool dir", spool.root), ("downloads dir", downloads_dir), ("output dir", out_dir)):
            if other is None:
                continue
            o = Path(other).resolve()
            if o == self.root or o.is_relative_to(self.root) or self.root.is_relative_to(o):
                raise ValueError(f"keep dir {self.root} overlaps {name} {o}")
        self._claim()
        if _st_dev(self.root) != _st_dev(spool.records):
            raise ValueError(f"keep dir {self.root} is not on the spool's filesystem ({spool.root}): files are "
                             "moved by rename, never copied")
        self.max_bytes = None if max_bytes is None or max_bytes <= 0 else int(max_bytes)
        self.floor_bytes = int(floor_bytes or 0)
        self.spool_max_bytes = int(spool_max_bytes or 0)
        self.status_path = Path(status_path) if status_path else None
        self.log_event = log_event or (lambda ev: None)
        self.failures: dict[str, int] = {}
        self.per_record: dict[str, int] = {}
        self.total_bytes = 0
        self._scan()
        self.write_status()

    def _claim(self):
        marker = self.root / KEEP_MARKER
        if self.root.exists():
            if not self.root.is_dir():
                raise ValueError(f"keep dir {self.root} is not a directory")
            if not marker.exists() and any(self.root.iterdir()):
                raise ValueError(f"keep dir {self.root} exists, is not empty and was not created by the survey "
                                 f"(no {KEEP_MARKER} file); pass a new or empty directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text("envision-survey keep dir: fetched files of eye-positive records; the survey "
                              "never deletes here\n", encoding="utf-8")

    def record_dir(self, rid: str) -> Path:
        if not valid_record_id(rid):
            raise ValueError(f"not a Zenodo record id: {rid!r}")
        return self.root / rid

    @staticmethod
    def _dir_bytes(d: Path) -> tuple[int, int]:
        """(bytes, files) under ``d`` without KEPT.json."""
        total = n = 0
        for dirpath, _dirs, files in os.walk(d):
            for fn in files:
                if fn == KEPT and Path(dirpath) == d:
                    continue
                try:
                    total += os.lstat(os.path.join(dirpath, fn)).st_size
                    n += 1
                except OSError:
                    pass
        return total, n

    def _scan(self):
        """Totals from the KEPT.json markers (a dir without one, a move a
        kill interrupted, is measured)."""
        self.per_record.clear()
        try:
            dirs = [p for p in self.root.iterdir() if p.is_dir() and valid_record_id(p.name)]
        except OSError:
            dirs = []
        for d in dirs:
            b = None
            try:
                b = int(json.loads((d / KEPT).read_text(encoding="utf-8")).get("bytes_total"))
            except (OSError, ValueError, TypeError):
                b = None
            if b is None:
                b = self._dir_bytes(d)[0]
            self.per_record[d.name] = b
        self.total_bytes = sum(self.per_record.values())

    def summary(self) -> dict:
        return {"dir": str(self.root), "records": len(self.per_record), "bytes": self.total_bytes,
                "bytes_gb": round(self.total_bytes / 1e9, 3),
                "max_gb": round(self.max_bytes / 1e9, 3) if self.max_bytes else None,
                "updated_epoch": time.time()}

    def write_status(self):
        if self.status_path is None:
            return
        try:
            tmp = self.status_path.with_name(f"{self.status_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.summary()), encoding="utf-8")
            tmp.replace(self.status_path)
        except OSError:
            pass

    # ------------------------------------------------------------ decision
    def _free_if_spool_empty(self, need: int) -> int:
        free = shutil.disk_usage(self.root).free
        try:
            used = self.spool.bytes_used()
        except OSError:
            used = 0
        return free + used - need

    def prepare(self, rid: str, row: dict, manifests: tuple = ()) -> dict:
        """Decide whether record ``rid`` is kept and put the kept fields into
        ``row`` (before the row is written). Returns the plan for move()."""
        src = self.spool.record_dir(rid)
        n_eye = n_eye_images(row)
        local = local_originals(*manifests)
        plan = keep_plan(src) if n_eye else []
        data = [r for r in plan if not _is_meta(r)]
        need = 0
        for rel in plan:
            try:
                need += (src / rel).stat().st_size
            except OSError:
                pass
        data_bytes = 0
        for rel in data:
            try:
                data_bytes += (src / rel).stat().st_size
            except OSError:
                pass
        reason = ""
        if not n_eye:
            reason = "no eye image"
        elif self.max_bytes is not None and self.total_bytes - self.per_record.get(rid, 0) + need > self.max_bytes:
            reason = (f"--keep-max-gb {self.max_bytes / 1e9:g} reached (keep dir {self.total_bytes / 1e9:.1f} GB, "
                      f"record {need / 1e9:.2f} GB)")
        elif need and self._free_if_spool_empty(need) < self.floor_bytes + self.spool_max_bytes:
            reason = (f"disk: keeping would leave less than --disk-floor-gb {self.floor_bytes / 1e9:g} + "
                      f"--spool-max-gb {self.spool_max_bytes / 1e9:g} free with an empty spool")
        kept = not reason
        row["n_eye_images"] = n_eye
        row["kept"] = kept
        row["kept_reason"] = reason
        row["kept_path"] = str(self.record_dir(rid)) if kept else ""
        row["kept_files"] = len(data) if kept else 0
        row["kept_bytes"] = data_bytes if kept else 0
        row["kept_local_files"] = len(local) if kept else 0
        row["kept_local_paths"] = local[:MAX_LOCAL_IN_ROW] if kept else []
        return {"rid": rid, "kept": kept, "local": local, "reason": reason}

    # ---------------------------------------------------------------- move
    def move(self, rid: str, local: list[str] | None = None) -> dict:
        """Rename every planned file of the spool dir of ``rid`` into the
        keep dir (one rename per file), then write KEPT.json. Safe to call
        again after a kill: it moves what is left."""
        src = self.spool.record_dir(rid)
        dst = self.record_dir(rid)
        dst.mkdir(parents=True, exist_ok=True)
        moved = 0
        for rel in keep_plan(src):
            s = src / rel
            d = dst / rel
            if ".." in Path(rel).parts:
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            _rename(s, d)                          # atomic, same filesystem
            moved += 1
        if local is None:
            # a resumed move: the manifests moved along with the files
            mans = []
            for name in (MANIFEST, MANIFEST_DEEP):
                try:
                    mans.append(json.loads((dst / name).read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
            local = local_originals(*mans)
        total, n_files = self._dir_bytes(dst)
        data_bytes = data_files = 0
        for dirpath, _dirs, files in os.walk(dst):
            for fn in files:
                rel = (Path(dirpath) / fn).relative_to(dst).as_posix()
                if rel == KEPT or _is_meta(rel):
                    continue
                try:
                    data_bytes += os.lstat(os.path.join(dirpath, fn)).st_size
                    data_files += 1
                except OSError:
                    pass
        doc = {"record_id": rid, "kept_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "files": data_files, "bytes": data_bytes, "files_total": n_files, "bytes_total": total,
               "local_originals": local}
        _fsync_write(dst / KEPT, json.dumps(doc))
        self.per_record[rid] = total
        self.total_bytes = sum(self.per_record.values())
        self.write_status()
        return {"moved": moved, **doc}


# ---------------------------------------------------------------------------
# backfill: records finished before retention existed
# ---------------------------------------------------------------------------
def _pred_eye_count(path: Path) -> int | None:
    """Eye-class lines (thresholded label, MASK excluded) of a predictions
    file; None when there is no readable file."""
    import gzip
    try:
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("label") in EYE_CLASSES:
                    n += 1
        return n
    except (OSError, EOFError):
        return None


def backfill_ids(results_path: Path, predictions_dir: Path | None = None,
                 keep_dir: Path | None = None) -> tuple[list[str], dict]:
    """Record ids whose last final row was written before retention (no
    ``kept`` field) and holds at least one eye image, by the row's n_<eye
    class> counts or by its predictions file (either suffices). Left out:
    ids that already have a KEPT.json in ``keep_dir``, and rows of the
    pipeline with spool_bytes 0 (only local originals, read in place:
    nothing was fetched, so nothing was deleted). Streams the results file
    (only a few fields per record stay in memory)."""
    from .results import is_final
    last: dict[str, tuple] = {}          # rid -> (final, has_kept, n_eye, n_classified, nothing fetched)
    with open(results_path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rid = str(row.get("record_id") or "")
            if not rid:
                continue
            last[rid] = (is_final(row), "kept" in row, n_eye_images(row), int(row.get("n_classified") or 0),
                         "spool_bytes" in row and not row.get("spool_bytes"))
    by_rows, by_preds, local_only = set(), set(), set()
    n_pre = 0
    for rid, (final, has_kept, n_eye, n_cls, nothing_fetched) in last.items():
        if not final or has_kept:
            continue
        n_pre += 1
        if nothing_fetched:
            if n_eye > 0:
                local_only.add(rid)
            continue
        if n_eye > 0:
            by_rows.add(rid)
        if predictions_dir is not None and n_cls > 0:
            n = _pred_eye_count(Path(predictions_dir) / f"{rid}.jsonl.gz")
            if n:
                by_preds.add(rid)
    ids = by_rows | by_preds
    already = set()
    if keep_dir is not None:
        already = {rid for rid in ids if (Path(keep_dir) / rid / KEPT).is_file()}
        ids -= already
    out = sorted(ids, key=lambda r: (len(r), r))
    summary = {"records_with_rows": len(last), "final_rows_before_retention": n_pre,
               "eye_by_rows": len(by_rows), "eye_by_predictions": len(by_preds),
               "rows_only": sorted(by_rows - by_preds)[:50], "predictions_only": sorted(by_preds - by_rows)[:50],
               "already_kept": len(already), "eye_local_only_not_listed": sorted(local_only)[:50],
               "ids": len(out)}
    return out, summary
