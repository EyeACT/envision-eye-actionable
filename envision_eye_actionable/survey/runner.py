"""Per-record survey loop: fetch, enumerate, sample, classify, describe.

For every record of the scrape (no SetFit filtering):

1. Metadata: legacy record JSON (discovery metadata dir or API) and DataCite
   JSON (API, cached).
2. Files: each Zenodo file is used in place when a complete copy exists under
   ``downloads_dir/<id>/`` (never modified or deleted). A missing zip is not
   downloaded: its members are listed remotely and a seeded sample of at
   most ``remote_cap`` images is fetched member by member (remote.py;
   members over ``remote_max_member_mb`` are never fetched, and at most
   ``remote_record_budget_gb`` is fetched per record). Other
   missing files that can hold images (rar/7z/tar/gz, top-level images) are
   streamed into ``scratch_dir/records/<id>/`` up to ``max_download_gb``
   per record (smallest first, split sets as one unit; the rest is listed
   as over the budget, and ``skipped_size`` only when nothing is readable)
   and when the disk floor allows; that dir is deleted when the record finishes. ``sampling_mode``
   records which of local / remote_zip / download were used. Before that,
   each archive to download is probed (probe.py: its listing read with a
   few range requests) and archives whose whole listing holds no image,
   DICOM, volume or nested archive member are not downloaded; their
   listing is catalogued. A record whose every file to read was probed
   that way gets status ``no_images_in_archives``. Undecided probes keep
   the download (``archive_probe=False`` turns probing off).
3. Enumerate image files, including archive members (archives.RecordWalker).
4. Classify a seeded random sample of at most ``max_images`` images
   (``remote_cap`` for records read only through remote zip sampling) with
   the ONNX model; predictions under ``threshold`` count as UNCERTAIN, and
   threshold-free argmax counts are kept next to them. Every pixel-bearing
   format is converted to an RGB frame first (images.py: TIFF variants,
   SVG, DICOM, volumes and header pairs, arrays, microscopy, vendor OCT,
   video). Masks and label maps (pixel check, and a model argmax that is an
   eye class) count as MASK, not as a modality; blank frames are not
   classified. A record has eye images (status ok) only when eye classes
   make up at least ``min_eye_fraction`` of its non-mask classified images;
   a record whose classified images are at least half masks with fewer than
   50 non-mask images left gets status mask_dominated instead (eye classes
   review-only) unless a confident non-OCTA eye class exempts it. A record
   with no classified image gets no_image_files (nothing pixel-bearing
   listed) or images_unreadable (pixel files listed, none decoded; reasons
   in n_pixel_files_unread_by_reason). When a record mixes
   local or downloaded files with remote zips (sampled at a lower rate),
   class fractions weight each source by its image population.
5. Aggregate DICOM-aligned facts, write the two AI-READI/CMDS JSON files and
   validate them, catalogue weblinks when no usable image or no eye image
   was found. Records that end without images (skipped_disk, error, crashed)
   still get a metadata-only dataset_description from the Zenodo metadata.
6. Append one JSON line to ``survey_results.jsonl``. Records already in that
   file are skipped on restart (unless their status is listed in
   ``retry_statuses``).

Crash safety: a ``{"status": "started"}`` line is appended before a record is
processed. If the process dies inside a record (OOM kill, hang, manual stop),
that marker is the record's last line; the next run turns it into a
``crashed`` row, removes its scratch dir and skips it unless ``crashed`` is in
``retry_statuses``, so one bad record cannot block the rest. Any other stale
``scratch_dir/records`` sub-directory is removed at start as well: one survey
process per scratch dir and per out dir, enforced with a lock file in each
(flock on POSIX, msvcrt on Windows; concurrent processes, for example one
per partition from ``partition_records``, each get their own). Where no file
locking exists the survey warns and does not sweep stale scratch dirs.

Several processes on one disk: with ``shared_state_dir`` each process
publishes the bytes it is about to download (locks.DiskReservations) and
subtracts the other processes' reservations on the same file system from the
free space before checking the disk floor, both for downloads and for remote
member and nested archive fetches. Downloads also re-check the floor while
they stream and stop a file that would cross it.

Scratch safety: the survey only deletes inside a scratch dir it created. On
first use it writes ``.envision_survey_scratch`` into it; it refuses to start
on an existing, non-empty scratch dir without that marker, and every deletion
happens under ``scratch_dir/records/``.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import re as _re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import cmds as cmds_mod
from .archives import RecordWalker
from .classifier import OnnxClassifier
from .constants import (CLASSES, EYE_CLASSES, IMAGE_KINDS, MASK, MASK_DOMINATED_FRAC, MASK_DOMINATED_MIN_NON_MASK,
                        MASK_EXEMPT_MIN_CONF, MASK_EXEMPT_MIN_IMAGES, UNCERTAIN, detect_ext, dominant_of, file_kind,
                        split_first, split_stem, volume_header_of, wanted_for_download)
from .dicom_map import aggregate, ir_device_fields, modality_review, strip_html
from .images import load_frame, preprocess
from .locks import LOCKING_AVAILABLE, DiskReservations, try_lock
from .results import is_final, load_results
from .weblinks import LinkChecker, collect_links
from .zenodo import ZenodoClient, load_scrape, load_token, record_files, redact, safe_filename

RESULTS_NAME = "survey_results.jsonl"
EVENTS_NAME = "survey_events.jsonl"
SCRATCH_MARKER = ".envision_survey_scratch"
SCRATCH_RECORDS = "records"


@dataclass
class SurveyConfig:
    scrape_path: Path
    model_path: Path
    out_dir: Path = Path("./results/survey")
    downloads_dir: Path = Path("./data/downloads/zenodo")
    metadata_dir: Path | None = Path("./data/metadata/zenodo")
    scratch_dir: Path = Path("./data/survey_scratch")
    max_images: int = 2000
    remote_cap: int = 300
    remote_zip: bool = True
    remote_nested_max: int = 20
    remote_nested_gb: float = 2.0
    remote_record_budget_gb: float = 2.0
    remote_max_member_mb: float = 200.0
    max_download_gb: float = 15.0
    seed: int = 42
    threshold: float = 0.6
    batch_size: int = 16
    threads: int = 2
    disk_floor_gb: float = 80.0
    download_workers: int = 3
    download_all: bool = False
    archive_probe: bool = True             # list archives by range reads before downloading them whole
    archive_probe_mb: float = 64.0         # bytes read at most per archive probe
    archive_probe_max_requests: int = 50   # range requests at most per archive probe
    series_sample_k: int = 3               # files downloaded per homogeneous series of large top-level images
    series_min_mb: float = 64.0            # ... whose files are at least this large
    token_file: Path | None = None         # Zenodo token file (ZENODO_TOKEN wins); the token is never logged
    metadata_cache_dir: Path | None = None  # fetched record metadata cache (default: <out_dir>/cache)
    min_class_fraction: float = 0.01
    min_eye_fraction: float = 0.05
    max_pixels: int = 80_000_000
    max_depth: int = 3
    check_links: bool = True
    link_interval: float = 1.0
    zenodo_interval: float = 0.5
    zenodo_per_minute: int = 120
    shared_state_dir: Path | None = None   # 429 cooldown, request budget and disk reservations shared
    zenodo_shared_per_minute: int = 120    # combined requests per minute of the processes sharing that dir
    triage_cap: int = 50                   # images per sampled source on the triage pass (0: one pass)
    # fetch / process / pipeline (spool.py, pipeline.py)
    spool_dir: Path | None = None          # producer writes records here, the consumer deletes them
    state_dir: Path | None = None          # status files, role locks, logs, request budget, 429 cooldown
    spool_max_gb: float = 300.0            # the producer pauses while the spool holds more than this
    fetch_workers: int = 4                 # records fetched at once by the producer
    poll_s: float = 10.0                   # polling interval of the roles
    order: str = "cost"                    # producer order: cost (cheap records first) or scrape
    max_attempts: int = 2                  # consumer starts of one record before it is marked crashed
    consumer_grace_s: float = 600.0        # producer stops when the consumer is gone this long and it must wait
    max_role_restarts: int = 20            # pipeline: restarts of fetch or process (each) before it gives up
    restart_backoff_s: float = 10.0        # pipeline: first restart delay, doubled per restart, at most 300 s
    keep_dir: Path | None = None           # process: fetched files of records with an eye image are moved here
    keep_max_gb: float = 200.0             # keep dir size cap (0: none); past it records are not kept (kept false)
    refetch_ids: list[str] = field(default_factory=list)   # fetch: redo these finished pre-retention records
    keep_scratch: bool = False
    write_predictions: bool = True
    offline: bool = False
    ids: list[str] = field(default_factory=list)
    limit: int | None = None
    retry_statuses: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_event(path: Path, event: dict):
    """Append one event line (registered secrets redacted)."""
    event = {"ts": _now(), **event}
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(redact(json.dumps(event, default=str)) + "\n")


LOCK_NAME = ".envision_survey.lock"


def acquire_lock(directory: Path):
    """Exclusive advisory lock on ``directory`` (a lock file in it), held
    until the returned handle is closed or the process ends. Two survey
    processes must not share an out dir (each treats the other's 'started'
    rows as crashes) or a scratch dir (each sweeps the other's record dirs).
    Raises ValueError when another process holds the lock. Where the
    platform has no file locking (neither fcntl nor msvcrt) it warns and
    returns None: the one-process guarantee is then off."""
    if not LOCKING_AVAILABLE:
        print(f"[survey] warning: no file locking on this platform; nothing stops a second survey process "
              f"from using {directory} (stale scratch dirs are not swept)", file=sys.stderr, flush=True)
        return None
    fp = open(Path(directory) / LOCK_NAME, "a+")
    if not try_lock(fp):
        fp.close()
        raise ValueError(f"{directory} is in use by another survey process; give every concurrent process "
                         "its own --out-dir and --scratch-dir") from None
    return fp


class _RedactingWriter:
    """Text file wrapper that removes registered secrets from every write."""

    def __init__(self, fp):
        self.fp = fp

    def write(self, text: str):
        return self.fp.write(redact(text))

    def close(self):
        self.fp.close()


def _terminate_last_line(path: Path):
    """Append a newline when a killed write left the file without one, so the
    next appended line is not glued onto the broken one."""
    try:
        with open(path, "rb+") as fp:
            fp.seek(0, 2)
            if fp.tell() == 0:
                return
            fp.seek(-1, 2)
            if fp.read(1) != b"\n":
                fp.seek(0, 2)
                fp.write(b"\n")
    except FileNotFoundError:
        pass


class Survey:
    """``role`` run: one process fetches and classifies (scratch dir).
    ``role`` process: the consumer of the pipeline; it reads records the
    fetch role wrote into the spool and makes no Zenodo API request of its
    own (metadata comes from the cache the producer filled; weblink checks
    of zenodo.org links share the request budget through the state dir)."""

    def __init__(self, cfg: SurveyConfig, role: str = "run"):
        self.cfg = cfg
        self.role = role
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._guard_paths()
        if role == "run":
            self._claim_scratch()
            self._locks = [acquire_lock(cfg.out_dir), acquire_lock(cfg.scratch_dir)]
        else:
            self._locks = [acquire_lock(cfg.out_dir)]
        self.results_path = cfg.out_dir / RESULTS_NAME
        self.events_path = cfg.out_dir / EVENTS_NAME
        _terminate_last_line(self.results_path)
        _terminate_last_line(self.events_path)
        self.zenodo = ZenodoClient(cfg.metadata_cache_dir or (cfg.out_dir / "cache"), cfg.metadata_dir,
                                   min_interval=cfg.zenodo_interval, offline=cfg.offline or role == "process",
                                   max_per_minute=cfg.zenodo_per_minute,
                                   shared_state_dir=cfg.shared_state_dir,
                                   shared_per_minute=cfg.zenodo_shared_per_minute,
                                   token=load_token(cfg.token_file) if role == "run" else None)
        self.disk = DiskReservations(cfg.shared_state_dir, cfg.spool_dir if role == "process" else cfg.scratch_dir)
        self.links = LinkChecker(cfg.out_dir / "cache" / "link_status.json",
                                 min_interval=cfg.link_interval, zenodo=self.zenodo) if cfg.check_links else None
        self.clf = OnnxClassifier(cfg.model_path, threads=cfg.threads)
        self.model_info = {"model_file": cfg.model_path.name, **{
            k: self.clf.meta.get(k) for k in ("arch", "checkpoint", "checkpoint_sha256", "onnx_sha256",
                                              "parity_max_abs_diff")}}

    @property
    def records_scratch(self) -> Path:
        """Per-record scratch root: the only place the survey deletes from."""
        return self.cfg.scratch_dir / SCRATCH_RECORDS

    def _guard_paths(self):
        """The scratch dir is deleted from per record: it must never be (or
        contain, or sit inside) the downloads dir, and the output dir must
        not sit inside it. Paths are made absolute first so a relative path
        cannot dodge the check."""
        cfg = self.cfg
        cfg.scratch_dir = cfg.scratch_dir.resolve()
        cfg.downloads_dir = cfg.downloads_dir.resolve()
        cfg.out_dir = cfg.out_dir.resolve()
        scratch, downloads = cfg.scratch_dir, cfg.downloads_dir
        if scratch == downloads or downloads.is_relative_to(scratch) or scratch.is_relative_to(downloads):
            raise ValueError(f"scratch dir {scratch} overlaps downloads dir {downloads}")
        if cfg.out_dir.is_relative_to(scratch):
            raise ValueError(f"output dir {cfg.out_dir} is inside scratch dir {scratch}")

    def _claim_scratch(self):
        """Use only a scratch dir the survey created (marker file), so a data
        folder passed as --scratch-dir by mistake is never emptied."""
        scratch = self.cfg.scratch_dir
        marker = scratch / SCRATCH_MARKER
        if scratch.exists():
            if not scratch.is_dir():
                raise ValueError(f"scratch dir {scratch} is not a directory")
            if not marker.exists() and any(scratch.iterdir()):
                raise ValueError(
                    f"scratch dir {scratch} exists, is not empty and was not created by the survey "
                    f"(no {SCRATCH_MARKER} file); pass a new or empty directory")
        scratch.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text("envision-survey scratch dir; the survey deletes files under records/ only\n",
                              encoding="utf-8")
        self.records_scratch.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        cfg = self.cfg
        all_records = load_scrape(cfg.scrape_path)
        records = list(all_records)
        if cfg.ids:
            want = {str(i) for i in cfg.ids}
            records = [r for r in records if r["source_id"] in want]
        done = load_results(self.results_path)
        crashed = self._recover_crashed(done, {r["source_id"]: r for r in all_records})
        self._sweep_scratch()
        todo = [r for r in records if not is_final(done.get(r["source_id"]), cfg.retry_statuses)]
        if cfg.limit is not None:
            todo = todo[: cfg.limit]
        print(f"[survey] {len(records)} records in scope, {len(records) - len(todo)} already done, "
              f"{len(todo)} to process" + (f", {len(crashed)} crashed earlier: {', '.join(crashed)}"
                                           if crashed else ""), flush=True)
        _log_event(self.events_path, {"event": "run_start", "n_todo": len(todo), "crashed": crashed,
                                      "config": {k: str(v) for k, v in asdict(cfg).items()}})
        stats = Counter()
        try:
            self._run_records(todo, stats)
        finally:
            disk = getattr(self, "disk", None)
            if disk is not None:
                disk.close()
        _log_event(self.events_path, {"event": "run_end", "stats": dict(stats)})
        return dict(stats)

    def _run_records(self, todo: list[dict], stats: Counter):
        for i, rec in enumerate(todo, 1):
            t0 = time.time()
            self._append_row({"record_id": rec["source_id"], "status": "started", "started_at": _now()})
            row = self.process_record(rec)
            row["elapsed_s"] = round(time.time() - t0, 1)
            self._append_row(row)
            if self.links:
                self.links.save()
            stats[row["status"]] += 1
            print(f"[survey] {i}/{len(todo)} {rec['source_id']} {row['status']} "
                  f"images={row.get('n_image_files', 0)} classified={row.get('n_classified', 0)} "
                  f"dominant={row.get('dominant_class') or '-'} ({row['elapsed_s']}s)", flush=True)

    def _append_row(self, row: dict):
        """Append one result line durably (flushed and fsynced: the pipeline
        deletes a record's spool dir right after), secrets redacted."""
        with open(self.results_path, "a", encoding="utf-8") as fp:
            fp.write(redact(json.dumps(row, default=str)) + "\n")
            fp.flush()
            os.fsync(fp.fileno())

    def crashed_row(self, rec: dict, started_at: str | None, in_scrape: bool = True,
                    error: str | None = None) -> dict:
        """Row of a record whose processing never finished (the process died
        inside it), with metadata-only CMDS documents when the metadata is
        at hand."""
        rid = rec["source_id"]
        row = self._base_row(rec)
        row.update({
            "status": "crashed",
            "error": error or ("process ended while this record was running (OOM kill, hang or manual stop); "
                               "skipped on restart, rerun with --retry-status crashed"),
            "started_at": started_at,
            "finished_at": _now(),
        })
        zen = getattr(self, "zenodo", None)
        if zen is not None and in_scrape:
            try:
                legacy, datacite = zen.legacy(rid), zen.datacite(rid)
            except Exception:  # noqa: BLE001 - recovery must not fail on metadata
                legacy = datacite = None
            if legacy is not None or datacite is not None:
                self._write_cmds_metadata_only(row, rec, legacy, datacite, "crashed: no classification")
        return row

    def _recover_crashed(self, done: dict[str, dict], scrape: dict[str, dict]) -> list[str]:
        """Turn records whose last line is 'started' into 'crashed' rows.

        Updates ``done`` in place and returns the crashed record ids.
        """
        crashed = []
        for rid, last in list(done.items()):
            if last.get("status") != "started":
                continue
            rec = scrape.get(rid) or {"source_id": rid}
            row = self.crashed_row(rec, last.get("started_at"), in_scrape=rid in scrape)
            self._append_row(row)
            done[rid] = row
            crashed.append(rid)
            _log_event(self.events_path, {"event": "record_crashed", "record": rid,
                                          "started_at": last.get("started_at")})
            scratch = self.records_scratch / rid
            if scratch.exists() and not self.cfg.keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)
        return crashed

    def _sweep_scratch(self):
        """Remove per-record scratch dirs left by killed runs.

        Only sub-directories of ``scratch_dir/records`` named like a Zenodo
        record id (all digits) are removed; _claim_scratch makes sure the
        scratch dir is the survey's own.
        """
        if self.cfg.keep_scratch or not self.records_scratch.is_dir():
            return
        if not LOCKING_AVAILABLE:
            return        # another process may be using this scratch dir: nothing proves it is stale
        for p in self.records_scratch.iterdir():
            if p.name.isdigit() and p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
                _log_event(self.events_path, {"event": "stale_scratch_removed", "path": str(p)})

    # --------------------------------------------------------------- record
    def _fetcher(self):
        """The fetch half (fetch.RecordFetcher) over this survey's client and
        disk reservations (run role)."""
        from .fetch import RecordFetcher
        disk = getattr(self, "disk", None)
        return RecordFetcher(self.cfg, self.zenodo, log_event=lambda ev: _log_event(self.events_path, ev),
                             reserved_by_others=disk.others if disk is not None else None)

    def process_record(self, rec: dict) -> dict:
        """One record in one process (run role): the triage fetch into the
        scratch dir, classification, and the deep pass right away when the
        triage sample asks for it. The scratch dir is removed at the end."""
        cfg = self.cfg
        rid = rec["source_id"]
        rdir = self.records_scratch / rid
        row = None
        try:
            fetcher = self._fetcher()
            m = fetcher.fetch(rec, rdir)
            listings = m.pop("_listings", None)
            row, want_deep = self.finish_record(rec, rdir, m, allow_deep_request=True)
            if want_deep:
                dm = fetcher.fetch_deep(rec, rdir, m, listings)
                row, _ = self.finish_record(rec, rdir, m, dm, allow_deep_request=False)
        except Exception as e:  # noqa: BLE001 - never let one record kill the run
            row = row or self._base_row(rec)
            row["status"] = "error"
            row["error"] = redact(f"{type(e).__name__}: {e}")[:500]
            row["traceback"] = redact(traceback.format_exc())[-2000:]
            _log_event(self.events_path, {"event": "record_error", "record": rid, "error": row["error"]})
        finally:
            if not cfg.keep_scratch and rdir.exists():
                shutil.rmtree(rdir, ignore_errors=True)
        row["finished_at"] = _now()
        return row

    def finish_record(self, rec: dict, rdir: Path, m: dict, dm: dict | None = None,
                      allow_deep_request: bool = True) -> tuple[dict | None, bool]:
        """The classify half of a record from its fetch manifest ``m`` (and
        the deep pass manifest ``dm``): walk the fetched and local files,
        classify the triage sample, then (eye classes in the triage sample)
        the deep sample, describe, write CMDS. Returns (row, False), or
        (None, True) when the deep pass needs files that were not fetched
        yet and ``allow_deep_request``: the caller fetches them and calls
        again with ``dm``. Never raises (status error)."""
        cfg = self.cfg
        rid = rec["source_id"]
        row = self._base_row(rec)
        row.update(m.get("row") or {})
        if dm:
            row.update(dm.get("row") or {})
            if dm.get("error"):
                row["deep_fetch_error"] = dm["error"]
        legacy = datacite = None
        self._agg_images = []
        self._extra_paths: list[str] = []
        self._remote_member_ext = {}
        work = rdir / "work"
        try:
            legacy = self.zenodo.legacy(rid)
            datacite = self.zenodo.datacite(rid)
            term = m.get("terminal")
            if term:
                return self._finish_terminal(row, rec, legacy, datacite, term), False

            items = list(m.get("items") or []) + list((dm or {}).get("items") or [])
            remote = ((dm or {}).get("remote") or m.get("remote"))
            unfetched_top = list((dm if dm is not None else m).get("unfetched_top") or [])
            for key in ("n_files_remote_zip", "n_files_local", "n_members_skipped_oversize",
                        "n_images_unsampleable", "remote_bytes_fetched"):
                row.setdefault(key, 0)
            row.setdefault("fraction_scope_note", "")

            def item_path(it):
                return Path(it["path"]) if it.get("local") else rdir / it["path"]

            files_items = [it for it in items if it.get("role") in ("local", "download")]
            item_pass = {str(item_path(it)): it.get("pass") or "triage" for it in files_items}
            available = [(item_path(it), it["key"]) for it in files_items]
            available = _colocate_split_sets(available, rdir / "dl")
            available, join_notes = join_split_sets(available, work / "joined", walker_room=self._room_bytes)
            if join_notes:
                row["split_join_detail"] = "; ".join(join_notes)[:1000]

            # ---- enumerate images (archives listed, not unpacked)
            disk = getattr(self, "disk", None)
            walker = RecordWalker(scratch=work, max_depth=cfg.max_depth,
                                  disk_floor_bytes=int(cfg.disk_floor_gb * 1e9),
                                  reserved_by_others=disk.others if disk is not None else None)
            work.mkdir(parents=True, exist_ok=True)
            entry_pass: dict[int, str] = {}
            for p, key in sorted(available, key=lambda t: t[1]):
                n0 = len(walker.entries)
                walker.add_file(p, key)
                for e in walker.entries[n0:]:
                    entry_pass[id(e)] = item_pass.get(str(p), "triage")
            n_probe_negative = self._add_probe_listings(walker, m.get("probe_negative"))
            n_seen = len(walker.entries) + len(unfetched_top)     # images seen in listings
            n_est = float(n_seen)                                 # population estimate
            n_file_entries = len(walker.entries)
            # Sampling strata: local and downloaded files, remote zip
            # members, images in nested archives fetched from remote zips.
            # Class fractions weight each stratum by its population.
            strata = {"files": n_est}
            entry_stratum: dict[int, str] = {}
            self._extra_paths = list(unfetched_top)
            if remote:
                n_seen_remote, remote_strata, entry_stratum = self._remote_into_walker(
                    row, remote, items, rdir, walker, entry_pass, first_entry=n_file_entries)
                n_seen += n_seen_remote
                n_est += sum(remote_strata.values())
                strata.update(remote_strata)
            row["n_images_listed"] = n_seen
            row["n_image_files"] = int(round(n_est))
            row["image_ext_counts"] = dict(walker.image_ext_counts.most_common(12))
            row["file_kind_counts"] = dict(walker.kind_counts)
            member_ext = Counter(walker.member_ext_counts)
            member_ext.update(getattr(self, "_remote_member_ext", {}) or {})
            for r in m.get("probe_negative") or []:
                member_ext.update(r.get("ext_counts") or {})
            row["member_ext_counts"] = dict(member_ext.most_common(25))
            row["n_archives_listed"] = (len(walker.containers) + int(row.get("n_files_remote_zip") or 0)
                                        + n_probe_negative)
            row["walk_errors"] = len(walker.errors)
            row["walk_error_detail"] = redact("; ".join(walker.errors[:5]))[:1000]

            # ---- sample and classify (triage sample, then the deep sample)
            triage = {"cap": int(cfg.triage_cap or 0) or None,
                      "fetch_available": bool(m.get("deep_available")) and dm is None,
                      "allow_request": allow_deep_request}
            if self._classify(row, rid, walker, cap=cfg.max_images, population=row["n_image_files"],
                              strata=strata, entry_stratum=entry_stratum, entry_pass=entry_pass,
                              triage=triage) == "deep":
                return None, True

            # ---- record-level DICOM-aligned facts and CMDS docs
            if row.get("n_classified", 0) > 0:
                row["status"] = "ok"
                self._describe(row, rec, legacy, datacite, walker)
                # Empty for a record with no eye class present (checked in
                # ir_device_fields), so the device fields match the status.
                row.update(ir_device_fields(row))
                if not row.get("present_eye_classes"):
                    # Images, but none of an eye modality (plots, masks, CT),
                    # or mostly masks with a small remainder: the eye data
                    # may live behind an external link.
                    row["status"] = "mask_dominated" if row.get("mask_dominated") else "no_eye_images"
                    self._catalogue_links(row, rec, legacy, datacite)
            elif (remote and row.get("sampling_mode") == "remote_zip"
                  and row.get("remote_listing_failures") == len(remote.get("zips") or [])):
                # Every zip listing failed (429s, 5xx, broken bodies): a
                # transient outcome, retried with --retry-status.
                row["status"] = "remote_listing_failed"
                self._finish_without_images(row, rec, legacy, datacite, walker)
            elif n_probe_negative and row.get("sampling_mode") == "none":
                # Every file the survey would read was an archive whose
                # probed listing holds no image member: nothing downloaded.
                row["status"] = "no_images_in_archives"
                self._finish_without_images(row, rec, legacy, datacite, walker)
            else:
                # No classified image. images_unreadable: pixel-bearing
                # files were listed (decode failed, fetch failed, over a
                # size cap, not sampled, blank); no_image_files: none were.
                unread = Counter(row.get("n_pixel_files_unread_by_reason") or {})
                # a sample that read files and found every one not pixel-bearing
                # (arrays of signals, text .fds) says the same of the unsampled rest
                sample_all_non_pixel = bool(row.get("n_sampled")) and bool(unread) and all(
                    k in NON_PIXEL_REASONS for k in unread)
                if row.get("n_blank"):
                    unread["blank"] += row["n_blank"]
                if unfetched_top:
                    unread["not_fetched"] += len(unfetched_top)
                if row.get("n_images_unsampleable"):
                    unread["oversize_member"] += row["n_images_unsampleable"]
                if row.get("n_remote_over_budget"):
                    unread["over_record_budget"] += row["n_remote_over_budget"]
                n_over = sum(1 for f in row.get("skipped_size_files") or [] if file_kind(f["key"]) in IMAGE_KINDS)
                if n_over:
                    unread["over_download_budget"] += n_over
                unsampled = max(0, (row.get("n_images_listed") or 0) - (row.get("n_sampled") or 0)
                                - len(unfetched_top) - (row.get("n_remote_over_budget") or 0))
                if unsampled:
                    unread["not_sampled_like_sample" if sample_all_non_pixel else "not_sampled"] += unsampled
                row["n_pixel_files_unread_by_reason"] = dict(unread.most_common())
                pixel_unread = sum(v for k, v in unread.items() if k not in NON_PIXEL_REASONS)
                row["status"] = "images_unreadable" if pixel_unread else "no_image_files"
                self._finish_without_images(row, rec, legacy, datacite, walker)
            if row["status"] != "remote_listing_failed" and (
                    row.get("download_failures") or row.get("remote_fetch_failures")
                    or row.get("remote_listing_failures")):
                row["status"] += "_partial_download"
        except Exception as e:  # noqa: BLE001 - never let one record kill the run
            row["status"] = "error"
            row["error"] = redact(f"{type(e).__name__}: {e}")[:500]
            row["traceback"] = redact(traceback.format_exc())[-2000:]
            _log_event(self.events_path, {"event": "record_error", "record": rid, "error": row["error"]})
            if (legacy is not None or datacite is not None) and not row.get("cmds_dir"):
                self._write_cmds_metadata_only(row, rec, legacy, datacite,
                                               f"error before the record finished: {row['error'][:200]}")
        finally:
            if work.exists() and not cfg.keep_scratch:
                shutil.rmtree(work, ignore_errors=True)
        return row, False

    def _finish_terminal(self, row: dict, rec: dict, legacy, datacite, term: dict) -> dict:
        """Rows of records the fetch half ended early (no files, restricted,
        skipped_size, skipped_disk, metadata_unavailable, fetch error)."""
        row["status"] = term["status"]
        if term.get("error"):
            row["error"] = redact(term["error"])
        if term.get("traceback"):
            row["traceback"] = redact(term["traceback"])
        row.setdefault("sampling_mode", "none")
        finish = term.get("finish")
        if finish == "without_images":
            self._finish_without_images(row, rec, legacy, datacite)
        elif finish == "metadata_links":
            self._write_cmds_metadata_only(row, rec, legacy, datacite, term.get("note") or row["status"])
            self._catalogue_links(row, rec, legacy, datacite)
        elif finish == "links_only":
            self._catalogue_links(row, rec, legacy, datacite)
        elif legacy is not None or datacite is not None:
            note = term.get("note") or f"error before the record finished: {str(row.get('error'))[:200]}"
            self._write_cmds_metadata_only(row, rec, legacy, datacite, note)
        return row

    def _base_row(self, rec: dict) -> dict:
        return {
            "record_id": rec["source_id"],
            "doi": rec.get("doi"),
            "title": rec.get("title"),
            "url": rec.get("url") or f"https://zenodo.org/records/{rec['source_id']}",
            "license": rec.get("license"),
            "access_type": rec.get("access_type"),
            "size_mb": rec.get("size_mb"),
            "scrape_file_count": rec.get("file_count"),
            "setfit_label": rec.get("label"),
            "setfit_prob_eye_imaging": rec.get("prob_eye_imaging"),
            "keywords": "; ".join(rec.get("keywords") or [])[:500],
            "status": "pending",
            "survey_version": 7,           # 7: two-pass sampling (triage, deep), fetch / process pipeline
            "triage_cap": self.cfg.triage_cap,
            "threshold": self.cfg.threshold,
            "max_images": self.cfg.max_images,
            "remote_cap": self.cfg.remote_cap,
            "max_download_gb": self.cfg.max_download_gb,
            "remote_record_budget_gb": self.cfg.remote_record_budget_gb,
            "remote_max_member_mb": self.cfg.remote_max_member_mb,
            "min_eye_fraction": self.cfg.min_eye_fraction,
            "seed": self.cfg.seed,
            **{f"model_{k}": v for k, v in self.model_info.items()},
        }

    def _room_bytes(self, nbytes: int) -> bool:
        """Room for ``nbytes`` more on the working disk above the floor,
        minus what other survey processes reserved."""
        disk = getattr(self, "disk", None)
        base = self.cfg.spool_dir if getattr(self, "role", "run") == "process" else self.cfg.scratch_dir
        free = disk.free() if disk is not None else shutil.disk_usage(base).free
        return free - nbytes >= self.cfg.disk_floor_gb * 1e9

    @staticmethod
    def _add_probe_listings(walker: RecordWalker, negatives: list[dict] | None) -> int:
        """Catalogue the listings of the archives probed negative (member
        kinds, names for the manufacturer hints). Returns how many."""
        for r in negatives or []:
            walker.kind_counts.update(r.get("kind_counts") or {})
            room = max(0, 5000 - len(walker.other_names))
            walker.other_names.extend(f"{r['key']}!/{n}" for n in (r.get("names") or [])[:room])
        return len(negatives or [])

    # --------------------------------------------------------------- remote
    def _remote_into_walker(self, row: dict, rs: dict, items: list[dict], rdir: Path, walker: RecordWalker,
                            entry_pass: dict[int, str], first_entry: int = 0
                            ) -> tuple[int, dict[str, float], dict[int, str]]:
        """Remote zip members and nested archives fetched by the fetch half
        (summary ``rs``, files in ``items``) into the walker, and the remote
        columns of the row.

        Returns (images seen in the listings and fetched nested archives,
        {stratum: population estimate}, {id(entry): stratum}). The two remote
        strata are sampled at different rates: ``remote`` holds the direct
        image members fetched (a sample of the listed ones), ``remote_nested``
        the images inside the nested archives fetched (all of them, from a
        sample of the listed archives). Walker entries before ``first_entry``
        belong to the local files and are not tagged."""
        cfg = self.cfg
        nested_ids: set[int] = set()
        n_direct_items = 0
        added = 0
        for it in items:
            if it.get("role") not in ("remote", "remote_nested"):
                continue
            path = rdir / it["path"]
            n0 = len(walker.entries)
            walker.add_file(path, it["key"])
            for e in walker.entries[n0:]:
                entry_pass[id(e)] = it.get("pass") or "triage"
                if it["role"] == "remote_nested":
                    nested_ids.add(id(e))
            if it["role"] == "remote":
                n_direct_items += 1
                added += len(walker.entries) - n0
        entry_stratum = {id(e): ("remote_nested" if id(e) in nested_ids else "remote")
                         for e in walker.entries[first_entry:]}
        walker.kind_counts.update(rs.get("member_kind_counts") or {})
        self._remote_member_ext = dict(rs.get("member_ext_counts") or {})
        walker.image_ext_counts.update(rs.get("unfetched_image_ext_counts") or {})
        room = max(0, 5000 - len(walker.other_names))
        walker.other_names.extend((rs.get("other_member_names") or [])[:room])
        self._extra_paths += list(rs.get("unfetched_image_paths") or [])
        # Members over --remote-max-member-mb (and empty ones) are outside the
        # sampling frame, so they are left out of the stratum population as
        # well: large members can be a different population (full-resolution
        # scans next to small masks or thumbnails), and projecting the class
        # mix of the small ones onto them would be a guess. They are reported
        # as n_images_unsampleable, unclassified and not in n_image_files.
        oversize = int(rs.get("n_members_skipped_oversize") or 0)
        zero = int(rs.get("n_members_zero_size") or 0)

        def in_frame(n: int) -> int:
            return max(0, n - oversize - zero)

        pool = rs.get("pool") or ""
        if pool == "noext":
            # Extensionless members: the DICOM share of the fetched sample
            # is extrapolated to all the sampleable ones.
            seen = added
            est = in_frame(int(rs.get("n_noext_listed") or 0)) * added / n_direct_items if n_direct_items else 0.0
        else:
            seen = int(rs.get("n_images_listed") or 0)
            est = float(in_frame(seen))
        n_nested_fetched = int(rs.get("n_nested_fetched") or 0)
        nested_images = int(rs.get("nested_images") or 0)
        nested_est = (nested_images * int(rs.get("n_nested_listed") or 0) / n_nested_fetched
                      if n_nested_fetched else 0.0)
        row.update({
            "remote_listing_detail": redact(rs.get("listing_detail") or ""),
            "remote_listing_sources": rs.get("listing_sources") or "",
            "remote_listings_truncated": rs.get("listings_truncated") or 0,
            "remote_pool": pool,
            "n_remote_images_listed": int(rs.get("n_images_listed") or 0),
            "n_remote_noext_listed": int(rs.get("n_noext_listed") or 0),
            "n_remote_nested_listed": int(rs.get("n_nested_listed") or 0),
            "n_remote_nested_fetched": n_nested_fetched,
            "n_remote_nested_images": nested_images,
            "n_remote_sampled": int(rs.get("n_sampled") or 0),
            "n_remote_fetched": int(rs.get("n_fetched") or 0),
            "n_remote_reused": int(rs.get("n_reused") or 0),
            "remote_listing_failures": int(rs.get("n_listing_failures") or 0),
            "n_remote_nested_over_cap": int(rs.get("n_nested_over_cap") or 0),
            "n_remote_nested_over_budget": int(rs.get("n_nested_over_budget") or 0),
            "remote_nested_bytes": int(rs.get("nested_bytes") or 0),
            "remote_nested_stop": rs.get("nested_stop") or "",
            "remote_fetch_failures": int(rs.get("n_failures") or 0),
            "remote_fetch_failure_detail": redact("; ".join((rs.get("failures") or [])[:5]))[:1000],
            "remote_requests": int(rs.get("n_requests") or 0),
            "n_members_skipped_oversize": oversize,
            "n_members_zero_size": zero,
            "n_images_unsampleable": oversize,
            "fraction_scope_note": (
                f"class fractions and est_files describe remote zip members up to --remote-max-member-mb "
                f"{cfg.remote_max_member_mb:g} only; {oversize} larger "
                f"{'extensionless ' if pool == 'noext' else 'image '}member(s) were not sampled, "
                f"are unclassified and are not counted in n_image_files" if oversize else ""),
            "n_remote_over_budget": int(rs.get("n_members_over_budget") or 0),
            "remote_budget_stop": rs.get("budget_stop") or "",
            # measured bytes received (failed transfers and retries included),
            # the quantity --remote-record-budget-gb bounds
            "remote_bytes_fetched": int(rs.get("bytes_received") or 0),
            "remote_bytes_fetched_ok": int(rs.get("members_bytes") or 0) + int(rs.get("nested_bytes") or 0),
        })
        strata = {"remote": est}
        if nested_ids or nested_est:
            strata["remote_nested"] = nested_est
        return seen + nested_images, strata, entry_stratum

    # ------------------------------------------------------------ classify
    def _classify(self, row: dict, rid: str, walker: RecordWalker, cap: int | None = None,
                  population: int | None = None, strata: dict[str, float] | None = None,
                  entry_stratum: dict[int, str] | None = None, entry_pass: dict[int, str] | None = None,
                  triage: dict | None = None) -> str | None:
        """Classify a seeded sample of the walker entries: at most ``cap``
        (default max_images) local and downloaded files and at most
        remote_cap remote zip images. ``population`` (default: the walker's
        entry count) scales the per-class file estimates.

        Two passes (``triage`` = {cap, fetch_available, allow_request}, cap
        None for one pass): each source group (files; remote members and
        nested archive images) is put in one seeded order, entries fetched
        by the triage pass first (``entry_pass``). The triage sample is the
        first ``triage["cap"]`` of each group. When its non-mask thresholded
        labels hold eye classes at ``min_eye_fraction`` or more, the deep
        sample follows: the rest of each group's order up to its cap (the
        triage images are kept, not classified again). When the deep sample
        needs files the fetch half has not fetched yet (``fetch_available``)
        and ``allow_request``, nothing is written and "deep" is returned:
        the caller fetches them and runs the record again. Otherwise returns
        None. Row: sampling_pass (single, triage or deep), triage_cap,
        n_triage_classified, triage_eye_fraction, deep_pass_reason.

        ``strata`` maps a stratum name to its image population and
        ``entry_stratum`` maps id(entry) to its stratum (default "files").
        The strata are "files" (local and downloaded files), "remote" (direct
        members of remote zips) and "remote_nested" (images inside the nested
        archives fetched from remote zips). When the classified images come
        from more than one stratum (sampled at different rates), each image
        counts with weight population / classified of its stratum, and the
        frac_*, argmax_frac_*, mean_*, dominant and est_files values come
        from the weighted counts. The n_* and argmax_* columns stay raw
        sample counts.

        One file gives one prediction, except: a video's frames (images
        .frames) are classified together and their probabilities averaged;
        a vendor OCT file's fundus or SLO image (images .views) is a
        prediction of its own (path ``file#fundus``). Constant (blank)
        frames are not classified (n_blank).

        Mask rule (images.is_mask_like flags masks, label maps and also
        flat-colour graphics): a flagged image is labelled MASK only when the
        model's argmax is an eye class; with argmax NEG it keeps the model
        label (NEG or UNCERTAIN, ``mask_like_veto`` in the predictions
        file), since plots, word clouds and logos are what trips the pixel
        check there. MASK images are left out of the dominant classes, mean
        confidence and probabilities, eye image fraction and eye class
        presence: the eye classes together must reach ``min_eye_fraction`` of
        the non-mask images (thresholded labels), and each listed eye class
        ``min_class_fraction``; otherwise no eye class is present and the
        record gets status no_eye_images. A mask-dominated record (frac_MASK
        at least MASK_DOMINATED_FRAC and fewer than
        MASK_DOMINATED_MIN_NON_MASK non-mask images) lists no eye class
        either (``review_eye_classes``, status mask_dominated), unless its
        non-mask images hold at least MASK_EXEMPT_MIN_IMAGES confident
        (mean top-1 at least MASK_EXEMPT_MIN_CONF) images of one eye class
        other than OCTA (paired segmentation datasets: photos next to their
        masks). OCTA does not exempt: missed masks come out as OCTA."""
        strata = strata or {}
        entry_stratum = entry_stratum or {}
        entry_pass = entry_pass or {}
        triage = triage or {}
        cfg = self.cfg
        cap = cfg.max_images if cap is None else cap
        entries = sorted(walker.entries, key=lambda e: e.display)
        population = len(entries) if population is None else population
        tcap = triage.get("cap")
        stage1, stage2 = [], []
        for group, gcap, salt in (("files", cap, ""), ("remote", cfg.remote_cap, ":remote")):
            members = [e for e in entries if (entry_stratum.get(id(e), "files") == "files") == (group == "files")]
            first = [e for e in members if entry_pass.get(id(e), "triage") != "deep"]
            later = [e for e in members if entry_pass.get(id(e), "triage") == "deep"]
            rng = random.Random(f"{cfg.seed}:{rid}{salt}")
            rng.shuffle(first)
            rng.shuffle(later)
            order = (first + later)[:gcap]
            # the triage sample holds triage-pass entries only, so a record
            # run again with its deep files gets the same triage decision
            n1 = len(order) if tcap is None else min(tcap, len(first), gcap)
            stage1 += order[:n1]
            stage2 += order[n1:]

        preds: list[dict] = []          # per classified image
        errors = Counter()
        unread = Counter()              # files that gave no image, by reason
        fmt_ok, fmt_bad, conversions = Counter(), Counter(), Counter()
        n_blank = 0
        batch: list[np.ndarray] = []
        batch_meta: list[tuple] = []    # (display, facts, n arrays, stratum)
        pred_fp = None
        pred_path = cfg.out_dir / "predictions" / f"{rid}.jsonl.gz"
        pred_part = pred_path.with_name(pred_path.name + ".part")
        if cfg.write_predictions:
            (cfg.out_dir / "predictions").mkdir(exist_ok=True)
            pred_fp = _RedactingWriter(gzip.open(pred_part, "wt", encoding="utf-8"))

        def flush():
            if not batch:
                return
            probs_all = self.clf.predict(np.stack(batch))
            k = 0
            for display, facts, n_arr, stratum in batch_meta:
                pr = probs_all[k:k + n_arr].mean(axis=0)
                k += n_arr
                top = int(pr.argmax())
                conf = float(pr[top])
                model_label = CLASSES[top] if conf >= cfg.threshold else UNCERTAIN
                flagged = bool(facts.get("mask_like"))
                # Masks and label maps get no modality (MASK in the
                # thresholded and the argmax counts alike) when the model
                # would call them an eye class; flagged images the model
                # calls NEG are graphics and keep the model label.
                veto = flagged and CLASSES[top] == "NEG"
                mask = flagged and not veto
                label = MASK if mask else model_label
                preds.append({"cls": label, "top": MASK if mask else CLASSES[top], "conf": conf, "probs": pr,
                              "facts": facts, "path": display, "mask": mask, "stratum": stratum})
                if pred_fp:
                    rec = {"path": display, "label": label, "top": CLASSES[top],
                           "p": round(conf, 4), "probs": [round(float(x), 4) for x in pr],
                           "rows": facts.get("rows"), "cols": facts.get("cols"),
                           "frames": facts.get("frames"), "format": facts.get("source_format"),
                           "conversion": facts.get("conversion")}
                    if n_arr > 1:
                        rec["n_frames_averaged"] = n_arr
                    if mask:
                        rec.update({"mask_like": True, "model_label": model_label})
                    elif veto:
                        rec.update({"mask_like": True, "mask_like_veto": True})
                    pred_fp.write(json.dumps(rec) + "\n")
            batch.clear()
            batch_meta.clear()

        def error_line(display: str, err: str, facts: dict | None, reason: str):
            unread[reason] += 1
            if facts:
                fmt_bad[facts.get("source_format") or facts.get("format") or "?"] += 1
            if pred_fp:
                line = {"path": display, "error": err, "reason": reason}
                for key in ("rows", "cols", "frames", "format", "source_format", "compression", "bits",
                            "samples", "conversion"):
                    if facts and facts.get(key) is not None:
                        line[key] = facts[key]
                pred_fp.write(json.dumps(line, default=str) + "\n")

        def add(display: str, img, facts: dict, extra: list, stratum: str) -> bool:
            """Queue one prediction (image plus averaged frames); False when
            blank or not preprocessable."""
            nonlocal n_blank
            if facts.get("blank"):
                n_blank += 1
                if pred_fp:
                    pred_fp.write(json.dumps({"path": display, "blank": True, "rows": facts.get("rows"),
                                              "cols": facts.get("cols"),
                                              "format": facts.get("source_format")}) + "\n")
                return False
            try:
                arrs = [preprocess(img)] + [preprocess(f) for f in extra]
            except Exception as e:  # noqa: BLE001
                errors[f"preprocess {type(e).__name__}"] += 1
                error_line(display, f"preprocess {type(e).__name__}: {e}"[:200], facts, "decode_error")
                return False
            batch.extend(arrs)
            batch_meta.append((display, facts, len(arrs), stratum))
            fmt_ok[facts.get("source_format") or facts.get("format") or "?"] += 1
            if facts.get("conversion"):
                conversions[re.sub(r"\d+", "#", str(facts["conversion"]))[:80]] += 1
            return True

        inventory_only = []   # facts of files we could not decode (still useful)

        def run_stage(sample: list):
            for entry, data, path, err in walker.read_entries(sample):
                if err:
                    err = redact(err)
                    errors[err.split(":")[0][:60]] += 1
                    error_line(entry.display, err, None, unread_reason(err, fetch=True))
                    continue
                loaded = load_frame(entry.kind, entry.ext, data, path, cfg.max_pixels)
                del data
                if loaded.image is None:
                    errors[(loaded.error or "undecodable").split(":")[0][:60]] += 1
                    inventory_only.append(loaded.facts)
                    error_line(entry.display, redact(loaded.error or "undecodable"), loaded.facts,
                               unread_reason(loaded.error or ""))
                    continue
                stratum = entry_stratum.get(id(entry), "files")
                add(entry.display, loaded.image, loaded.facts, loaded.frames, stratum)
                for suffix, vimg, vfacts in loaded.views:
                    add(f"{entry.display}#{suffix}", vimg,
                        {**{k: v for k, v in loaded.facts.items() if k in ("make", "model", "source_format")},
                         **vfacts, "view": suffix}, [], stratum)
                if len(batch) >= cfg.batch_size:
                    flush()
            flush()

        request_deep = False
        try:
            run_stage(stage1)
            n_sampled = len(stage1)
            if tcap is None:
                row["sampling_pass"] = "single"
            else:
                nm1 = [p for p in preds if not p["mask"]]
                eye1 = (sum(1 for p in nm1 if p["cls"] in EYE_CLASSES) / len(nm1)) if nm1 else None
                row["triage_cap"] = tcap
                row["n_triage_classified"] = len(preds)
                row["triage_eye_fraction"] = round(eye1, 4) if eye1 is not None else None
                wants = eye1 is not None and eye1 >= cfg.min_eye_fraction
                if not wants:
                    row["sampling_pass"] = "triage"
                    row["deep_pass_reason"] = ("no non-mask image in the triage sample" if eye1 is None else
                                               f"triage eye fraction {eye1:.3f} < {cfg.min_eye_fraction:g}")
                elif triage.get("fetch_available") and triage.get("allow_request"):
                    request_deep = True
                else:
                    row["sampling_pass"] = "deep" if stage2 else "triage"
                    row["deep_pass_reason"] = (f"triage eye fraction {eye1:.3f} >= {cfg.min_eye_fraction:g}"
                                               + ("" if stage2 else "; nothing more to sample"))
                    run_stage(stage2)
                    n_sampled += len(stage2)
            row["n_sampled"] = n_sampled
        finally:
            if pred_fp:
                pred_fp.close()
                if request_deep:
                    pred_part.unlink(missing_ok=True)
                else:
                    pred_part.replace(pred_path)
        if request_deep:
            self._agg_images = []
            return "deep"

        n = len(preds)
        row["n_classified"] = n
        row["n_blank"] = n_blank
        row["n_load_errors"] = sum(errors.values())
        row["load_error_top"] = "; ".join(f"{k} ({v})" for k, v in errors.most_common(4))
        row["n_inventory_only"] = len(inventory_only)
        row["n_pixel_files_unread_by_reason"] = dict(unread.most_common())
        row["formats_classified"] = dict(fmt_ok.most_common(12))
        row["formats_unread"] = dict(fmt_bad.most_common(12))
        row["conversion_counts"] = dict(conversions.most_common(12))
        row["n_mask_like_veto"] = 0
        counts = Counter(p["cls"] for p in preds)          # raw sample counts
        raw = Counter(p["top"] for p in preds)
        # Stratum weights (1.0 for all when one stratum was classified).
        by_stratum = Counter(p["stratum"] for p in preds)
        if len(by_stratum) > 1:
            weight = {s: max(float(strata.get(s, k)), k) / k for s, k in by_stratum.items()}
        else:
            weight = {s: 1.0 for s in by_stratum}
        for p in preds:
            p["w"] = weight[p["stratum"]]
        total_w = sum(p["w"] for p in preds)
        wcounts, wraw = Counter(), Counter()
        for p in preds:
            wcounts[p["cls"]] += p["w"]
            wraw[p["top"]] += p["w"]
        labels = CLASSES + [UNCERTAIN, MASK]
        frac = {c: wcounts.get(c, 0) / total_w for c in labels} if n else {}
        row["class_weighting"] = "stratified" if len(by_stratum) > 1 else "uniform"
        row["n_classified_by_source"] = dict(by_stratum)
        row["n_mask_like_veto"] = sum(1 for p in preds if p["facts"].get("mask_like") and not p["mask"])
        for c in labels:
            row[f"n_{c}"] = counts.get(c, 0)
            row[f"frac_{c}"] = round(frac[c], 4) if n else None
        for c in CLASSES + [MASK]:
            row[f"argmax_{c}"] = raw.get(c, 0)
            row[f"argmax_frac_{c}"] = round(wraw.get(c, 0) / total_w, 4) if n else None
        # Masks carry no modality: the dominant classes, the mean confidence
        # and probabilities, and the eye decision below are taken over the
        # other images. MASK is dominant only when every image is a mask.
        nm = [p for p in preds if not p["mask"]]
        nm_w = sum(p["w"] for p in nm)
        row["n_classified_non_mask"] = len(nm)
        # IR share of the non-mask images (the IR trigger in ir_device_fields
        # uses these, so masks cannot dilute a substantial IR share).
        row["frac_IR_non_mask"] = round(wcounts.get("IR", 0) / nm_w, 4) if nm else None
        row["argmax_frac_IR_non_mask"] = round(wraw.get("IR", 0) / nm_w, 4) if nm else None
        row["argmax_dominant_class"] = ((dominant_of({c: wraw.get(c, 0) for c in CLASSES}, CLASSES) if nm
                                         else MASK) if n else None)
        row["mean_confidence"] = round(float(sum(p["w"] * p["conf"] for p in nm) / nm_w), 4) if nm else None
        mean_probs = (np.sum([p["w"] * np.asarray(p["probs"], dtype=np.float64) for p in nm], axis=0) / nm_w
                      if nm else None)
        for i, c in enumerate(CLASSES):
            row[f"mean_prob_{c}"] = round(float(mean_probs[i]), 4) if nm else None
        # Dominant label counts UNCERTAIN too, so a record of out-of-distribution
        # images is reported as UNCERTAIN rather than by a handful of hits.
        row["dominant_class"] = ((dominant_of({c: wcounts.get(c, 0) for c in CLASSES + [UNCERTAIN]}) if nm
                                  else MASK) if n else None)
        # Eye decision over the non-mask images, thresholded labels: the eye
        # classes together must reach min_eye_fraction (else the record has
        # no eye images), and each listed class min_class_fraction.
        nm_frac = {c: wcounts.get(c, 0) / nm_w for c in EYE_CLASSES} if nm else {}
        eye_frac = sum(nm_frac.values()) if nm else None
        present = [c for c in EYE_CLASSES
                   if counts.get(c, 0) >= 1 and nm_frac[c] >= cfg.min_class_fraction] \
            if nm and eye_frac >= cfg.min_eye_fraction else []
        # Mask-dominated record: when masks are at least MASK_DOMINATED_FRAC
        # of the classified images (weighted, frac_MASK) and fewer than
        # MASK_DOMINATED_MIN_NON_MASK images are left, the eye decision would
        # rest on a small remainder (often mask-like images the pixel check
        # missed). The eye classes found there are listed for review only
        # (review_eye_classes), not as present: status mask_dominated, no
        # CMDS modality directories. Exempt: a confident eye class (not
        # OCTA) among the non-mask images (mask_exempt_class).
        exempt = mask_exempt_class(nm)
        row["mask_exempt_class"] = exempt or ""
        row["mask_dominated"] = bool(n and frac[MASK] >= MASK_DOMINATED_FRAC
                                     and len(nm) < MASK_DOMINATED_MIN_NON_MASK and not exempt)
        row["review_eye_classes"] = ", ".join(present) if row["mask_dominated"] else ""
        if row["mask_dominated"]:
            present = []
        row["dominant_eye_class"] = dominant_of({c: wcounts[c] for c in present}, EYE_CLASSES) if present else None
        row["present_eye_classes"] = ", ".join(present)
        row["eye_image_fraction"] = round(eye_frac, 4) if eye_frac is not None else None
        row["_class_detail"] = {
            c: {"count": counts[c],
                "est_files": int(round(population * frac[c])) if n else 0,
                "mean_conf": float(np.mean([p["conf"] for p in preds if p["cls"] == c]))}
            for c in present}
        # Facts of classified images for aggregation (probs dropped to save RAM).
        # "top" (the threshold-free argmax class) goes along so the IR device
        # evidence covers the same images as the argmax-or-threshold IR trigger.
        self._agg_images = [{"cls": p["cls"], "top": p["top"], "facts": p["facts"], "path": p["path"]}
                            for p in preds]
        self._agg_images += [{"cls": UNCERTAIN, "facts": f, "path": ""} for f in inventory_only if f]

    # ------------------------------------------------------------ describe
    def _record_text(self, rec: dict, legacy: dict | None) -> str:
        lmeta = (legacy or {}).get("metadata") or {}
        return " ".join([rec.get("title") or "", strip_html(lmeta.get("description") or rec.get("description") or ""),
                         " ".join(rec.get("keywords") or [])])

    def _describe(self, row: dict, rec: dict, legacy, datacite, walker: RecordWalker):
        present = [c for c in (row.get("present_eye_classes") or "").split(", ") if c]
        # DICOM columns follow the dominant eye modality when one is present,
        # else the dominant class (NEG -> OT; UNCERTAIN -> no mapping).
        dominant = row.get("dominant_eye_class") or row.get("dominant_class")
        classes_for_map = list(present)
        if row.get("n_NEG"):
            classes_for_map.append("NEG")
        paths = [e.display for e in walker.entries] + list(getattr(self, "_extra_paths", []) or [])
        agg = aggregate(self._agg_images, paths, classes_for_map,
                        dominant if dominant in CLASSES else None, self._record_text(rec, legacy),
                        walker.other_names)
        row.update(agg)
        row.update(modality_review(row))
        self._write_cmds(row, rec, legacy, datacite, present, walker)

    def _finish_without_images(self, row, rec, legacy, datacite, walker: RecordWalker | None = None):
        """No usable images: DICOM status n/a, CMDS docs, weblink catalogue."""
        row.setdefault("n_image_files", len(walker.entries) if walker else 0)
        row.setdefault("n_classified", 0)
        row["dicom_mapping_status"] = "not_applicable"
        from .dicom_map import manufacturer_hints
        hints = manufacturer_hints(self._record_text(rec, legacy), walker.other_names if walker else [])
        row["manufacturer_hint_text"] = "; ".join(f"{h['manufacturer']} [{h['matched']}]" for h in hints)
        self._agg_images = []
        self._write_cmds(row, rec, legacy, datacite, [], walker)
        self._catalogue_links(row, rec, legacy, datacite)

    def _catalogue_links(self, row, rec, legacy, datacite):
        """Weblinks of a record whose own files gave no eye images."""
        links = collect_links(rec["source_id"], rec, legacy, datacite)
        if self.links:
            for link in links:
                link.update(self.links.check(link["url"]))
        row["weblinks"] = links
        row["n_weblinks"] = len(links)
        row["n_weblinks_dataset_likely"] = sum(1 for link in links if link["dataset_likely"])

    def _write_cmds_metadata_only(self, row, rec, legacy, datacite, note: str):
        """dataset_description from the Zenodo metadata alone; the structure
        description carries no classification (empty directoryList)."""
        try:
            self._agg_images = []
            row.pop("_class_detail", None)
            self._write_cmds(row, rec, legacy, datacite, [], None)
            row["cmds_note"] = note
        except Exception as e:  # noqa: BLE001 - metadata docs must never hide the record status
            row["cmds_note"] = f"{note}; metadata-only CMDS failed: {type(e).__name__}: {e}"[:500]

    def _write_cmds(self, row, rec, legacy, datacite, present, walker):
        summary = {
            "present_classes": present,
            "total_bytes": row.get("zenodo_bytes") or int((rec.get("size_mb") or 0) * 1e6),
            "n_files": row.get("n_files_zenodo"),
            "n_images": row.get("n_image_files"),
            "image_ext_counts": row.get("image_ext_counts") or {},
            "files_ext_counts": row.get("files_ext_counts") or {},
        }
        dd, prov = cmds_mod.build_dataset_description(rec, legacy, datacite, summary)
        dicom_classes = sorted({im["cls"] for im in (self._agg_images or [])
                                if (im.get("facts") or {}).get("dicom")})
        dsd = cmds_mod.build_structure_description(row.get("_class_detail") or {}, present, {
            "n_images": row.get("n_image_files", 0), "n_classified": row.get("n_classified", 0),
            "dicom_classes": dicom_classes,
            "formats": sorted((row.get("image_ext_counts") or {}).keys()),
            "review_note": row.get("modality_review_flags") if present else "",
        })
        out = self.cfg.out_dir / "cmds" / rec["source_id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset_description.json").write_text(redact(json.dumps(dd, indent=2, ensure_ascii=False)),
                                                      encoding="utf-8")
        (out / "dataset_structure_description.json").write_text(
            redact(json.dumps(dsd, indent=2, ensure_ascii=False)), encoding="utf-8")
        dd_err = cmds_mod.validate(dd, "dataset_description")
        dsd_err = cmds_mod.validate(dsd, "dataset_structure_description")
        row["cmds_dir"] = out.relative_to(self.cfg.out_dir).as_posix()   # relative to the results dir
        row["dicom_classes_sampled"] = ", ".join(dicom_classes)
        row.update(cmds_mod.flatten_dd(dd))
        row["dd_valid"] = not dd_err
        row["dd_errors"] = "; ".join(dd_err[:5])
        row["dsd_valid"] = not dsd_err
        row["dsd_errors"] = "; ".join(dsd_err[:5])
        row["cmds_placeholder_fields"] = ", ".join(k for k, v in prov.items() if v.startswith("placeholder"))
        row["cmds_derived_fields"] = ", ".join(k for k, v in prov.items() if v.startswith("derived"))
        row["cmds_classifier_fields"] = ", ".join(k for k, v in prov.items() if "classifier" in v)
        row["cmds_provenance"] = prov
        row["cmds_resourceTypeValue"] = dd["resourceType"]["resourceTypeValue"]
        row["cmds_creators"] = "; ".join(c["creatorName"] for c in dd["creator"])[:500]
        row["cmds_publicationYear"] = dd["publicationYear"]
        row["cmds_version"] = dd["version"]
        row["cmds_accessType"] = dd["accessType"]
        row["cmds_rights"] = "; ".join(r["rightsName"] for r in dd["rights"])
        row["cmds_managingOrganization"] = dd["managingOrganization"]["name"]
        row["cmds_directories"] = "; ".join(
            f"{d['directoryName']}/{s['directoryName']} ({s.get('numberOfFiles')})"
            for d in dsd["directoryList"] for s in d.get("directoryList", []))


def plan_files(files: list[dict], local_dir: Path, remote_zip: bool = True, offline: bool = False,
               download_all: bool = False) -> tuple[list[tuple[Path, str]], list[dict], list[dict], int]:
    """How each Zenodo file of a record will be read: (local copies as
    (path, key), zips to sample remotely, files to download whole, number of
    files not needed). A local copy counts only when complete (same size)."""
    available: list[tuple[Path, str]] = []
    remote_zips: list[dict] = []
    to_fetch: list[dict] = []
    skipped = 0
    keys = {f["key"].lower() for f in files}
    all_keys = [f["key"] for f in files]
    for f in files:
        p = local_dir / safe_filename(f["key"])
        if p.exists() and (not f["size"] or p.stat().st_size == f["size"]):
            available.append((p, f["key"]))
        elif remote_zip and not offline and remote_zip_candidate(f["key"], keys):
            remote_zips.append(f)
        elif (download_all or wanted_for_download(f["key"])
              or (file_kind(f["key"]) == "volume_data" and volume_header_of(f["key"], all_keys))):
            # a volume data file (x.raw, x.img) only next to its header
            to_fetch.append(f)
        else:
            skipped += 1
    return available, remote_zips, to_fetch, skipped


PARTITION_MODES = ("local", "remote_zip", "download", "none")


def expected_mode(files: list[dict], local_dir: Path, remote_zip: bool = True, download_all: bool = False) -> str:
    """The partition a record falls in, by the Zenodo work it needs.

    remote_zip: at least one zip is sampled remotely (hundreds of requests
    per record, few bytes each: the request-heavy partition), whatever else
    the record has. download: no remote zip, but files to download whole
    (one request per file, many bytes: the bandwidth-heavy partition).
    local: everything needed is on disk (metadata requests only). none: no
    files, or none an image survey reads."""
    available, remote_zips, to_fetch, _ = plan_files(files, local_dir, remote_zip=remote_zip,
                                                     download_all=download_all)
    if remote_zips:
        return "remote_zip"
    if to_fetch:
        return "download"
    if available:
        return "local"
    return "none"


def partition_records(scrape_path: Path, downloads_dir: Path, metadata_dirs: list[Path], out_dir: Path,
                      remote_zip: bool = True, download_all: bool = False) -> dict:
    """Write ``ids_<mode>.txt`` (one record id per line, scrape order) for
    the modes in PARTITION_MODES, plus ``ids_unknown.txt`` for records with
    no legacy metadata file in ``metadata_dirs`` (the discovery metadata dir,
    a survey cache ``cache/legacy`` dir); no Zenodo request is made. Returns
    and writes ``partition_summary.json`` with the counts."""
    groups: dict[str, list[str]] = {m: [] for m in (*PARTITION_MODES, "unknown")}
    for rec in load_scrape(scrape_path):
        rid = rec["source_id"]
        legacy = None
        for d in metadata_dirs:
            p = Path(d) / f"{rid}.json"
            if p.is_file():
                try:
                    legacy = json.loads(p.read_text(encoding="utf-8"))
                    break
                except (OSError, ValueError):
                    continue
        if legacy is None:
            groups["unknown"].append(rid)
            continue
        groups[expected_mode(record_files(legacy), Path(downloads_dir) / rid, remote_zip=remote_zip,
                             download_all=download_all)].append(rid)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, ids in groups.items():
        (out_dir / f"ids_{mode}.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    summary = {"counts": {m: len(v) for m, v in groups.items()}, "total": sum(len(v) for v in groups.values()),
               "files": {m: f"ids_{m}.txt" for m in groups}}
    (out_dir / "partition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_ids_file(path: Path) -> list[str]:
    """Record ids from a text file: whitespace or comma separated, '#' starts
    a comment. Order kept, duplicates dropped."""
    out: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        for tok in re.split(r"[\s,]+", line.split("#", 1)[0]):
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


def fit_budget(files: list[dict], budget: float) -> tuple[list[dict], list[dict]]:
    """Split ``files`` into (kept, over): whole units, smallest first, while
    the running total stays within ``budget`` bytes. The parts of a split
    archive (x.part1.rar, x.r00, x.z01, x.7z.001) form one unit, since a
    partial set cannot be read. Both lists keep the input order."""
    units: dict[str, list[int]] = {}
    for i, f in enumerate(files):
        units.setdefault(_split_key(f["key"]) or f"file:{i}", []).append(i)
    order = sorted(units.values(), key=lambda idx: (sum(files[i]["size"] for i in idx), idx[0]))
    keep: set[int] = set()
    total = 0
    for idx in order:
        size = sum(files[i]["size"] for i in idx)
        if total + size > budget:
            break              # sorted ascending: nothing later fits either
        keep.update(idx)
        total += size
    return ([f for i, f in enumerate(files) if i in keep], [f for i, f in enumerate(files) if i not in keep])


def remote_zip_candidate(key: str, all_keys: set[str]) -> bool:
    """A Zenodo file the remote sampler can read: a plain .zip that is not
    the first part of a spanned set (x.zip next to x.z01)."""
    low = key.lower()
    if not low.endswith(".zip"):
        return False
    return f"{low[:-4]}.z01" not in all_keys


_SPLIT_PATTERNS = [
    re.compile(r"^(?P<stem>.+)\.(?:z\d{2,3}|zip)$"),
    re.compile(r"^(?P<stem>.+)\.part\d+\.rar$"),
    re.compile(r"^(?P<stem>.+)\.(?:r\d{2,3}|rar)$"),
    re.compile(r"^(?P<stem>.+\.(?:7z|zip))\.\d{3}$"),
]


def _split_key(name: str) -> str | None:
    low = name.lower()
    stem = split_stem(name)
    if stem is not None:
        return f"g:{stem.lower()}"
    for i, rx in enumerate(_SPLIT_PATTERNS):
        m = rx.match(low)
        if m:
            return f"{i}:{m.group('stem')}"
    return None


def _colocate_split_sets(available: list[tuple[Path, str]], dl_dir: Path) -> list[tuple[Path, str]]:
    """Split archives need all parts in one directory. When a set is partly
    local and partly streamed, symlink the local parts next to the streamed
    ones (no copy, and the originals are never touched)."""
    sets: dict[str, list[int]] = {}
    for i, (_p, key) in enumerate(available):
        k = _split_key(key)
        if k:
            sets.setdefault(k, []).append(i)
    out = list(available)
    for idxs in sets.values():
        if len(idxs) < 2:
            continue
        in_dl = [i for i in idxs if out[i][0].parent == dl_dir]
        if not in_dl or len(in_dl) == len(idxs):
            continue
        for i in idxs:
            p, key = out[i]
            if p.parent == dl_dir:
                continue
            link = dl_dir / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
            out[i] = (link, key)
    return out


UNREAD_REASONS = (
    ("too_large", "too_large"), ("companion", "companion_missing"), ("volume data", "companion_missing"),
    ("image-shaped", "not_image_shaped"), ("image shaped", "not_image_shaped"),
    ("not an hdf5", "not_image_shaped"), ("proprietary", "no_reader"), ("no reader", "no_reader"),
    ("no pixel data", "no_pixel_data"), ("pixel decode", "codec"), ("memoryerror", "memory"),
    ("disk floor", "disk_floor"), ("wrong_format", "wrong_format"),
)
# Reasons that show a file was not pixel-bearing after all (an HDF5 of model
# weights, a Fire Dynamics Simulator .fds): they do not make a record
# images_unreadable.
NON_PIXEL_REASONS = {"not_image_shaped", "wrong_format", "no_pixel_data", "not_sampled_like_sample"}


def unread_reason(err: str, fetch: bool = False) -> str:
    """Short reason a pixel-bearing file gave no image (the
    n_pixel_files_unread_by_reason keys)."""
    low = (err or "").lower()
    for needle, reason in UNREAD_REASONS:
        if needle in low:
            return reason
    return "fetch_failed" if fetch else "decode_error"


def mask_exempt_class(non_mask: list[dict]) -> str | None:
    """The eye class that exempts a record from mask_dominated: a class
    other than OCTA with at least MASK_EXEMPT_MIN_IMAGES thresholded
    non-mask images whose mean top-1 confidence is at least
    MASK_EXEMPT_MIN_CONF (the most such images wins). None otherwise."""
    best = None
    for c in EYE_CLASSES:
        if c == "OCTA":
            continue
        confs = [p["conf"] for p in non_mask if p["cls"] == c]
        if len(confs) >= MASK_EXEMPT_MIN_IMAGES and float(np.mean(confs)) >= MASK_EXEMPT_MIN_CONF:
            if best is None or len(confs) > best[1]:
                best = (c, len(confs))
    return best[0] if best else None


def series_key(key: str) -> tuple:
    """Series of a top-level file: kind, extension and name with every digit
    run replaced (x_001.tif and x_002.tif share one)."""
    name = key.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return file_kind(key), detect_ext(key), _re_digits.sub("#", name)


_re_digits = re.compile(r"\d+")


def sample_series(files: list[dict], k: int, min_bytes: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Seeded ``k`` files per homogeneous series of large top-level images
    (same series_key, each at least ``min_bytes``; more than ``k`` of them):
    one land-cover GeoTIFF of 900 MB says what the other 29 are. Returns
    (files to fetch, series files left out). Smaller files and every other
    kind are kept. k <= 0 keeps everything."""
    if k <= 0:
        return files, []
    groups: dict[tuple, list[int]] = {}
    for i, f in enumerate(files):
        if file_kind(f["key"]) in IMAGE_KINDS and int(f.get("size") or 0) >= min_bytes:
            groups.setdefault(series_key(f["key"]), []).append(i)
    drop: set[int] = set()
    for idx in groups.values():
        if len(idx) > k:
            keep = set(rng.sample(sorted(idx, key=lambda i: files[i]["key"]), k))
            drop.update(i for i in idx if i not in keep)
    return ([f for i, f in enumerate(files) if i not in drop], [f for i, f in enumerate(files) if i in drop])


def join_split_sets(available: list[tuple[Path, str]], out_dir: Path, walker_room=None
                    ) -> tuple[list[tuple[Path, str]], list[str]]:
    """Concatenate generic split sets (x.tar.001 + x.tar.002, x.zip.partaa +
    x.zip.partab, x.part01 + x.part02) into one file under ``out_dir`` so
    the walker reads them as one archive. The parts are replaced in the
    returned list by the joined file (named after the set, with .7z added
    when that name has no archive extension, so 7z detects the format);
    originals are never modified. Sets whose parts are not all present, or
    that 7z reads itself (x.7z.001, x.zip.001), are left alone.
    ``walker_room(nbytes)`` guards the disk. Returns (files, notes)."""
    sets: dict[str, list[int]] = {}
    for i, (_p, key) in enumerate(available):
        low = key.lower()
        if low.endswith((".7z.001", ".7z.002")) or _re.search(r"\.(?:7z|zip)\.\d{3}$", low):
            continue
        stem = split_stem(key)
        if stem is not None:
            sets.setdefault(stem.lower(), []).append(i)
    if not sets:
        return available, []
    notes = []
    drop: set[int] = set()
    added = []
    for stem, idx in sets.items():
        parts = sorted(idx, key=lambda i: available[i][1].lower())
        if len(parts) < 2 or not split_first(available[parts[0]][1]):
            continue
        total = sum(available[i][0].stat().st_size for i in parts)
        if walker_room is not None and not walker_room(total):
            notes.append(f"{available[parts[0]][1]}: split set not joined (disk floor)")
            continue
        name = split_stem(available[parts[0]][1]).replace("\\", "/").rsplit("/", 1)[-1] or "joined"
        if file_kind(name) not in ("archive", *IMAGE_KINDS):
            name += ".7z"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / safe_filename(name)
        with open(dest, "wb") as out:
            for i in parts:
                with open(available[i][0], "rb") as src:
                    shutil.copyfileobj(src, out, 8 << 20)
        drop.update(parts)
        added.append((dest, split_stem(available[parts[0]][1]) or name))
        notes.append(f"{len(parts)} parts joined into {name}")
    out = [t for i, t in enumerate(available) if i not in drop] + added
    return out, notes


def run_survey(cfg: SurveyConfig) -> dict:
    return Survey(cfg).run()
