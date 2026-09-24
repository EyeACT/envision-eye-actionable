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
   records which of local / remote_zip / download were used.
3. Enumerate image files, including archive members (archives.RecordWalker).
4. Classify a seeded random sample of at most ``max_images`` images
   (``remote_cap`` for records read only through remote zip sampling) with
   the ONNX model; predictions under ``threshold`` count as UNCERTAIN, and
   threshold-free argmax counts are kept next to them. Masks and label maps
   (pixel check) count as MASK, not as a modality. A record has eye images
   (status ok) only when eye classes make up at least ``min_eye_fraction``
   of its non-mask classified images; a record whose classified images are
   at least half masks with fewer than 50 non-mask images left gets status
   mask_dominated instead (eye classes review-only). When a record mixes
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
import random
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import cmds as cmds_mod
from .archives import RecordWalker
from .classifier import OnnxClassifier
from .constants import (CLASSES, EYE_CLASSES, IMAGE_KINDS, MASK, MASK_DOMINATED_FRAC, MASK_DOMINATED_MIN_NON_MASK,
                        UNCERTAIN, detect_ext, dominant_of, file_kind, wanted_for_download)
from .dicom_map import aggregate, ir_device_fields, modality_review, strip_html
from .images import load_frame, preprocess
from .locks import LOCKING_AVAILABLE, DiskReservations, try_lock
from .remote import sample_remote_zips
from .results import load_results
from .weblinks import LinkChecker, collect_links
from .zenodo import ZenodoClient, load_scrape, record_files, safe_filename, stream_download

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
    min_class_fraction: float = 0.01
    min_eye_fraction: float = 0.05
    max_pixels: int = 80_000_000
    max_depth: int = 3
    check_links: bool = True
    link_interval: float = 1.0
    zenodo_interval: float = 0.5
    zenodo_per_minute: int = 110
    shared_state_dir: Path | None = None   # 429 cooldown, request budget and disk reservations shared
    zenodo_shared_per_minute: int = 110    # combined requests per minute of the processes sharing that dir
    keep_scratch: bool = False
    write_predictions: bool = True
    offline: bool = False
    ids: list[str] = field(default_factory=list)
    limit: int | None = None
    retry_statuses: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_event(path: Path, event: dict):
    event = {"ts": _now(), **event}
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, default=str) + "\n")


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
    def __init__(self, cfg: SurveyConfig):
        self.cfg = cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._guard_paths()
        self._claim_scratch()
        self._locks = [acquire_lock(cfg.out_dir), acquire_lock(cfg.scratch_dir)]
        self.results_path = cfg.out_dir / RESULTS_NAME
        self.events_path = cfg.out_dir / EVENTS_NAME
        _terminate_last_line(self.results_path)
        _terminate_last_line(self.events_path)
        self.zenodo = ZenodoClient(cfg.out_dir / "cache", cfg.metadata_dir,
                                   min_interval=cfg.zenodo_interval, offline=cfg.offline,
                                   max_per_minute=cfg.zenodo_per_minute,
                                   shared_state_dir=cfg.shared_state_dir,
                                   shared_per_minute=cfg.zenodo_shared_per_minute)
        self.disk = DiskReservations(cfg.shared_state_dir, cfg.scratch_dir)
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
        todo = [r for r in records
                if r["source_id"] not in done or done[r["source_id"]].get("status") in cfg.retry_statuses]
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
        with open(self.results_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, default=str) + "\n")
            fp.flush()

    def _recover_crashed(self, done: dict[str, dict], scrape: dict[str, dict]) -> list[str]:
        """Turn records whose last line is 'started' into 'crashed' rows.

        Updates ``done`` in place and returns the crashed record ids.
        """
        crashed = []
        for rid, last in list(done.items()):
            if last.get("status") != "started":
                continue
            rec = scrape.get(rid) or {"source_id": rid}
            row = self._base_row(rec)
            row.update({
                "status": "crashed",
                "error": ("process ended while this record was running (OOM kill, hang or manual stop); "
                          "skipped on restart, rerun with --retry-status crashed"),
                "started_at": last.get("started_at"),
                "finished_at": _now(),
            })
            zen = getattr(self, "zenodo", None)
            if zen is not None and rid in scrape:
                try:
                    legacy, datacite = zen.legacy(rid), zen.datacite(rid)
                except Exception:  # noqa: BLE001 - recovery must not fail on metadata
                    legacy = datacite = None
                if legacy is not None or datacite is not None:
                    self._write_cmds_metadata_only(row, rec, legacy, datacite,
                                                   "crashed: no classification")
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
    def process_record(self, rec: dict) -> dict:
        cfg = self.cfg
        rid = rec["source_id"]
        row = self._base_row(rec)
        scratch = self.records_scratch / rid
        legacy = datacite = None
        self._agg_images = []
        self._extra_paths: list[str] = []
        try:
            legacy = self.zenodo.legacy(rid)
            datacite = self.zenodo.datacite(rid)
            row["zenodo_legacy_ok"] = legacy is not None
            row["zenodo_datacite_ok"] = datacite is not None
            lmeta = (legacy or {}).get("metadata") or {}
            access = (lmeta.get("access_right") or rec.get("access_type") or "").lower()
            row["access_right"] = access

            files = record_files(legacy)
            row["n_files_zenodo"] = len(files)
            row["zenodo_bytes"] = sum(f["size"] for f in files)
            row["files_ext_counts"] = dict(Counter(detect_ext(f["key"]) for f in files).most_common(15))
            if not files:
                row["status"] = "restricted" if access and access != "open" else "no_files"
                row["sampling_mode"] = "none"
                self._finish_without_images(row, rec, legacy, datacite)
                return row

            # ---- plan: local copies, remote zip sampling, whole-file downloads
            available, remote_zips, to_fetch, skipped = plan_files(
                files, cfg.downloads_dir / rid, remote_zip=cfg.remote_zip, offline=cfg.offline,
                download_all=cfg.download_all)
            n_local_files = len(available)
            # Top-level image files beyond the remote cap: a seeded sample is
            # downloaded, the rest is counted (and its names read) only.
            top_images = sorted((f for f in to_fetch if file_kind(f["key"]) in IMAGE_KINDS), key=lambda f: f["key"])
            unfetched_top: list[str] = []
            if len(top_images) > cfg.remote_cap:
                keep = {f["key"] for f in random.Random(f"{cfg.seed}:{rid}:files").sample(top_images, cfg.remote_cap)}
                unfetched_top = [f["key"] for f in top_images if f["key"] not in keep]
                to_fetch = [f for f in to_fetch if f["key"] in keep or file_kind(f["key"]) not in IMAGE_KINDS]
            row["n_files_local"] = n_local_files
            row["n_files_remote_zip"] = len(remote_zips)
            row["n_members_skipped_oversize"] = 0      # set by _sample_remote when zips are sampled
            row["n_images_unsampleable"] = 0
            row["fraction_scope_note"] = ""
            row["remote_bytes_fetched"] = 0
            row["n_files_to_download"] = len(to_fetch)
            row["n_files_not_needed"] = skipped
            row["n_toplevel_images_not_fetched"] = len(unfetched_top)
            need = sum(f["size"] for f in to_fetch)
            row["download_bytes_planned"] = need
            if to_fetch and need > cfg.max_download_gb * 1e9:
                # Fill the budget smallest first (split archive sets as one
                # unit), so one huge RAR does not also drop the small files.
                to_fetch, over = fit_budget(to_fetch, cfg.max_download_gb * 1e9)
                over_bytes = sum(f["size"] for f in over)
                row["n_files_skipped_size"] = len(over)
                row["skipped_size_files"] = [{"key": f["key"], "size": f["size"], "url": f.get("url")}
                                             for f in over[:500]]
                _log_event(self.events_path, {"event": "skipped_size", "record": rid, "need": need,
                                              "n_over": len(over), "over_bytes": over_bytes})
                note = (f"{len(over)} non-zip file(s), {over_bytes / 1e9:.1f} GB, did not fit "
                        f"--max-download-gb {cfg.max_download_gb:g}")
                if not available and not remote_zips and not to_fetch:
                    row["status"] = "skipped_size"
                    row["sampling_mode"] = "none"
                    row["error"] = note
                    row["files_list"] = [{"key": f["key"], "size": f["size"], "url": f.get("url")}
                                         for f in files[:500]]
                    self._write_cmds_metadata_only(row, rec, legacy, datacite,
                                                   "skipped_size: files not read, no classification")
                    self._catalogue_links(row, rec, legacy, datacite)
                    return row
                row["size_note"] = note + "; those files were not read"
                row["n_files_to_download"] = len(to_fetch)
            need = sum(f["size"] for f in to_fetch)
            disk = getattr(self, "disk", None)
            # Free space minus the downloads other survey processes have in
            # flight on the same disk (with --shared-state-dir).
            free = disk.free() if disk is not None else shutil.disk_usage(cfg.scratch_dir).free
            if to_fetch and free - need < cfg.disk_floor_gb * 1e9:
                row["status"] = "skipped_disk"
                row["sampling_mode"] = "none"
                row["error"] = (f"needs {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB, "
                                f"floor {cfg.disk_floor_gb:.0f} GB")
                _log_event(self.events_path, {"event": "skipped_disk", "record": rid, "need": need, "free": free})
                self._write_cmds_metadata_only(row, rec, legacy, datacite,
                                               "skipped_disk: files not read, no classification")
                return row

            # ---- stream the missing non-zip files
            if to_fetch:
                dl_dir = scratch / "dl"
                if disk is not None:
                    disk.reserve(need)            # released once the files are on disk
                try:
                    failures = self._download(to_fetch, dl_dir, rid)
                finally:
                    if disk is not None:
                        disk.release()
                row["download_failures"] = len(failures)
                row["download_failure_detail"] = "; ".join(f"{k}: {e}" for k, e in failures[:5])
                for f in to_fetch:
                    p = dl_dir / safe_filename(f["key"])
                    if p.exists():
                        available.append((p, f["key"]))
                row["bytes_downloaded"] = sum(p.stat().st_size for p, _ in available if p.is_relative_to(scratch))
                available = _colocate_split_sets(available, dl_dir)
            modes = [m for m, used in (("local", n_local_files), ("remote_zip", remote_zips), ("download", to_fetch))
                     if used]
            row["sampling_mode"] = "+".join(modes) or "none"

            # ---- enumerate images (archives listed, not unpacked)
            walker = RecordWalker(scratch=scratch, max_depth=cfg.max_depth,
                                  disk_floor_bytes=int(cfg.disk_floor_gb * 1e9),
                                  reserved_by_others=disk.others if disk is not None else None)
            scratch.mkdir(parents=True, exist_ok=True)
            for p, key in sorted(available, key=lambda t: t[1]):
                walker.add_file(p, key)
            n_seen = len(walker.entries) + len(unfetched_top)     # images seen in listings
            n_est = float(n_seen)                                 # population estimate
            n_file_entries = len(walker.entries)
            # Sampling strata: local and downloaded files (sampled at up to
            # max_images) and remote zips (sampled at remote_cap). Class
            # fractions weight each stratum by its population, so a small
            # local zip cannot outvote a large remote one.
            strata = {"files": n_est}
            entry_stratum: dict[int, str] = {}
            self._extra_paths = list(unfetched_top)
            if remote_zips:
                n_seen_remote, remote_strata, entry_stratum = self._sample_remote(
                    row, rid, remote_zips, scratch, walker, first_entry=n_file_entries)
                n_seen += n_seen_remote
                n_est += sum(remote_strata.values())
                strata.update(remote_strata)
            row["n_images_listed"] = n_seen
            row["n_image_files"] = int(round(n_est))
            row["image_ext_counts"] = dict(walker.image_ext_counts.most_common(12))
            row["file_kind_counts"] = dict(walker.kind_counts)
            row["n_archives_listed"] = len(walker.containers) + len(remote_zips)
            row["walk_errors"] = len(walker.errors)
            row["walk_error_detail"] = "; ".join(walker.errors[:5])[:1000]

            # ---- sample and classify
            cap = cfg.remote_cap if row["sampling_mode"] == "remote_zip" else cfg.max_images
            self._classify(row, rid, walker, cap=cap, population=row["n_image_files"],
                           strata=strata, entry_stratum=entry_stratum)

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
            elif (remote_zips and row["sampling_mode"] == "remote_zip"
                  and row.get("remote_listing_failures") == len(remote_zips)):
                # Every zip listing failed (429s, 5xx, broken bodies): a
                # transient outcome, retried with --retry-status.
                row["status"] = "remote_listing_failed"
                self._finish_without_images(row, rec, legacy, datacite, walker)
            else:
                row["status"] = "no_usable_images"
                self._finish_without_images(row, rec, legacy, datacite, walker)
            if row["status"] != "remote_listing_failed" and (
                    row.get("download_failures") or row.get("remote_fetch_failures")
                    or row.get("remote_listing_failures")):
                row["status"] += "_partial_download"
        except Exception as e:  # noqa: BLE001 - never let one record kill the run
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"[:500]
            row["traceback"] = traceback.format_exc()[-2000:]
            _log_event(self.events_path, {"event": "record_error", "record": rid, "error": row["error"]})
            if (legacy is not None or datacite is not None) and not row.get("cmds_dir"):
                self._write_cmds_metadata_only(row, rec, legacy, datacite,
                                               f"error before the record finished: {row['error'][:200]}")
        finally:
            if not cfg.keep_scratch and scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
            row["finished_at"] = _now()
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
            "survey_version": 4,           # 4: dominance mask test, mask_dominated status
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

    def _download(self, files: list[dict], dl_dir: Path, rid: str) -> list[tuple[str, str]]:
        dl_dir.mkdir(parents=True, exist_ok=True)
        session = self.zenodo.session
        failures = []
        floor = self.cfg.disk_floor_gb * 1e9

        def room() -> bool:
            # Checked while the file streams: another process may have used
            # the space since the check before the downloads.
            return shutil.disk_usage(dl_dir).free >= floor

        def one(f):
            dest = dl_dir / safe_filename(f["key"])
            if not f.get("url"):
                return f["key"], "no download URL"
            # The client's throttle is shared by all workers and by the
            # metadata calls; a 429 on any of them pauses every request.
            ok, err = stream_download(session, f["url"], dest, f["size"] or None, f.get("md5"),
                                      throttle=self.zenodo, room_check=room)
            return f["key"], (None if ok else err)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.download_workers)) as ex:
            for key, err in ex.map(one, files):
                if err:
                    failures.append((key, err))
                    _log_event(self.events_path, {"event": "download_failed", "record": rid,
                                                  "file": key, "error": err})
        return failures

    # --------------------------------------------------------------- remote
    def _sample_remote(self, row: dict, rid: str, zips: list[dict], scratch: Path,
                       walker: RecordWalker, first_entry: int = 0
                       ) -> tuple[int, dict[str, float], dict[int, str]]:
        """Remote zip sampling into the walker.

        Returns (images seen in the listings and fetched nested archives,
        {stratum: population estimate}, {id(entry): stratum}). The two remote
        strata are sampled at different rates: ``remote`` holds the direct
        image members fetched (a sample of the listed ones), ``remote_nested``
        the images inside the nested archives fetched (all of them, from a
        sample of the listed archives). Walker entries before ``first_entry``
        belong to the local files and are not tagged."""
        cfg = self.cfg
        nested_ids: set[int] = set()

        def on_nested(path: Path, display: str) -> int:
            n0 = len(walker.entries)
            walker.add_file(path, display)
            walker._drop_if_unused(path)
            nested_ids.update(id(e) for e in walker.entries[n0:])
            return len(walker.entries) - n0

        res = sample_remote_zips(
            self.zenodo, rid, zips, scratch / "remote", cfg.remote_cap, random.Random(f"{cfg.seed}:{rid}:remote"),
            workers=cfg.download_workers, nested_max=cfg.remote_nested_max,
            nested_budget_bytes=int(cfg.remote_nested_gb * 1e9), room_for=walker._room_for, on_nested=on_nested,
            max_member_bytes=int(cfg.remote_max_member_mb * 1e6),
            record_budget_bytes=int(cfg.remote_record_budget_gb * 1e9))
        n0 = len(walker.entries)
        for path, display in res.fetched:
            walker.add_file(path, display)
        added = len(walker.entries) - n0
        entry_stratum = {id(e): ("remote_nested" if id(e) in nested_ids else "remote")
                         for e in walker.entries[first_entry:]}
        walker.kind_counts.update(res.member_kind_counts)
        walker.image_ext_counts.update(res.unfetched_image_ext_counts)
        room = max(0, 5000 - len(walker.other_names))
        walker.other_names.extend(res.other_member_names[:room])
        self._extra_paths += res.unfetched_image_paths
        # Members over --remote-max-member-mb (and empty ones) are outside the
        # sampling frame, so they are left out of the stratum population as
        # well: large members can be a different population (full-resolution
        # scans next to small masks or thumbnails), and projecting the class
        # mix of the small ones onto them would be a guess. They are reported
        # as n_images_unsampleable, unclassified and not in n_image_files.
        unsampleable = res.n_members_skipped_oversize
        def in_frame(n: int) -> int:
            return max(0, n - res.n_members_skipped_oversize - res.n_members_zero_size)

        if res.pool == "noext":
            # Extensionless members: the DICOM share of the fetched sample
            # is extrapolated to all the sampleable ones.
            seen = added
            est = in_frame(res.n_noext_listed) * added / len(res.fetched) if res.fetched else 0.0
        else:
            seen = res.n_images_listed
            est = float(in_frame(res.n_images_listed))
        nested_est = (res.nested_images * res.n_nested_listed / res.n_nested_fetched
                      if res.n_nested_fetched else 0.0)
        row.update({
            "remote_listing_detail": res.detail(),
            "remote_listing_sources": ", ".join(sorted({lst.source for lst in res.listings})),
            "remote_listings_truncated": sum(1 for lst in res.listings if lst.truncated_listing),
            "remote_pool": res.pool,
            "n_remote_images_listed": res.n_images_listed,
            "n_remote_noext_listed": res.n_noext_listed,
            "n_remote_nested_listed": res.n_nested_listed,
            "n_remote_nested_fetched": res.n_nested_fetched,
            "n_remote_nested_images": res.nested_images,
            "n_remote_sampled": len(res.sampled),
            "n_remote_fetched": len(res.fetched),
            "remote_listing_failures": res.n_listing_failures,
            "n_remote_nested_over_cap": res.n_nested_over_cap,
            "n_remote_nested_over_budget": res.n_nested_over_budget,
            "remote_nested_bytes": res.nested_bytes,
            "remote_nested_stop": res.nested_stop,
            "remote_fetch_failures": len(res.failures),
            "remote_fetch_failure_detail": "; ".join(res.failures[:5])[:1000],
            "remote_requests": res.n_requests,
            "n_members_skipped_oversize": res.n_members_skipped_oversize,
            "n_members_zero_size": res.n_members_zero_size,
            "n_images_unsampleable": unsampleable,
            "fraction_scope_note": (
                f"class fractions and est_files describe remote zip members up to --remote-max-member-mb "
                f"{cfg.remote_max_member_mb:g} only; {unsampleable} larger "
                f"{'extensionless ' if res.pool == 'noext' else 'image '}member(s) were not sampled, "
                f"are unclassified and are not counted in n_image_files" if unsampleable else ""),
            "n_remote_over_budget": res.n_members_over_budget,
            "remote_budget_stop": res.budget_stop,
            # measured bytes received (failed transfers and retries included),
            # the quantity --remote-record-budget-gb bounds
            "remote_bytes_fetched": res.bytes_received,
            "remote_bytes_fetched_ok": res.bytes_fetched,   # listed sizes of the fetches that worked
        })
        if res.n_members_over_budget or res.n_members_skipped_oversize:
            _log_event(self.events_path, {"event": "remote_byte_limits", "record": rid,
                                          "skipped_oversize": res.n_members_skipped_oversize,
                                          "over_budget": res.n_members_over_budget,
                                          "bytes": res.bytes_received})
        if res.nested_bytes:
            _log_event(self.events_path, {"event": "remote_nested", "record": rid,
                                          "fetched": res.n_nested_fetched, "listed": res.n_nested_listed,
                                          "bytes": res.nested_bytes, "images": res.nested_images,
                                          "stop": res.nested_stop})
        strata = {"remote": est}
        if nested_ids or nested_est:
            strata["remote_nested"] = nested_est
        return seen + res.nested_images, strata, entry_stratum

    # ------------------------------------------------------------ classify
    def _classify(self, row: dict, rid: str, walker: RecordWalker, cap: int | None = None,
                  population: int | None = None, strata: dict[str, float] | None = None,
                  entry_stratum: dict[int, str] | None = None):
        """Classify a seeded sample of at most ``cap`` walker entries.
        ``population`` (default: the walker's entry count) scales the
        per-class file estimates.

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

        Images the pixel check flags as masks or label maps (facts
        ``mask_like``) are labelled MASK in the thresholded and the argmax
        counts; their model prediction goes to the predictions file only.
        Dominant classes, mean confidence and probabilities, eye image
        fraction and eye class presence are taken over the non-mask images:
        the eye classes together must reach ``min_eye_fraction`` of them
        (thresholded labels), and each listed eye class
        ``min_class_fraction``; otherwise no eye class is present and the
        record gets status no_eye_images. A mask-dominated record (frac_MASK
        at least MASK_DOMINATED_FRAC and fewer than
        MASK_DOMINATED_MIN_NON_MASK non-mask images) lists no eye class
        either: the classes the rule finds go to ``review_eye_classes`` and
        the record gets status mask_dominated."""
        strata = strata or {}
        entry_stratum = entry_stratum or {}
        cfg = self.cfg
        cap = cfg.max_images if cap is None else cap
        entries = sorted(walker.entries, key=lambda e: e.display)
        population = len(entries) if population is None else population
        rng = random.Random(f"{cfg.seed}:{rid}")
        sample = entries if len(entries) <= cap else rng.sample(entries, cap)
        row["n_sampled"] = len(sample)

        preds: list[dict] = []          # per classified image
        errors = Counter()
        batch, batch_meta = [], []
        pred_fp = None
        if cfg.write_predictions:
            (cfg.out_dir / "predictions").mkdir(exist_ok=True)
            pred_fp = gzip.open(cfg.out_dir / "predictions" / f"{rid}.jsonl.gz", "wt", encoding="utf-8")

        def flush():
            if not batch:
                return
            probs = self.clf.predict(np.stack(batch))
            for (entry, facts), pr in zip(batch_meta, probs):
                top = int(pr.argmax())
                conf = float(pr[top])
                model_label = CLASSES[top] if conf >= cfg.threshold else UNCERTAIN
                # Masks and label maps get no modality: MASK in the
                # thresholded and the argmax counts alike. The model's own
                # prediction is kept in the predictions file.
                mask = bool(facts.get("mask_like"))
                label = MASK if mask else model_label
                preds.append({"cls": label, "top": MASK if mask else CLASSES[top], "conf": conf, "probs": pr,
                              "facts": facts, "path": entry.display, "mask": mask,
                              "stratum": entry_stratum.get(id(entry), "files")})
                if pred_fp:
                    rec = {"path": entry.display, "label": label, "top": CLASSES[top],
                           "p": round(conf, 4), "probs": [round(float(x), 4) for x in pr],
                           "rows": facts.get("rows"), "cols": facts.get("cols"),
                           "frames": facts.get("frames")}
                    if mask:
                        rec.update({"mask_like": True, "model_label": model_label})
                    pred_fp.write(json.dumps(rec) + "\n")
            batch.clear()
            batch_meta.clear()

        inventory_only = []   # facts of files we could not decode (still useful)
        try:
            for entry, data, path, err in walker.read_entries(sample):
                if err:
                    errors[err.split(":")[0][:60]] += 1
                    if pred_fp:
                        pred_fp.write(json.dumps({"path": entry.display, "error": err}) + "\n")
                    continue
                loaded = load_frame(entry.kind, entry.ext, data, path, cfg.max_pixels)
                del data
                if loaded.image is None:
                    errors[(loaded.error or "undecodable").split(":")[0][:60]] += 1
                    inventory_only.append(loaded.facts)
                    if pred_fp:
                        pred_fp.write(json.dumps({"path": entry.display, "error": loaded.error}) + "\n")
                    continue
                try:
                    batch.append(preprocess(loaded.image))
                except Exception as e:  # noqa: BLE001
                    errors[f"preprocess {type(e).__name__}"] += 1
                    continue
                batch_meta.append((entry, loaded.facts))
                if len(batch) >= cfg.batch_size:
                    flush()
            flush()
        finally:
            if pred_fp:
                pred_fp.close()

        n = len(preds)
        row["n_classified"] = n
        row["n_load_errors"] = sum(errors.values())
        row["load_error_top"] = "; ".join(f"{k} ({v})" for k, v in errors.most_common(4))
        row["n_inventory_only"] = len(inventory_only)
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
        # CMDS modality directories.
        row["mask_dominated"] = bool(n and frac[MASK] >= MASK_DOMINATED_FRAC
                                     and len(nm) < MASK_DOMINATED_MIN_NON_MASK)
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
        (out / "dataset_description.json").write_text(json.dumps(dd, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "dataset_structure_description.json").write_text(
            json.dumps(dsd, indent=2, ensure_ascii=False), encoding="utf-8")
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
    for f in files:
        p = local_dir / safe_filename(f["key"])
        if p.exists() and (not f["size"] or p.stat().st_size == f["size"]):
            available.append((p, f["key"]))
        elif remote_zip and not offline and remote_zip_candidate(f["key"], keys):
            remote_zips.append(f)
        elif download_all or wanted_for_download(f["key"]):
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


def run_survey(cfg: SurveyConfig) -> dict:
    return Survey(cfg).run()
