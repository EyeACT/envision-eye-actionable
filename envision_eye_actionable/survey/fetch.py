"""The fetch half of a record: metadata, plan, archive probe, downloads and
remote zip sampling, written into one directory with a manifest.

``RecordFetcher`` is used by both entry points:

* ``envision-survey run`` (one process): the record dir is the scratch dir
  ``<scratch>/records/<id>``, and the process classifies right after.
* ``envision-survey fetch`` (the producer of the pipeline): the record dir is
  ``<spool>/records/<id>``, and the manifest is published with a READY
  marker for the ``process`` role (fetch.Producer, below).

Two passes. The triage pass fetches at most ``triage_cap`` images from each
source that is sampled (remote zip members, top-level image files; archives
downloaded whole are downloaded whole once). When the consumer finds eye
classes in the triage sample (at least ``min_eye_fraction``) and the record
has more images to fetch (``deep_available``), it asks for the deep pass,
which fetches the rest of the same seeded samples, up to ``remote_cap``,
into ``<record dir>/deep``. The triage files are reused, not fetched again.

Manifest items (``items``): {path, key, role, size, local, pass}. ``path``
is relative to the record dir, except for local originals (``local``
true: files of the discovery downloads dir, read in place, never copied,
never deleted). role: local, download, remote (a remote zip member),
remote_nested (a nested archive fetched from a remote zip).
"""

from __future__ import annotations

import random
import shutil
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .archives import RecordWalker
from .constants import IMAGE_KINDS, detect_ext, file_kind
from .probe import RecordProbe, probe_record_archives
from .remote import listing_from_dict, listing_to_dict, sample_remote_zips
from .zenodo import ZenodoClient, record_files, redact, safe_filename, stream_download

MANIFEST_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


class RecordFetcher:
    """Fetch one record into ``rdir``. ``room(nbytes) -> bool`` decides
    whether ``nbytes`` more may be written now (it may block while the
    consumer frees space; False means the record cannot be fetched now:
    status skipped_disk). ``log_event(dict)`` records events."""

    def __init__(self, cfg, zenodo: ZenodoClient, room=None, log_event=None, reserved_by_others=None):
        self.cfg = cfg
        self.zenodo = zenodo
        self.room = room or self._default_room
        self.log_event = log_event or (lambda ev: None)
        self.reserved_by_others = reserved_by_others

    def _default_room(self, nbytes: int, deep: bool = False) -> bool:
        free = shutil.disk_usage(self.cfg.scratch_dir if not getattr(self.cfg, "spool_dir", None)
                                 else self.cfg.spool_dir).free
        if self.reserved_by_others is not None:
            free -= self.reserved_by_others()
        return free - nbytes >= self.cfg.disk_floor_gb * 1e9

    @property
    def triage_n(self) -> int:
        """Images fetched per sampled source on the triage pass (0 or a cap
        at or above remote_cap: one pass, everything up to remote_cap)."""
        t = int(getattr(self.cfg, "triage_cap", 0) or 0)
        return self.cfg.remote_cap if t <= 0 else min(t, self.cfg.remote_cap)

    # ------------------------------------------------------------- triage
    def fetch(self, rec: dict, rdir: Path) -> dict:
        """Triage pass. Returns the manifest (never raises: a failure is a
        terminal status in the manifest)."""
        from .runner import (fit_budget, plan_files, sample_series)
        cfg = self.cfg
        rid = rec["source_id"]
        row: dict = {}
        m = {"version": MANIFEST_VERSION, "record_id": rid, "pass": "triage", "created": _now(), "items": [],
             "terminal": None, "row": row, "unfetched_top": [], "probe_negative": [], "remote": None,
             "top_pending": [], "deep_available": False}
        rdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
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
            if legacy is None and not cfg.offline:
                # Transient (429s used up, 5xx) or the record is gone: no
                # file list to plan from. Retryable with --retry-status.
                m["terminal"] = {"status": "metadata_unavailable", "finish": "links_only",
                                 "error": "record metadata could not be fetched from Zenodo"}
                row["sampling_mode"] = "none"
                return m
            if not files:
                m["terminal"] = {"status": "restricted" if access and access != "open" else "no_files",
                                 "finish": "without_images"}
                row["sampling_mode"] = "none"
                return m

            available, remote_zips, to_fetch, skipped = plan_files(
                files, cfg.downloads_dir / rid, remote_zip=cfg.remote_zip, offline=cfg.offline,
                download_all=cfg.download_all)
            n_local_files = len(available)
            probe = self._probe_archives(row, rid, files, to_fetch, rec)
            if probe is not None:
                negative = {r.key for r in probe.by_outcome("no_images")}
                to_fetch = [f for f in to_fetch if f["key"] not in negative]
                m["probe_negative"] = [{"key": r.key, "size": r.size, "kind_counts": dict(r.kind_counts),
                                        "ext_counts": dict(r.ext_counts), "names": r.names[:5000]}
                                       for r in probe.by_outcome("no_images")]
            unfetched_top: list[str] = []
            to_fetch, series_rest = sample_series(to_fetch, cfg.series_sample_k, int(cfg.series_min_mb * 1e6),
                                                  random.Random(f"{cfg.seed}:{rid}:series"))
            unfetched_top += [f["key"] for f in series_rest]
            row["n_series_files_not_fetched"] = len(series_rest)
            # Top-level image files: one seeded order; the triage pass takes
            # its first triage_n, the deep pass the next ones up to
            # remote_cap, the rest is counted (and its names read) only.
            top_images = sorted((f for f in to_fetch if file_kind(f["key"]) in IMAGE_KINDS), key=lambda f: f["key"])
            random.Random(f"{cfg.seed}:{rid}:files").shuffle(top_images)
            deep_top = top_images[self.triage_n:cfg.remote_cap]
            beyond = top_images[cfg.remote_cap:]
            keep = {f["key"] for f in top_images[:self.triage_n]}
            to_fetch = [f for f in to_fetch if f["key"] in keep or file_kind(f["key"]) not in IMAGE_KINDS]
            unfetched_top += [f["key"] for f in deep_top + beyond]
            m["top_pending"] = [{"key": f["key"], "size": f["size"], "url": f.get("url"), "md5": f.get("md5")}
                                for f in deep_top]
            m["top_beyond"] = [f["key"] for f in beyond]
            m["series_rest"] = [f["key"] for f in series_rest]
            row["n_files_local"] = n_local_files
            row["n_files_remote_zip"] = len(remote_zips)
            row["n_files_to_download"] = len(to_fetch)
            row["n_files_not_needed"] = skipped
            row["n_toplevel_images_not_fetched"] = len(unfetched_top)
            need = sum(f["size"] for f in to_fetch)
            row["download_bytes_planned"] = need
            if to_fetch and need > cfg.max_download_gb * 1e9:
                to_fetch, over = fit_budget(to_fetch, cfg.max_download_gb * 1e9)
                over_bytes = sum(f["size"] for f in over)
                row["n_files_skipped_size"] = len(over)
                row["skipped_size_files"] = [{"key": f["key"], "size": f["size"], "url": f.get("url")}
                                             for f in over[:500]]
                self.log_event({"event": "skipped_size", "record": rid, "need": need, "n_over": len(over),
                                "over_bytes": over_bytes})
                note = (f"{len(over)} non-zip file(s), {over_bytes / 1e9:.1f} GB, did not fit "
                        f"--max-download-gb {cfg.max_download_gb:g}")
                if not available and not remote_zips and not to_fetch:
                    row["sampling_mode"] = "none"
                    row["files_list"] = [{"key": f["key"], "size": f["size"], "url": f.get("url")}
                                         for f in files[:500]]
                    m["terminal"] = {"status": "skipped_size", "finish": "metadata_links", "error": note,
                                     "note": "skipped_size: files not read, no classification"}
                    m["top_pending"] = []
                    return m
                row["size_note"] = note + "; those files were not read"
                row["n_files_to_download"] = len(to_fetch)
            need = sum(f["size"] for f in to_fetch)
            row["download_bytes"] = need
            remote_need = min(sum(int(z.get("size") or 0) for z in remote_zips),
                              int(cfg.remote_record_budget_gb * 1e9)) if remote_zips else 0
            if (to_fetch or remote_zips) and not self.room(need + remote_need):
                free = shutil.disk_usage(rdir).free
                row["sampling_mode"] = "none"
                m["terminal"] = {"status": "skipped_disk", "finish": "metadata_only",
                                 "error": (f"needs {(need + remote_need) / 1e9:.1f} GB, free {free / 1e9:.1f} GB, "
                                           f"floor {cfg.disk_floor_gb:.0f} GB"),
                                 "note": "skipped_disk: files not read, no classification"}
                m["top_pending"] = []
                self.log_event({"event": "skipped_disk", "record": rid, "need": need + remote_need, "free": free})
                return m
            items = [{"path": str(Path(p).resolve()), "key": key, "role": "local", "local": True, "size": None,
                      "pass": "triage"} for p, key in available]
            if to_fetch:
                failures, got = self._download(to_fetch, rdir / "dl", rid)
                row["download_failures"] = len(failures)
                row["download_failure_detail"] = redact("; ".join(f"{k}: {e}" for k, e in failures[:5]))
                row["bytes_downloaded"] = sum(it["size"] for it in got)
                items += got
            if remote_zips:
                m["remote"] = self._remote(rid, remote_zips, rdir, items, cap=self.triage_n, pass_="triage")
                m["_listings"] = m["remote"].pop("_listings", None)
            modes = [md for md, used in (("local", n_local_files), ("remote_zip", remote_zips),
                                         ("download", to_fetch)) if used]
            row["sampling_mode"] = "+".join(modes) or "none"
            m["items"] = items
            m["unfetched_top"] = unfetched_top
            m["deep_available"] = bool(m["top_pending"]) or bool((m["remote"] or {}).get("more_available"))
            return m
        except YieldToRequests:
            raise                                    # not a result: the producer requeues the record
        except Exception as e:  # noqa: BLE001 - one record never stops the producer
            m["terminal"] = {"status": "error", "finish": "metadata_only",
                             "error": redact(f"{type(e).__name__}: {e}")[:500],
                             "traceback": redact(traceback.format_exc())[-2000:]}
            self.log_event({"event": "fetch_error", "record": rid, "error": m["terminal"]["error"]})
            return m
        finally:
            m["fetch_s"] = round(time.time() - t0, 1)
            m["bytes"] = sum(int(it.get("size") or 0) for it in m["items"] if not it.get("local"))

    # --------------------------------------------------------------- deep
    def fetch_deep(self, rec: dict, rdir: Path, triage: dict, listings: list[dict] | None) -> dict:
        """Deep pass: the rest of the seeded samples up to remote_cap, into
        ``rdir/deep``. Returns the deep manifest: its items are the new
        files only, its ``row``, ``remote`` and ``unfetched_top`` replace the
        triage ones (totals over both passes)."""
        from .runner import fit_budget
        cfg = self.cfg
        rid = rec["source_id"]
        ddir = rdir / "deep"
        ddir.mkdir(parents=True, exist_ok=True)
        trow = triage.get("row") or {}
        row: dict = {}
        m = {"version": MANIFEST_VERSION, "record_id": rid, "pass": "deep", "created": _now(), "items": [],
             "row": row, "unfetched_top": list(triage.get("unfetched_top") or []), "remote": triage.get("remote"),
             "error": None}
        t0 = time.time()
        try:
            pending = list(triage.get("top_pending") or [])
            items: list[dict] = []
            if pending:
                budget = cfg.max_download_gb * 1e9 - int(trow.get("bytes_downloaded") or 0)
                pending, over = fit_budget(pending, max(0.0, budget))
                need = sum(f["size"] for f in pending)
                if pending and not self.room(need, deep=True):
                    over, pending = over + pending, []
                failures, got = self._download(pending, ddir / "dl", rid) if pending else ([], [])
                for it in got:
                    it["path"] = f"deep/{it['path']}"
                    it["pass"] = "deep"
                items += got
                fetched = {it["key"] for it in got}
                m["unfetched_top"] = (list(triage.get("series_rest") or []) + list(triage.get("top_beyond") or [])
                                      + [f["key"] for f in triage.get("top_pending") or [] if f["key"] not in fetched])
                row["n_files_to_download"] = int(trow.get("n_files_to_download") or 0) + len(pending)
                row["n_toplevel_images_not_fetched"] = len(m["unfetched_top"])
                row["download_failures"] = int(trow.get("download_failures") or 0) + len(failures)
                if failures:
                    row["download_failure_detail"] = redact("; ".join(
                        [trow.get("download_failure_detail") or ""] + [f"{k}: {e}" for k, e in failures[:5]]))[:1000]
                row["bytes_downloaded"] = int(trow.get("bytes_downloaded") or 0) + sum(it["size"] for it in got)
                row["n_deep_top_files_over_budget"] = len(over)
            prev = triage.get("remote")
            if prev and prev.get("more_available"):
                zips = [{"key": k, "size": s} for k, s in prev.get("zips") or []]
                m["remote"] = self._remote(rid, zips, ddir, items, cap=cfg.remote_cap, pass_="deep", prev=prev,
                                           listings=listings, rel_prefix="deep/")
            m["items"] = items
        except Exception as e:  # noqa: BLE001
            m["error"] = redact(f"{type(e).__name__}: {e}")[:500]
            self.log_event({"event": "fetch_deep_error", "record": rid, "error": m["error"]})
        m["fetch_s"] = round(time.time() - t0, 1)
        m["bytes"] = sum(int(it.get("size") or 0) for it in m["items"])
        return m

    # ------------------------------------------------------------- helpers
    def _probe_archives(self, row: dict, rid: str, files: list[dict], to_fetch: list[dict],
                        rec: dict | None = None) -> RecordProbe | None:
        """Probe the archives to download. The result is kept on ``rec``
        (``_probe_cache``): a triage job that gives its slot to a request
        (YieldToRequests, raised later in fetch) is run again from the start,
        and must not spend the request budget on the same probe again."""
        cfg = self.cfg
        row["archive_probe_enabled"] = bool(cfg.archive_probe and not cfg.offline)
        if not row["archive_probe_enabled"] or not to_fetch:
            return None
        cached = (rec or {}).get("_probe_cache")
        keys = sorted(f["key"] for f in to_fetch)
        if cached is not None and cached[0] == keys:
            _, probe, fields = cached
            if fields:
                row.update(fields)
                row["archive_probe_reused"] = True
            return probe
        probe = self._run_probe(row, rid, files, to_fetch)
        if rec is not None:
            fields = {k: v for k, v in row.items() if k.startswith(("archive_probe", "n_archive"))
                      and k != "archive_probe_enabled"}
            rec["_probe_cache"] = (keys, probe, fields)
        return probe

    def _run_probe(self, row: dict, rid: str, files: list[dict], to_fetch: list[dict]) -> RecordProbe | None:
        cfg = self.cfg
        probe = probe_record_archives(
            self.zenodo, rid, to_fetch, [f["key"] for f in files],
            stream_bytes=int(cfg.archive_probe_mb * (1 << 20)), max_requests=cfg.archive_probe_max_requests)
        if not probe.results:
            return None
        neg = probe.by_outcome("no_images")
        row.update({
            "archive_probes": [r.summary() for r in probe.results[:500]],
            "n_archives_probed": sum(1 for r in probe.results if r.outcome != "not_probed"),
            "n_archive_probe_images": len(probe.by_outcome("images")),
            "n_archive_probe_no_images": len(neg),
            "n_archive_probe_unknown": len(probe.by_outcome("unknown")),
            "n_archive_probe_not_probed": len(probe.by_outcome("not_probed")),
            "archive_probe_bytes": probe.bytes_used,
            "archive_probe_requests": probe.n_requests,
            "archive_probe_bytes_avoided": sum(r.size for r in neg),
            "archive_probe_stop": probe.stop,
            "archive_probe_detail": probe.detail(),
        })
        self.log_event({
            "event": "archive_probe", "record": rid,
            "archives": [{k: r.summary()[k] for k in ("key", "size", "outcome", "method", "n_members",
                                                       "n_images", "n_nested", "n_noext", "bytes", "requests",
                                                       "note")}
                         for r in probe.results[:200]],
            "bytes": probe.bytes_used, "requests": probe.n_requests})
        return probe

    def _download(self, files: list[dict], dl_dir: Path, rid: str) -> tuple[list[tuple[str, str]], list[dict]]:
        """Stream ``files`` into ``dl_dir``. Returns (failures, items)."""
        dl_dir.mkdir(parents=True, exist_ok=True)
        session = self.zenodo.session
        floor = self.cfg.disk_floor_gb * 1e9
        failures, items = [], []

        def room() -> bool:
            return shutil.disk_usage(dl_dir).free >= floor

        dests = {}
        taken: set[str] = {p.name for p in dl_dir.iterdir()}
        for i, f in enumerate(files):
            # two keys with one base name (a/x.tif, b/x.tif) get their own files
            name = safe_filename(f["key"])
            if name in taken:
                name = f"{i:04d}_{name}"
            taken.add(name)
            dests[id(f)] = dl_dir / name

        def one(f):
            dest = dests[id(f)]
            if not f.get("url"):
                return f, dest, "no download URL"
            ok, err = stream_download(session, f["url"], dest, f["size"] or None, f.get("md5"),
                                      throttle=self.zenodo, room_check=room)
            return f, dest, (None if ok else err)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.download_workers)) as ex:
            for f, dest, err in ex.map(one, files):
                if err:
                    failures.append((f["key"], redact(err)))
                    self.log_event({"event": "download_failed", "record": rid, "file": f["key"],
                                    "error": redact(err)})
                elif dest.exists():
                    items.append({"path": dest.relative_to(dl_dir.parent).as_posix(), "key": f["key"],
                                  "role": "download", "local": False, "size": _file_size(dest), "pass": "triage"})
        return failures, items

    def _remote(self, rid: str, zips: list[dict], rdir: Path, items: list[dict], cap: int, pass_: str,
                prev: dict | None = None, listings: list[dict] | None = None, rel_prefix: str = "") -> dict:
        """Remote zip sampling into ``rdir/remote``; appends the fetched
        members and nested archives to ``items`` and returns the remote
        summary (totals over the passes when ``prev`` is the triage one)."""
        cfg = self.cfg
        walk_dir = rdir / ".fetchwalk"
        walk_dir.mkdir(parents=True, exist_ok=True)
        walker = RecordWalker(scratch=walk_dir, max_depth=cfg.max_depth,
                              disk_floor_bytes=int(cfg.disk_floor_gb * 1e9),
                              reserved_by_others=self.reserved_by_others)
        nested_files: list[tuple[Path, str]] = []

        def on_nested(path: Path, display: str) -> int:
            n0 = len(walker.entries)
            walker.add_file(path, display)
            walker._drop_if_unused(path)
            if path.exists():
                nested_files.append((path, display))
            return len(walker.entries) - n0

        prev = prev or {}
        before = {k: prev.get(k) for k in ("nested_images", "n_nested_fetched", "nested_bytes", "bytes_received",
                                           "nested_done")} if prev else None
        pre = [listing_from_dict(d) for d in listings] if listings else None
        try:
            res = sample_remote_zips(
                self.zenodo, rid, zips, rdir / "remote", cap, random.Random(f"{cfg.seed}:{rid}:remote"),
                workers=cfg.download_workers, nested_max=cfg.remote_nested_max,
                nested_budget_bytes=int(cfg.remote_nested_gb * 1e9), room_for=walker._room_for,
                on_nested=on_nested, max_member_bytes=int(cfg.remote_max_member_mb * 1e6),
                record_budget_bytes=int(cfg.remote_record_budget_gb * 1e9), listings=pre,
                floor_cap=self.triage_n if prev else None,
                skip_members={tuple(x) for x in prev.get("fetched_keys") or []} if prev else None,
                before=before)
        finally:
            shutil.rmtree(walk_dir, ignore_errors=True)
        base = rdir.parent if rel_prefix else rdir
        for path, display in res.fetched:
            items.append({"path": path.relative_to(base).as_posix(), "key": display, "role": "remote",
                          "local": False, "size": _file_size(path), "pass": pass_})
            # a volume pair's data member sits next to its header; it is
            # validated as its own file
            if path.parent.parent.name == "pairs":
                for sib in path.parent.iterdir():
                    if sib != path:
                        items.append({"path": sib.relative_to(base).as_posix(), "key": display + "#companion",
                                      "role": "companion", "local": False, "size": _file_size(sib),
                                      "pass": pass_})
        for path, display in nested_files:
            items.append({"path": path.relative_to(base).as_posix(), "key": display, "role": "remote_nested",
                          "local": False, "size": _file_size(path), "pass": pass_})
        fetched_keys = [[k, n] for k, n in prev.get("fetched_keys") or []] if prev else []
        fetched_keys += [disp.split("!/", 1) for _, disp in res.fetched]
        s = {
            "zips": [[z["key"], int(z.get("size") or 0)] for z in zips],
            "pool": res.pool,
            "n_images_listed": res.n_images_listed,
            "n_noext_listed": res.n_noext_listed,
            "n_nested_listed": res.n_nested_listed,
            "n_nested_fetched": int(prev.get("n_nested_fetched") or 0) + res.n_nested_fetched,
            "nested_images": int(prev.get("nested_images") or 0) + res.nested_images,
            "nested_bytes": int(prev.get("nested_bytes") or 0) + res.nested_bytes,
            "n_nested_over_cap": max(int(prev.get("n_nested_over_cap") or 0), res.n_nested_over_cap),
            "n_nested_over_budget": max(int(prev.get("n_nested_over_budget") or 0), res.n_nested_over_budget),
            "nested_stop": res.nested_stop or prev.get("nested_stop") or "",
            "nested_done": sorted([list(x) for x in res.nested_done]),
            "n_members_skipped_oversize": res.n_members_skipped_oversize,
            "n_members_zero_size": res.n_members_zero_size,
            "n_pool_in_frame": res.n_pool_in_frame,
            "n_sampled": len(res.sampled),
            "n_fetched": int(prev.get("n_fetched") or 0) + len(res.fetched),
            "n_reused": res.n_reused,
            "failures": [redact(x) for x in (list(prev.get("failures") or []) + res.failures)][:50],
            "n_failures": int(prev.get("n_failures") or 0) + len(res.failures),
            "n_listing_failures": res.n_listing_failures,
            "listing_detail": res.detail() if not listings else prev.get("listing_detail", ""),
            "listing_sources": ", ".join(sorted({lst.source for lst in res.listings})),
            "listings_truncated": sum(1 for lst in res.listings if lst.truncated_listing),
            "n_requests": int(prev.get("n_requests") or 0) + res.n_requests,
            "n_members_over_budget": res.n_members_over_budget,
            "budget_stop": res.budget_stop,
            "bytes_received": int(prev.get("bytes_received") or 0) + res.bytes_received,
            "members_bytes": int(prev.get("members_bytes") or 0) + res.members_bytes,
            "unfetched_image_paths": res.unfetched_image_paths,
            "other_member_names": res.other_member_names,
            "member_kind_counts": res.member_kind_counts,
            "member_ext_counts": res.member_ext_counts,
            "unfetched_image_ext_counts": res.unfetched_image_ext_counts,
            "fetched_keys": fetched_keys,
            "more_available": res.more_available,
            "pass": pass_,
        }
        if not prev:
            s["_listings"] = [listing_to_dict(lst) for lst in res.listings]
        if res.n_members_over_budget or res.n_members_skipped_oversize:
            self.log_event({"event": "remote_byte_limits", "record": rid, "pass": pass_,
                            "skipped_oversize": res.n_members_skipped_oversize,
                            "over_budget": res.n_members_over_budget, "bytes": res.bytes_received})
        if res.nested_bytes:
            self.log_event({"event": "remote_nested", "record": rid, "pass": pass_,
                            "fetched": res.n_nested_fetched, "listed": res.n_nested_listed,
                            "bytes": res.nested_bytes, "images": res.nested_images, "stop": res.nested_stop})
        return s

class YieldToRequests(Exception):
    """Raised by RoomGate.acquire in a triage job that waits for room while
    a deep or refetch request waits for a fetch slot: the job gives up its
    slot (the producer puts the record back at the front of its queue)."""


class RoomGate:
    """The producer's disk gate: a record may start writing ``nbytes`` when
    the spool (plus what records in flight reserved) stays under
    ``spool_max_bytes`` and the free space stays above ``floor_bytes``.
    Waits while either is exceeded; a record larger than the spool budget
    goes ahead once nothing else is in the spool or in flight. Returns False
    (the record gets skipped_disk) only when even an empty spool leaves no
    room above the floor, or when ``should_stop()`` says the consumer is
    gone (nothing will free space). Raises YieldToRequests from a job that
    is not deep while it would wait and ``should_yield()`` is true.

    The check and the reservation are one step under ``_gate``: two jobs
    never both pass against the same free space."""

    def __init__(self, spool, spool_max_bytes: int, floor_bytes: int, poll_s: float = 15.0,
                 should_stop=None, on_wait=None, others=None, should_yield=None):
        self.spool, self.spool_max, self.floor = spool, int(spool_max_bytes), int(floor_bytes)
        self.poll_s = poll_s
        self.should_stop = should_stop or (lambda: False)
        self.on_wait = on_wait or (lambda state: None)
        self.others = others or (lambda: 0)
        self.should_yield = should_yield or (lambda: False)
        self._lock = threading.Lock()
        self._gate = threading.Lock()
        self.inflight: dict[int, int] = {}
        self.waiting = 0
        self.stopped = False

    def _free(self) -> int:
        return shutil.disk_usage(self.spool.root).free - self.others()

    def reserved(self) -> int:
        with self._lock:
            return sum(self.inflight.values())

    def acquire(self, nbytes: int, deep: bool = False) -> bool:
        """``deep``: a deep pass of a record already in the spool; it is not
        held back by the spool budget (records awaiting their deep pass
        could otherwise fill the spool and wait on each other), only by the
        disk floor."""
        me = threading.get_ident()
        nbytes = max(0, int(nbytes))
        waited = False

        def done_waiting():
            if waited:
                with self._lock:
                    self.waiting -= 1

        while True:
            with self._gate:
                with self._lock:
                    others = sum(v for k, v in self.inflight.items() if k != me)
                used = self.spool.bytes_used()
                free = self._free() - others
                alone = used == 0 and others == 0
                if free - nbytes < self.floor and alone:
                    done_waiting()
                    return False                         # no room even with an empty spool
                if free - nbytes >= self.floor and (deep or used + others + nbytes <= self.spool_max or alone):
                    with self._lock:
                        self.inflight[me] = nbytes
                    done_waiting()
                    return True
            if self.should_stop():
                self.stopped = True
                done_waiting()
                return False
            if not deep and self.should_yield():
                done_waiting()
                raise YieldToRequests()
            if not waited:
                waited = True
                with self._lock:
                    self.waiting += 1
            self.on_wait({"spool_bytes": used, "reserved": others, "free": free, "need": nbytes})
            time.sleep(self.poll_s)

    def release(self):
        with self._lock:
            self.inflight.pop(threading.get_ident(), None)
