"""Producer / consumer pipeline for the full Zenodo pull.

Three roles, three processes, one shared state dir and one spool dir:

* ``fetch`` (Producer): plans every record of the scrape, reads what can
  be read remotely (zip members through the container API, top-level
  image files one by one, archive listings by range reads), downloads
  whole only the archives that cannot be read remotely and whose probe
  allows it, and publishes each record into the spool (spool.py). It holds
  the only Zenodo token and makes every Zenodo API request of the job,
  through one throttle (``--rpm``; ``--shared-rpm`` across processes, 429
  Retry-After shared through the state dir). Disk high-water mark: a record
  starts only while the spool holds less than ``--spool-max-gb`` and the
  drive keeps ``--disk-floor-gb`` free (fetch.RoomGate); otherwise the
  producer waits for the consumer. When the consumer is gone for
  ``--consumer-grace-s`` and the spool is full, the producer stops (exit
  code 3) instead of waiting for ever.
* ``process`` (Processor): takes ready records from the spool, classifies
  (format conversion, MASK, UNCERTAIN, triage then deep pass), writes the
  result row, CMDS JSON, predictions and weblink catalogue, and deletes the
  record's spool dir only after its row is durably written (fsync). It
  never deletes anything outside ``<spool>/records`` and never touches the
  discovery downloads (local originals are read in place). A record whose
  triage sample holds eye classes and has more images to fetch goes back
  to the producer as a deep request; the triage files stay in the spool
  and are reused. With ``--keep-dir`` the fetched files of a record with at
  least one eye image are moved into the keep dir instead of deleted
  (keep.py; ``--keep-max-gb`` and the disk floor bound it).
* ``monitor``: every ``--interval`` seconds, disk, spool size and states,
  request queue, results by status, rates and ETA into
  ``<state>/status.json`` (and one line in ``monitor.jsonl``).

``pipeline`` starts the three as child processes (fetch first). A fetch
or process that exits non-zero (a crash, an OOM kill) is logged in
``pipeline_events.jsonl`` and ``pipeline_status.json`` (and so shows in
the monitor status) and started again after a backoff, at most
``--max-role-restarts`` times per role. The pipeline ends when fetch and
process have both ended and no restart is pending.

Resuming: everything is resumable after a kill. A record is done only when
its result row is written; at start the producer leaves out records with a
final row and records a previous run left ready, awaiting_deep or
deep_ready in the spool (the consumer and the request files own them),
skips a queued record that got a final row since it started, and deletes
and refetches a record dir it left without READY. A record left
awaiting_deep without a deep request gets its request again. The consumer
re-validates every spool item (existence and size) before it processes a
record; a record that fails validation is deleted from the spool and
requested again from the producer. A record the consumer started
``--max-attempts`` times without finishing (a crash inside it) gets status
crashed.

Fetch slots: a triage job of the queue that waits for spool room gives
its slot to a pending deep or refetch request (the record goes back to
the front of the queue), so records awaiting their deep pass can always
be served and freed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from .constants import IMAGE_KINDS, file_kind, wanted_for_download
from .locks import LOCKING_AVAILABLE, DiskReservations, try_lock, unlock
from .results import is_final, load_statuses
from .spool import Spool, valid_record_id
from .zenodo import ZenodoClient, load_scrape, load_token, record_files, redact

PRODUCER_GONE_CHECKS = 3        # consumer: producer absent this many polls in a row -> drain and stop
STATUS_EVERY_S = 15.0
EXIT_CONSUMER_GONE = 3
MAX_REFETCHES = 2               # consumer: a record whose spool fails validation this often gets an error row


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, doc: dict):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(redact(json.dumps(doc, indent=1, default=str)), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, doc: dict):
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(redact(json.dumps({"ts": _now(), **doc}, default=str)) + "\n")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# role locks (liveness) and status files
# ---------------------------------------------------------------------------
SPOOL_OWNER = ".state_dir"       # in the spool root: the state dir that owns the spool


class RoleLock:
    """One process per role and state dir; the lock also tells the other
    roles this one is alive (released by the OS when the process dies).

    With ``spool_root`` the role also locks ``<spool>/.lock_<role>``, and
    the spool records its owning state dir (``<spool>/.state_dir``): the
    producer and the consumer delete record dirs they assume nobody else
    writes, so two runs given one spool with different state dirs must not
    both start."""

    def __init__(self, state_dir: Path, role: str, spool_root: Path | None = None):
        d = Path(state_dir) / "locks"
        d.mkdir(parents=True, exist_ok=True)
        self.fp = open(d / f"{role}.lock", "a+")
        self.spool_fp = None
        if LOCKING_AVAILABLE and not try_lock(self.fp):
            self.fp.close()
            raise ValueError(f"another '{role}' process is running with state dir {state_dir}")
        if spool_root is None:
            return
        try:
            root = Path(spool_root)
            owner = root / SPOOL_OWNER
            me = str(Path(state_dir).resolve())
            try:
                have = owner.read_text(encoding="utf-8").strip()
            except OSError:
                have = ""
            if have and have != me:
                raise ValueError(f"spool dir {root} belongs to state dir {have}, not {me}; pass that --state-dir "
                                 f"(or a new --spool-dir). Delete {owner} only when no survey uses the spool")
            self.spool_fp = open(root / f".lock_{role}", "a+")
            if LOCKING_AVAILABLE and not try_lock(self.spool_fp):
                raise ValueError(f"another '{role}' process is running with spool dir {root}")
            if not have:
                owner.write_text(me + "\n", encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def close(self):
        for name in ("spool_fp", "fp"):
            fp = getattr(self, name, None)
            if fp is not None:
                unlock(fp)
                fp.close()
                setattr(self, name, None)


def role_alive(state_dir: Path, role: str) -> bool:
    """True while a process holds the role's lock. Without file locking the
    role's status heartbeat (younger than 3 status periods) decides."""
    path = Path(state_dir) / "locks" / f"{role}.lock"
    if not LOCKING_AVAILABLE:
        st = _read_json(Path(state_dir) / f"{role}_status.json") or {}
        return st.get("state") not in ("finished", "stopped", "failed", None) and \
            time.time() - float(st.get("heartbeat_epoch") or 0) < 3 * STATUS_EVERY_S
    if not path.exists():
        return False
    with open(path, "a+") as fp:
        if try_lock(fp):
            unlock(fp)
            return False
        return True


class RoleStatus:
    """<state>/<role>_status.json, rewritten by a heartbeat thread."""

    def __init__(self, state_dir: Path, role: str):
        self.path = Path(state_dir) / f"{role}_status.json"
        self.doc = {"role": role, "pid": os.getpid(), "started_at": _now(), "state": "starting", "counters": {},
                    "current": []}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)

    def start(self):
        self.write()
        self._thread.start()

    def update(self, **kw):
        with self._lock:
            self.doc.update(kw)

    def count(self, key: str, n: int = 1):
        with self._lock:
            c = self.doc["counters"]
            c[key] = c.get(key, 0) + n

    def write(self):
        with self._lock:
            self.doc["heartbeat"] = _now()
            self.doc["heartbeat_epoch"] = time.time()
            doc = json.loads(json.dumps(self.doc, default=str))
        try:
            _write_json(self.path, doc)
        except OSError:
            pass

    def _beat(self):
        while not self._stop.wait(STATUS_EVERY_S):
            self.write()

    def close(self, state: str):
        self.update(state=state, ended_at=_now())
        self._stop.set()
        self.write()


# ---------------------------------------------------------------------------
# order of work
# ---------------------------------------------------------------------------
def fetch_cost(rec: dict, legacy: dict | None, downloads_dir: Path, cfg) -> float:
    """Bytes the producer expects to move for a record: files downloaded
    whole (non-zip archives, top-level images; at most --max-download-gb)
    plus a nominal 50 MB per remote zip. From the record JSON when cached,
    else from the scrape (file_names and size_mb). Cheap records go first,
    so the catalogue fills early and the big archives come last."""
    from .runner import plan_files
    zip_cost = 50e6
    cap = cfg.max_download_gb * 1e9
    if legacy is not None:
        files = record_files(legacy)
        if not files:
            return 0.0
        _avail, zips, to_fetch, _ = plan_files(files, downloads_dir / rec["source_id"], remote_zip=cfg.remote_zip,
                                               download_all=cfg.download_all)
        return min(cap, float(sum(f["size"] for f in to_fetch))) + zip_cost * len(zips)
    names = [str(n) for n in rec.get("file_names") or []]
    size = float(rec.get("size_mb") or 0) * 1e6
    if not names:
        return size
    zips = sum(1 for n in names if n.lower().endswith(".zip"))
    whole = [n for n in names if not n.lower().endswith(".zip")
             and (wanted_for_download(n) or file_kind(n) in IMAGE_KINDS)]
    if whole:
        return min(cap, size)
    return zip_cost * zips


def order_records(records: list[dict], cfg, zenodo: ZenodoClient) -> list[dict]:
    if cfg.order != "cost":
        return list(records)
    costs = []
    for i, rec in enumerate(records):
        legacy = _cached_legacy(zenodo, rec["source_id"])
        costs.append((fetch_cost(rec, legacy, cfg.downloads_dir, cfg), i, rec))
    costs.sort(key=lambda t: (t[0], t[1]))
    return [rec for _, _, rec in costs]


def _cached_legacy(zenodo: ZenodoClient, rid: str) -> dict | None:
    """Record JSON from the metadata dir or the cache, never the network."""
    for p in ([zenodo.metadata_dir / f"{rid}.json"] if zenodo.metadata_dir else []) + \
            [zenodo.cache_dir / "legacy" / f"{rid}.json"]:
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return None


def records_in_scope(cfg) -> list[dict]:
    records = load_scrape(cfg.scrape_path)
    if cfg.ids:
        want = {str(i) for i in cfg.ids}
        records = [r for r in records if r["source_id"] in want]
    return records


# ---------------------------------------------------------------------------
# metadata prefetch (pass 0)
# ---------------------------------------------------------------------------
def prefetch_metadata(cfg, threads: int = 2, progress_every: int = 500) -> dict:
    """Fetch and cache the record JSON (legacy) and the DataCite JSON of
    every record in scope that is neither in --metadata-dir nor in
    --metadata-cache-dir. Resumable (cached records cost nothing), rate
    limited like every other Zenodo request."""
    cache = cfg.metadata_cache_dir or (cfg.out_dir / "cache")
    zen = ZenodoClient(cache, cfg.metadata_dir, min_interval=cfg.zenodo_interval, offline=cfg.offline,
                       max_per_minute=cfg.zenodo_per_minute, shared_state_dir=cfg.state_dir or cfg.shared_state_dir,
                       shared_per_minute=cfg.zenodo_shared_per_minute, token=load_token(cfg.token_file))
    records = records_in_scope(cfg)
    stats = Counter()
    mismatch = []
    lock = threading.Lock()

    def one(rec):
        rid = rec["source_id"]
        have_l = (cfg.metadata_dir and (cfg.metadata_dir / f"{rid}.json").is_file()) or \
            (cache / "legacy" / f"{rid}.json").is_file()
        have_d = (cache / "datacite" / f"{rid}.json").is_file()
        if have_l and have_d:
            return "cached", rid, None
        legacy = zen.legacy(rid)
        datacite = zen.datacite(rid)
        n = len(record_files(legacy)) if legacy else None
        return ("fetched" if legacy is not None and datacite is not None else "failed"), rid, n

    t0 = time.time()
    want_files = {r["source_id"]: r.get("file_count") for r in records}
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for i, (st, rid, n) in enumerate(ex.map(one, records), 1):
            with lock:
                stats[st] += 1
            want = want_files.get(rid)
            if n is not None and isinstance(want, int) and n < want:
                mismatch.append(rid)          # the record JSON lists fewer files than the scrape counted
            if i % progress_every == 0:
                rate = i / max(1e-6, time.time() - t0)
                print(f"[metadata] {i}/{len(records)} {dict(stats)} ({rate * 3600:.0f}/h)", flush=True)
    summary = {"records": len(records), **stats, "cache_dir": str(cache), "requests": zen.n_requests,
               "file_count_below_scrape": mismatch[:100]}
    print(f"[metadata] done {json.dumps(summary)}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# producer
# ---------------------------------------------------------------------------
class Producer:
    def __init__(self, cfg):
        from .fetch import RecordFetcher, RoomGate
        self.cfg = cfg
        if cfg.spool_dir is None or cfg.state_dir is None:
            raise ValueError("fetch needs --spool-dir and --state-dir")
        cfg.downloads_dir = Path(cfg.downloads_dir).resolve()
        self.state = Path(cfg.state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.spool = Spool(cfg.spool_dir, cfg.downloads_dir, cfg.out_dir)
        self.lock = RoleLock(self.state, "fetch", self.spool.root)
        self.status = RoleStatus(self.state, "fetch")
        self.events = self.state / "fetch_events.jsonl"
        self.zenodo = ZenodoClient(cfg.metadata_cache_dir or (cfg.out_dir / "cache"), cfg.metadata_dir,
                                   min_interval=cfg.zenodo_interval, offline=cfg.offline,
                                   max_per_minute=cfg.zenodo_per_minute, shared_state_dir=self.state,
                                   shared_per_minute=cfg.zenodo_shared_per_minute, token=load_token(cfg.token_file))
        self.disk = DiskReservations(self.state, self.spool.root)
        self._consumer_seen = time.monotonic()
        self._tl = threading.local()                  # .yieldable: this thread runs a triage job of the queue
        self.gate = RoomGate(self.spool, int(cfg.spool_max_gb * 1e9), int(cfg.disk_floor_gb * 1e9),
                             poll_s=cfg.poll_s, should_stop=self._consumer_gone, on_wait=self._on_wait,
                             others=self.disk.others, should_yield=self._should_yield)
        self.fetcher = RecordFetcher(cfg, self.zenodo, room=self._acquire, log_event=self._event,
                                     reserved_by_others=self.disk.others)
        self.inflight: dict[str, str] = {}            # rid -> kind
        self._lock = threading.Lock()
        self._yielded: deque = deque()                # triage records that gave their slot to a request
        self.results_tail: ResultsTail | None = None  # final rows written since this producer started
        self._tail_lock = threading.Lock()
        self._last_repair = 0.0
        self.stop_reason = ""

    # -- hooks
    def _event(self, ev: dict):
        _append_jsonl(self.events, ev)

    def _consumer_gone(self) -> bool:
        if role_alive(self.state, "process"):
            self._consumer_seen = time.monotonic()
            return False
        return time.monotonic() - self._consumer_seen > self.cfg.consumer_grace_s

    def _on_wait(self, info: dict):
        self.status.update(state="waiting_for_room", waiting=info)

    def _should_yield(self) -> bool:
        """A triage job of the queue waiting for room gives its fetch slot
        to a pending deep or refetch request nobody serves yet: requests
        free or finish spooled records, a waiting triage job does not
        (without this every slot can wait for room that only the requests
        would make)."""
        if not getattr(self._tl, "yieldable", False):
            return False
        with self._lock:
            busy = set(self.inflight)
        return any(str(r.get("record_id")) not in busy for r in self.spool.pending_requests())

    def _finished_this_run(self, rid: str) -> bool:
        """A final row for ``rid`` was written after this producer started
        (a record the consumer finished from the spool it found there)."""
        if self.results_tail is None:
            return False
        with self._tail_lock:
            self.results_tail.poll()
            st = self.results_tail.status.get(rid)
        return st is not None and is_final({"status": st})

    def _repair_requests(self):
        """A record left awaiting_deep with no request (the consumer died
        between its marker and its request, or the producer between clearing
        the request and publishing the deep pass) gets its deep request
        again; nothing else would ever move it. Likewise a record the
        consumer sent back (RETURNED, its markers removed) whose refetch
        request a kill prevented gets that request.
        Called from the main loop only, the thread that starts jobs, so a
        record not in flight here cannot start meanwhile."""
        with self._lock:
            busy = set(self.inflight)
        have = {str(r.get("record_id")) for r in self.spool.pending_requests()}
        for rid, st in self.spool.states().items():
            if rid in busy or rid in have:
                continue
            if st == "awaiting_deep":
                self.spool.request(rid, "deep")
                self._event({"event": "deep_request_restored", "record": rid})
            elif st == "fetching" and self.spool.returned(rid):
                try:
                    n = int(self.spool.read_manifest(rid).get("refetches") or 0) + 1
                except (OSError, ValueError, AttributeError):
                    n = 1
                self.spool.request(rid, "refetch", {"refetches": n})
                self._event({"event": "refetch_request_restored", "record": rid, "refetches": n})

    def _remove_partial(self, why: str):
        """Delete every record dir left 'fetching' by a killed or failed
        producer job (not the ones the consumer sent back, RETURNED: those
        get a refetch request). Only one producer runs per spool (the role
        lock), so no other job owns them."""
        with self._lock:
            busy = set(self.inflight)
        for rid, st in self.spool.states().items():
            if st != "fetching" or rid in busy:
                continue
            if self.spool.returned(rid):
                continue
            try:
                self.spool.remove(rid)
            except (OSError, ValueError) as e:
                self._event({"event": "partial_spool_remove_failed", "record": rid,
                             "error": f"{type(e).__name__}: {e}"})
                continue
            self._event({"event": "partial_spool_removed", "record": rid, "reason": why})

    def _acquire(self, nbytes: int, deep: bool = False) -> bool:
        ok = self.gate.acquire(nbytes, deep=deep)
        self.disk.reserve(self.gate.reserved)        # evaluated under the reservation's write lock
        if ok:
            self.status.update(state="running", waiting=None)
        return ok

    def _release(self):
        self.gate.release()
        self.disk.reserve(self.gate.reserved)

    # -- jobs
    def _triage_queued(self, rec: dict) -> str:
        """A triage job taken from the queue (not a refetch request): it
        skips a record finished since start, and it may give its slot to a
        request while it waits for room (the record goes back to the front
        of the queue)."""
        from .fetch import YieldToRequests
        rid = rec["source_id"]
        if self._finished_this_run(rid):
            rec.pop("_probe_cache", None)
            return "skipped_finished"
        self._tl.yieldable = True
        try:
            out = self._triage(rec)
            rec.pop("_probe_cache", None)
            return out
        except YieldToRequests:
            # the archive probe result stays on rec (_probe_cache): the retry
            # does not spend the request budget on the same probe again
            self.spool.remove(rid)
            self._yielded.append(rec)
            self._event({"event": "triage_yielded", "record": rid})
            return "yielded_to_request"
        except BaseException:
            rec.pop("_probe_cache", None)
            raise
        finally:
            self._tl.yieldable = False

    def _triage(self, rec: dict, before_publish=None) -> str:
        rid = rec["source_id"]
        st = self.spool.state(rid)
        if st in ("ready", "awaiting_deep", "deep_ready"):
            return "skipped_in_spool"
        if st == "keep_failed":
            return "skipped_keep_failed"         # finished; its files wait for the keep dir
        if st == "fetching":
            # left by a killed producer (only one producer runs per state dir)
            self.spool.remove(rid)
            self._event({"event": "partial_spool_removed", "record": rid})
        rdir = self.spool.record_dir(rid)
        try:
            m = self.fetcher.fetch(rec, rdir)
        finally:
            self._release()
        status = (m.get("terminal") or {}).get("status") or "ready"
        if status == "skipped_disk" and self.gate.stopped:
            # not a result: nothing frees space while the consumer is gone
            self.spool.remove(rid)
            self.stop_reason = "consumer gone and spool full"
            return "stopped"
        listings = m.pop("_listings", None)
        if listings:
            self.spool.write_listings(rid, listings)
        m["refetches"] = int(rec.get("_refetches") or 0)
        if before_publish is not None:
            before_publish()                     # e.g. clear the served request while nobody else sees the record
        self.spool.write_manifest(rid, m)
        self._event({"event": "fetched", "record": rid, "status": status, "bytes": m.get("bytes"),
                     "items": len(m.get("items") or []), "s": m.get("fetch_s"),
                     "deep_available": m.get("deep_available")})
        self.status.count("records_fetched")
        self.status.count("bytes_fetched", int(m.get("bytes") or 0))
        return status

    def _deep(self, rec: dict) -> str:
        rid = rec["source_id"]
        st = self.spool.state(rid)
        if st != "awaiting_deep":
            self.spool.clear_request(rid)
            return f"deep_request_dropped_{st}"
        self.spool.remove_deep(rid)
        m = self.spool.read_manifest(rid)
        listings = self.spool.read_listings(rid)
        try:
            dm = self.fetcher.fetch_deep(rec, self.spool.record_dir(rid), m, listings)
        finally:
            self._release()
        # the request goes before the DEEP_READY marker: once the marker is
        # there the consumer may process the record and ask for it again
        # (a refetch), and that new request must survive. A producer killed
        # in between leaves awaiting_deep without a request, which
        # _repair_requests restores.
        self.spool.clear_request(rid)
        self.spool.write_manifest(rid, dm, deep=True)
        self._event({"event": "fetched_deep", "record": rid, "bytes": dm.get("bytes"),
                     "items": len(dm.get("items") or []), "s": dm.get("fetch_s"), "error": dm.get("error")})
        self.status.count("deep_fetched")
        self.status.count("bytes_fetched", int(dm.get("bytes") or 0))
        return "deep"

    def _refetch(self, rec: dict) -> str:
        rid = rec["source_id"]
        st = self.spool.state(rid)
        if st not in ("absent", "fetching"):
            self.spool.clear_request(rid)
            return "refetch_dropped"
        # cleared before READY is published (see _deep); a producer killed
        # before that leaves the record without a final row, so its next
        # start triages it from the queue
        out = self._triage(rec, before_publish=lambda: self.spool.clear_request(rid))
        if out == "skipped_in_spool":
            self.spool.clear_request(rid)
        return out

    def run(self) -> int:
        self.status.start()
        code = 0
        try:
            code = self._run()
        except Exception as e:  # noqa: BLE001 - logged, then the process fails visibly
            self._event({"event": "fetch_crashed", "error": redact(f"{type(e).__name__}: {e}"),
                         "traceback": redact(traceback.format_exc())[-3000:]})
            self.status.close("failed")
            raise
        finally:
            self.disk.close()
        self.status.close("stopped" if code else "finished")
        self.lock.close()
        return code

    def _run(self) -> int:
        cfg = self.cfg
        scope = records_in_scope(cfg)
        by_id = {r["source_id"]: r for r in scope}
        results_path = cfg.out_dir / "survey_results.jsonl"
        self.results_tail = ResultsTail(results_path)
        try:
            self.results_tail.offset = results_path.stat().st_size   # only rows written from now on
        except OSError:
            pass
        done = load_statuses(results_path)
        # --refetch-keep-ids: records finished before retention existed (their
        # last final row has no 'kept' field) are fetched and classified
        # again so that their files reach the keep dir; the new row replaces
        # the old one (last line wins). Rows written with retention are not
        # redone, so the option is safe to leave on across restarts.
        refetch = {str(i) for i in (getattr(cfg, "refetch_ids", None) or [])}
        refetch = {rid for rid in refetch if is_final(done.get(rid)) and "kept" not in (done.get(rid) or {})}
        todo = [r for r in scope if not is_final(done.get(r["source_id"]), cfg.retry_statuses)
                or r["source_id"] in refetch]
        # records a previous producer left ready, awaiting_deep or deep_ready
        # belong to the consumer and the request files, not to the queue
        # (fetching dirs stay in the queue: they are removed and refetched)
        owned = {rid for rid, st in self.spool.states().items()
                 if st in ("ready", "awaiting_deep", "deep_ready", "keep_failed")}
        todo = [r for r in todo if r["source_id"] not in owned]
        # dirs a killed producer left half written, whatever this run's
        # --ids / --limit / --retry-status: they take room in the spool
        self._remove_partial("left by an earlier producer")
        self._repair_requests()
        todo = order_records(todo, cfg, self.zenodo)
        if refetch:
            # the backfill goes first (cheapest first among it)
            todo = [r for r in todo if r["source_id"] in refetch] + [r for r in todo if r["source_id"] not in refetch]
        if cfg.limit is not None:
            todo = todo[: cfg.limit]
        self.status.update(state="running", n_scope=len(scope), n_todo=len(todo), order=cfg.order,
                           n_refetch_keep=len(refetch))
        self._event({"event": "fetch_start", "n_scope": len(scope), "n_todo": len(todo),
                     "n_refetch_keep": len(refetch),
                     "config": {k: (v if k != "ids" else f"{len(v)} ids") for k, v in vars(cfg).items()
                                if k not in ("token", "refetch_ids")}})
        print(f"[fetch] {len(scope)} records in scope, {len(todo)} to fetch", flush=True)
        queue = deque(todo)
        futures = {}
        idle_ends = 0
        with ThreadPoolExecutor(max_workers=max(1, cfg.fetch_workers)) as ex:
            while True:
                while self._yielded:                  # back to the front, in their order
                    queue.appendleft(self._yielded.pop())
                if time.monotonic() - self._last_repair > max(30.0, 20 * cfg.poll_s):
                    self._last_repair = time.monotonic()
                    self._repair_requests()
                # requests (deep, refetch) go before new records
                while len(futures) < cfg.fetch_workers and not self.stop_reason:
                    job = self._next_job(queue, by_id)
                    if job is None:
                        break
                    kind, rec = job
                    with self._lock:
                        self.inflight[rec["source_id"]] = kind
                    fn = {"triage": self._triage_queued, "deep": self._deep, "refetch": self._refetch}[kind]
                    futures[ex.submit(fn, rec)] = (kind, rec["source_id"])
                self.status.update(current=[f"{k}:{r}" for r, k in self.inflight.items()], queue=len(queue),
                                   requests=len(self.spool.pending_requests()))
                if futures:
                    idle_ends = 0
                    finished, _ = wait(list(futures), timeout=cfg.poll_s, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        kind, rid = futures.pop(fut)
                        with self._lock:
                            self.inflight.pop(rid, None)
                        try:
                            out = fut.result()
                            print(f"[fetch] {kind} {rid} {out}", flush=True)
                        except Exception as e:  # noqa: BLE001 - one job never stops the producer
                            self.status.count("job_errors")
                            self._event({"event": "fetch_job_error", "record": rid, "kind": kind,
                                         "error": redact(f"{type(e).__name__}: {e}"),
                                         "traceback": redact(traceback.format_exc())[-2000:]})
                            print(f"[fetch] {kind} {rid} job error {redact(type(e).__name__)}", flush=True)
                            if kind in ("triage", "refetch"):
                                self._remove_partial(f"{kind} job error")
                    continue
                if self.stop_reason:
                    print(f"[fetch] stopping: {self.stop_reason}", flush=True)
                    self._event({"event": "fetch_stop", "reason": self.stop_reason})
                    return EXIT_CONSUMER_GONE
                # queue empty, nothing in flight, no request: wait while the
                # consumer may still ask for deep passes
                self._repair_requests()
                states = Counter(self.spool.states().values())
                waiting = states.get("ready", 0) + states.get("awaiting_deep", 0) + states.get("deep_ready", 0)
                waiting += len(self.spool.pending_requests())
                gone = self._consumer_gone()
                if waiting and not gone:
                    idle_ends = 0
                    self.status.update(state="idle_waiting_for_consumer", spool_states=dict(states))
                    time.sleep(cfg.poll_s)
                    continue
                if not gone:
                    # nothing left, twice, one poll apart: the consumer can be
                    # between deleting an invalid record and asking for it
                    # again
                    idle_ends += 1
                    if idle_ends < 2:
                        time.sleep(max(cfg.poll_s, 1.0))
                        continue
                self._event({"event": "fetch_end", "spool_states": dict(states)})
                print(f"[fetch] done; spool {dict(states)}", flush=True)
                return 0

    def _next_job(self, queue: deque, by_id: dict) -> tuple[str, dict] | None:
        for req in self.spool.pending_requests():
            rid = req.get("record_id")
            if rid in self.inflight or not valid_record_id(str(rid)):
                continue
            rec = dict(by_id.get(rid) or {"source_id": rid})
            rec["_refetches"] = int(req.get("refetches") or 0)
            return req.get("kind") or "refetch", rec
        while queue:
            rec = queue.popleft()
            if rec["source_id"] in self.inflight:
                continue
            if self.spool.state(rec["source_id"]) in ("ready", "awaiting_deep", "deep_ready"):
                continue                               # served by a refetch meanwhile; the consumer owns it
            return "triage", rec
        return None


# ---------------------------------------------------------------------------
# consumer
# ---------------------------------------------------------------------------
class Processor:
    def __init__(self, cfg):
        from .runner import Survey
        if cfg.spool_dir is None or cfg.state_dir is None:
            raise ValueError("process needs --spool-dir and --state-dir")
        self.cfg = cfg
        cfg.downloads_dir = Path(cfg.downloads_dir).resolve()
        self.state = Path(cfg.state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.spool = Spool(cfg.spool_dir, cfg.downloads_dir, cfg.out_dir)
        self.lock = RoleLock(self.state, "process", self.spool.root)
        self.status = RoleStatus(self.state, "process")
        if cfg.shared_state_dir is None:
            cfg.shared_state_dir = self.state          # weblink checks of zenodo.org share the budget
        # The consumer never sends the token, but it registers it so that
        # redact() also works in this process (a token that reached a
        # manifest or an exception text never reaches the outputs).
        try:
            load_token(cfg.token_file)
        except ValueError:
            pass                                       # the producer, which needs it, reports that
        self.survey = Survey(cfg, role="process")
        self.scrape = {r["source_id"]: r for r in load_scrape(cfg.scrape_path)}
        # final rows (with their finish time): a record whose row was written
        # after its spool marker is done, even if its dir survived a kill
        self.results = ResultsTail(self.survey.results_path)
        self.keeper = None
        if getattr(cfg, "keep_dir", None):
            from .keep import STATUS_NAME, Keeper
            keep_max = getattr(cfg, "keep_max_gb", None)
            self.keeper = Keeper(cfg.keep_dir, self.spool, cfg.downloads_dir, cfg.out_dir,
                                 max_bytes=keep_max * 1e9 if keep_max else None,
                                 floor_bytes=cfg.disk_floor_gb * 1e9, spool_max_bytes=cfg.spool_max_gb * 1e9,
                                 status_path=self.state / STATUS_NAME, log_event=self.survey_event,
                                 scratch_dir=cfg.scratch_dir, state_dir=self.state)
            self.status.update(keep=self.keeper.summary())
            # records parked by an earlier consumer (moves that failed, or a
            # run without --keep-dir): their moves are tried again now
            for rid in self.spool.clear_keep_failed():
                self.survey_event({"event": "keep_retry_after_restart", "record": rid})
        self._stop = False
        self._in_record: tuple[str, int] | None = None     # (record id, attempts before the bump)

    def _on_signal(self, signum, _frame):
        """SIGTERM / SIGINT (a pipeline stop or restart): finish the current
        record, then exit 0. Its attempts bump is undone at once, so a stop
        (or a SIGKILL that follows it) never counts as a crash inside the
        record."""
        self._stop = True
        cur = self._in_record
        if cur is not None:
            rid, prev = cur
            try:
                p = self.spool.record_dir(rid) / ".attempts"
                if p.parent.is_dir():
                    tmp = p.with_name(".attempts.signal.tmp")
                    tmp.write_text(f"{prev}\n", encoding="ascii")
                    tmp.replace(p)
            except (OSError, ValueError):
                pass
        # no print here: a handler that writes to stdout while the main
        # thread is inside a print raises "reentrant call"

    def run(self) -> int:
        self.status.start()
        prev_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                prev_handlers[sig] = signal.signal(sig, self._on_signal)
        try:
            code = self._run()
        except Exception as e:  # noqa: BLE001
            _append_jsonl(self.state / "process_events.jsonl",
                          {"event": "process_crashed", "error": redact(f"{type(e).__name__}: {e}"),
                           "traceback": redact(traceback.format_exc())[-3000:]})
            self.status.close("failed")
            raise
        finally:
            for sig, h in prev_handlers.items():
                signal.signal(sig, h)
            if self.survey.links:
                self.survey.links.save()
            disk = getattr(self.survey, "disk", None)
            if disk is not None:
                disk.close()
        self.status.close("stopped" if self._stop else "finished")
        self.lock.close()
        return code

    def _run(self) -> int:
        cfg = self.cfg
        gone = 0
        n_done = 0
        t0 = time.time()
        print("[process] watching the spool", flush=True)
        while True:
            if self._stop:
                print("[process] stopping on a signal (after the current record)", flush=True)
                self.survey_event({"event": "process_stop", "reason": "signal"})
                return 0
            ready = self.spool.ready_records()
            if cfg.ids:
                want = {str(i) for i in cfg.ids}
                ready = [r for r in ready if r in want]
            if not ready:
                if role_alive(self.state, "fetch"):
                    gone = 0
                    self.status.update(state="idle", current=[])
                    time.sleep(cfg.poll_s)
                    continue
                gone += 1
                if gone < PRODUCER_GONE_CHECKS:
                    time.sleep(cfg.poll_s)
                    continue
                states = Counter(self.spool.states().values())
                pending = len(self.spool.pending_requests())
                msg = f"producer not running; spool {dict(states)}, {pending} requests wait for it"
                print(f"[process] stopping: {msg}", flush=True)
                self.survey_event({"event": "process_stop", "reason": msg})
                return 0
            gone = 0
            for rid in ready:
                if self._stop:
                    break
                self.status.update(state="running", current=[rid])
                out = self.process_one(rid)
                n_done += out == "row"
                self.status.count(out)
                self.status.update(rate_per_h=round(n_done / max(1e-6, time.time() - t0) * 3600, 1))

    def survey_event(self, ev: dict):
        from .runner import _log_event
        _log_event(self.survey.events_path, ev)

    def process_one(self, rid: str) -> str:
        s = self.survey
        st = self.spool.state(rid)
        if st not in ("ready", "deep_ready"):
            return "skipped"
        rec = self.scrape.get(rid) or {"source_id": rid}
        rdir = self.spool.record_dir(rid)
        if self._done_since_marker(rid, st):
            # its final row is written: the consumer died between the row and
            # deleting the dir; do not classify it a second time. A record
            # whose row says kept: the move a kill interrupted is finished
            # first (keep.py).
            if self.results.kept.get(rid):
                if self.keeper is None:
                    # its row says kept but this consumer has no keep dir: the
                    # files that did not move yet are never deleted
                    from .keep import keep_plan
                    left = len(keep_plan(rdir))
                    if left:
                        self.spool.mark_keep_failed(rid, "row says kept; no --keep-dir in this run")
                        self.survey_event({"event": "keep_parked_no_keep_dir", "record": rid, "files_left": left})
                        return "keep_parked"
                elif not self._keep_move(rid, None):
                    return "keep_move_retry"
            self.spool.remove(rid)
            self.survey_event({"event": "spool_already_done", "record": rid})
            return "already_done"
        m = None
        try:
            m = self.spool.read_manifest(rid)
            dm = self.spool.read_manifest(rid, deep=True) if st == "deep_ready" else None
            problems = self.spool.validate(rid, m) + (self.spool.validate(rid, dm) if dm else [])
        except (OSError, ValueError) as e:
            problems = [f"manifest unreadable: {type(e).__name__}: {e}"]
        if problems:
            refetches = int((m or {}).get("refetches") or 0)
            self.survey_event({"event": "spool_invalid", "record": rid, "problems": problems[:10],
                               "refetches": refetches})
            if refetches >= MAX_REFETCHES:
                # the same files fail again and again: an error row, not a
                # loop. The row goes first: a kill in between leaves the dir,
                # which the done check above then removes.
                row = s._base_row(rec)
                row.update({"status": "error", "sampling_mode": "none", "finished_at": _now(),
                            "error": redact(f"spooled files failed validation after {refetches} refetches: "
                                            f"{problems[0]}")[:500]})
                s._append_row(row)
                self.results.poll()
                self.spool.remove(rid)
                print(f"[process] {rid} error: spool invalid after {refetches} refetches", flush=True)
                return "row"
            # Back to the producer with no window in which the record has no
            # dir, no request and no row: the markers go first (the dir is
            # then 'fetching'; the producer removes and refetches such a dir,
            # and restores its request when a kill left none), then the
            # request. The producer, not the consumer, deletes the dir.
            self.spool.unmark(rid)
            self.spool.request(rid, "refetch", {"refetches": refetches + 1})
            print(f"[process] {rid} spool invalid ({problems[0]}); refetch requested", flush=True)
            return "refetch_requested"
        attempts = self.spool.bump_attempts(rid)
        self._in_record = (rid, attempts - 1)
        try:
            return self._process_bumped(rid, rec, rdir, m, dm, attempts)
        finally:
            self._in_record = None

    KEEP_MOVE_TRIES = 3
    keep_retry_s = 2.0                           # backoff base between failed tries (2 s, then 4 s)

    def _keep_move(self, rid: str, local) -> bool:
        """Move the record's files to the keep dir. True when every file
        moved (the caller may then delete the spool dir). False otherwise,
        and the spool dir must stay: the move is tried again (after a
        backoff) from the already_done path, and after KEEP_MOVE_TRIES
        failures the record is parked (keep_failed: its unmoved files stay in
        the spool, nothing deletes them, the consumer goes on with other
        records; the next process role start tries again)."""
        try:
            res = self.keeper.move(rid, local)
        except OSError as e:
            n = self.keeper.failures.get(rid, 0) + 1
            self.keeper.failures[rid] = n
            self.survey_event({"event": "keep_move_failed", "record": rid, "try": n,
                               "error": redact(f"{type(e).__name__}: {e}")[:300]})
            self.keeper.account(rid)             # the files that did move count against --keep-max-gb
            self.status.update(keep=self.keeper.summary())
            if n < self.KEEP_MOVE_TRIES:
                if self.keep_retry_s > 0:
                    time.sleep(self.keep_retry_s * 2 ** (n - 1))
                return False
            from .keep import keep_plan
            left = len(keep_plan(self.spool.record_dir(rid)))
            self.spool.mark_keep_failed(rid, f"keep move failed {n} times: {redact(type(e).__name__)}")
            self.keeper.failures.pop(rid, None)
            self.status.count("keep_failed")
            self.survey_event({"event": "keep_move_gave_up", "record": rid, "files_left_in_spool": left})
            return False
        self.keeper.failures.pop(rid, None)
        self.status.count("kept")
        self.status.update(keep=self.keeper.summary())
        self.survey_event({"event": "kept", "record": rid, "moved": res["moved"], "files": res["files"],
                           "bytes": res["bytes"]})
        return True

    def _done_since_marker(self, rid: str, st: str) -> bool:
        """True when the record's last final row was written after the spool
        marker the consumer is about to act on (a row of an earlier run, for
        example one retried with --retry-status, is older than the marker)."""
        self.results.poll()
        t = self.results.final_at.get(rid)
        if t is None or not is_final({"status": self.results.status.get(rid)}):
            return False
        marker = self.spool.record_dir(rid) / ("DEEP_READY" if st == "deep_ready" else "READY")
        try:
            return t >= int(marker.stat().st_mtime)     # finished_at has a 1 s resolution
        except OSError:
            return False

    def _process_bumped(self, rid, rec, rdir, m, dm, attempts) -> str:
        cfg = self.cfg
        s = self.survey
        if attempts > cfg.max_attempts:
            row = s.crashed_row(rec, None, in_scrape=rid in self.scrape,
                                error=(f"processing was started {attempts - 1} times and never finished (the "
                                       "consumer died inside this record); rerun with --retry-status crashed"))
            s._append_row(row)
            self.results.poll()
            self.spool.remove(rid)
            self.survey_event({"event": "record_crashed", "record": rid, "attempts": attempts - 1})
            print(f"[process] {rid} crashed (attempts {attempts - 1})", flush=True)
            return "crashed"
        started = _now()
        s._append_row({"record_id": rid, "status": "started", "started_at": started, "attempt": attempts,
                       "pass": "deep" if dm else "triage"})
        t0 = time.time()
        row, want_deep = s.finish_record(rec, rdir, m, dm, allow_deep_request=dm is None)
        if want_deep:
            self.spool.mark_awaiting_deep(rid)
            self.spool.request(rid, "deep")
            (rdir / ".attempts").write_text("0\n", encoding="ascii")
            s._append_row({"record_id": rid, "status": "awaiting_deep", "started_at": started,
                           "triage_s": round(time.time() - t0, 1)})
            print(f"[process] {rid} awaiting_deep", flush=True)
            return "deep_requested"
        row["started_at"] = started
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["fetch_s"] = (m.get("fetch_s") or 0) + ((dm or {}).get("fetch_s") or 0)
        row["spool_bytes"] = int(m.get("bytes") or 0) + int((dm or {}).get("bytes") or 0)
        row["finished_at"] = _now()
        plan = self.keeper.prepare(rid, row, (m, dm)) if self.keeper is not None else None
        s._append_row(row)                       # durable (fsync) before the spool dir goes
        self.results.poll()
        if plan is not None and plan["kept"]:
            if not self._keep_move(rid, plan["local"]):
                return "row"                     # the move is retried from the already_done path
        elif plan is not None and plan["reason"] != "no eye image":
            self.status.count("keep_refused")
            self.survey_event({"event": "keep_refused", "record": rid, "reason": plan["reason"]})
        self.spool.remove(rid)
        if s.links:
            try:
                s.links.save(min_interval_s=120)
            except OSError as e:                 # the link cache is a convenience, never a reason to stop
                self.survey_event({"event": "link_cache_save_failed", "error": f"{type(e).__name__}: {e}"})
        print(f"[process] {rid} {row['status']} images={row.get('n_image_files', 0)} "
              f"classified={row.get('n_classified', 0)} pass={row.get('sampling_pass') or '-'} "
              f"dominant={row.get('dominant_class') or '-'} ({row['elapsed_s']}s)", flush=True)
        return "row"


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------
class ResultsTail:
    """Reads the results JSONL incrementally: last status per record and the
    finish times of the last hour (the file can reach hundreds of MB)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.status: dict[str, str] = {}
        self.final_at: dict[str, float] = {}    # finished_at (epoch) of each record's last final row
        self.kept: dict[str, bool] = {}         # 'kept' of each record's last final row (keep dir)
        self.finished: deque = deque()          # epoch seconds of final rows seen
        self._partial = b""

    def poll(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:                 # replaced: read again
            self.offset, self.status, self.final_at, self.kept, self._partial = 0, {}, {}, {}, b""
        now = time.time()
        with open(self.path, "rb") as fp:
            fp.seek(self.offset)
            while True:
                chunk = fp.read(8 << 20)        # bounded memory on a results file of hundreds of MB
                if not chunk:
                    break
                self.offset = fp.tell()
                lines = (self._partial + chunk).split(b"\n")
                self._partial = lines.pop()     # an unfinished last line
                self._take(lines, now)
        while self.finished and now - self.finished[0] > 3600:
            self.finished.popleft()

    def _take(self, lines: list[bytes], now: float):
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rid = str(row.get("record_id") or "")
            if not rid:
                continue
            st = row.get("status")
            self.status[rid] = st
            if is_final(row):
                self.kept[rid] = bool(row.get("kept"))
                t = now
                fin = row.get("finished_at")
                if fin:
                    try:
                        t = datetime.fromisoformat(fin).timestamp()
                        self.final_at[rid] = t
                    except (TypeError, ValueError):
                        pass
                self.finished.append(t)
        while self.finished and now - self.finished[0] > 3600:
            self.finished.popleft()


class Monitor:
    def __init__(self, cfg):
        if cfg.state_dir is None:
            raise ValueError("monitor needs --state-dir")
        self.cfg = cfg
        self.state = Path(cfg.state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.spool = Spool(cfg.spool_dir, cfg.downloads_dir, cfg.out_dir) if cfg.spool_dir else None
        self.results = ResultsTail(cfg.out_dir / "survey_results.jsonl")
        try:
            self.n_scope = len(records_in_scope(cfg))
        except (OSError, ValueError):
            self.n_scope = None
        self.prev: dict | None = None

    def snapshot(self) -> dict:
        cfg = self.cfg
        self.results.poll()
        final = Counter(st for st in self.results.status.values() if st not in ("started", "awaiting_deep"))
        n_final = sum(final.values())
        disk_path = self.spool.root if self.spool else cfg.out_dir
        du = shutil.disk_usage(disk_path)
        snap = {"ts": _now(), "epoch": time.time(),
                "disk": {"path": str(disk_path), "total_gb": round(du.total / 1e9, 1),
                         "used_gb": round(du.used / 1e9, 1), "free_gb": round(du.free / 1e9, 1),
                         "floor_gb": cfg.disk_floor_gb}}
        if self.spool:
            states = Counter(self.spool.states().values())
            reqs = Counter(r.get("kind") for r in self.spool.pending_requests())
            spool_bytes = self.spool.bytes_used()
            snap["spool"] = {"bytes_gb": round(spool_bytes / 1e9, 2), "max_gb": cfg.spool_max_gb,
                             "records": dict(states), "queue_ready": states.get("ready", 0) + states.get("deep_ready", 0),
                             "requests": dict(reqs)}
        rate_h = len(self.results.finished)            # final rows in the last hour
        remaining = (self.n_scope - n_final) if self.n_scope is not None else None
        snap["results"] = {"n_final": n_final, "by_status": dict(final.most_common()),
                           "in_progress": sum(1 for st in self.results.status.values()
                                              if st in ("started", "awaiting_deep")),
                           "n_scope": self.n_scope, "remaining": remaining,
                           "rows_last_hour": rate_h,
                           "eta_hours": round(remaining / rate_h, 1) if remaining and rate_h else None}
        roles = {}
        for role in ("fetch", "process"):
            st = _read_json(self.state / f"{role}_status.json") or {}
            alive = role_alive(self.state, role)
            state = st.get("state")
            if not alive and state not in (None, "finished", "stopped", "failed"):
                state = f"dead (last state {state})"          # killed: it never wrote its end state
            roles[role] = {"alive": alive, "state": state,
                           "heartbeat": st.get("heartbeat"), "counters": st.get("counters"),
                           "current": st.get("current"), "waiting": st.get("waiting")}
        snap["roles"] = roles
        keep_dir = getattr(cfg, "keep_dir", None)
        if keep_dir:
            from .keep import STATUS_NAME
            ks = _read_json(self.state / STATUS_NAME) or {}
            counters = (roles.get("process") or {}).get("counters") or {}
            snap["keep"] = {"dir": str(keep_dir), "bytes_gb": ks.get("bytes_gb"), "records": ks.get("records"),
                            "max_gb": getattr(cfg, "keep_max_gb", None) or None,
                            "kept_this_run": counters.get("kept", 0),
                            "refused_this_run": counters.get("keep_refused", 0)}
            # kept files stay on the disk: keeping stops while the free space
            # with an empty spool would drop under floor + spool max (keep.py)
            snap["disk"]["keep_gb"] = ks.get("bytes_gb")
            snap["disk"]["keep_stops_below_gb"] = round(cfg.disk_floor_gb + cfg.spool_max_gb, 1)
        fstat = _read_json(self.state / "fetch_status.json") or {}
        if self.prev is not None:
            dt = snap["epoch"] - self.prev["epoch"]
            pb = (self.prev.get("_fetch_bytes") or 0)
            fb = (fstat.get("counters") or {}).get("bytes_fetched") or 0
            snap["fetch_mb_per_s"] = round((fb - pb) / 1e6 / dt, 2) if dt > 0 else None
        snap["_fetch_bytes"] = (fstat.get("counters") or {}).get("bytes_fetched") or 0
        pipe = _read_json(self.state / "pipeline_status.json")
        if pipe:
            snap["pipeline"] = pipe
        return snap

    def run(self, interval: float, once: bool = False, exit_when_idle: bool = False) -> int:
        idle = 0
        while True:
            snap = self.snapshot()
            self.prev = snap
            out = {k: v for k, v in snap.items() if not k.startswith("_")}
            _write_json(self.state / "status.json", out)
            _append_jsonl(self.state / "monitor.jsonl", {k: v for k, v in out.items() if k != "pipeline"})
            sp = out.get("spool") or {}
            res = out["results"]
            kp = out.get("keep")
            kept_txt = f" | keep {kp.get('bytes_gb')} GB {kp.get('records')} rec" if kp else ""
            print(f"[monitor] {out['ts']} free {out['disk']['free_gb']} GB{kept_txt} | spool {sp.get('bytes_gb')} GB "
                  f"{sp.get('records')} req {sp.get('requests')} | done {res['n_final']}/{res['n_scope']} "
                  f"({res['rows_last_hour']}/h, eta {res['eta_hours']} h) | fetch "
                  f"{'up' if out['roles']['fetch']['alive'] else 'down'} {out['roles']['fetch']['state']} "
                  f"| process {'up' if out['roles']['process']['alive'] else 'down'} "
                  f"{out['roles']['process']['state']}", flush=True)
            if once:
                return 0
            if exit_when_idle and not out["roles"]["fetch"]["alive"] and not out["roles"]["process"]["alive"]:
                idle += 1
                if idle >= 2:
                    return 0
            else:
                idle = 0
            time.sleep(interval)


# ---------------------------------------------------------------------------
# pipeline supervisor
# ---------------------------------------------------------------------------
PR_SET_PDEATHSIG = 1


def _die_with_parent(parent_pid: int):
    """preexec_fn for the role processes (Linux): the kernel sends SIGTERM to
    the child when the supervisor dies, also by SIGKILL or an OOM kill, so
    no fetch or process keeps running without a supervisor (and blocks the
    next pipeline start with 'already running'). SIGTERM is the graceful
    stop: process finishes its current record, fetch leaves partial records
    for the next start to remove. Elsewhere, or when prctl is unavailable,
    it does nothing."""
    def preexec():
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
        except (OSError, AttributeError):
            return
        if os.getppid() != parent_pid:           # the supervisor died before prctl took effect
            os.kill(os.getpid(), signal.SIGTERM)
    return preexec


def run_pipeline(cfg, passthrough: list[str], interval: float, base_cmd: list[str] | None = None) -> int:
    """Start fetch, then process and monitor, as child processes with the
    same options; log each child's exit. A fetch or process that exits
    non-zero (a crash, an OOM kill) is started again after a backoff
    (``restart_backoff_s`` doubled per restart, at most 300 s), at most
    ``max_role_restarts`` times per role; the consumer's attempts guard
    turns a record that keeps killing it into a crashed row. A process that
    ends cleanly (exit 0: it stops when fetch is gone for a few polls) while
    fetch still runs is started again after ``restart_backoff_s``, and a
    process that ended cleanly while fetch waits for its restart is started
    again with fetch. Those clean restarts do not count against
    ``max_role_restarts`` and do not grow the backoff: only crashes do.
    On Linux every role gets SIGTERM when the supervisor itself dies (also
    by SIGKILL), so a killed pipeline leaves no role running on its own.
    ``base_cmd`` replaces the python module command (tests)."""
    state = Path(cfg.state_dir)
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    events = state / "pipeline_events.jsonl"
    status_path = state / "pipeline_status.json"
    for role in ("fetch", "process"):
        if role_alive(state, role):
            raise ValueError(f"a '{role}' process is already running with state dir {state}")
    base = list(base_cmd) if base_cmd else [sys.executable, "-m", "envision_eye_actionable.survey"]
    max_restarts = int(getattr(cfg, "max_role_restarts", 20))
    backoff0 = float(getattr(cfg, "restart_backoff_s", 10.0))
    tick = min(5.0, max(0.2, float(getattr(cfg, "poll_s", 5.0))))
    procs: dict[str, dict] = {}
    restarts: Counter = Counter()               # restarts after a crash (capped, drive the backoff)
    clean_restarts: Counter = Counter()         # restarts after a clean exit (not capped)
    crashes: Counter = Counter()
    due: dict[str, float] = {}                  # role -> monotonic time of its restart
    due_clean: set[str] = set()                 # roles in due whose restart follows a clean exit
    given_up: set[str] = set()
    stopping = {"flag": False}

    def start(role: str):
        log = open(logs / f"{role}.log", "a", encoding="utf-8")
        log.write(f"\n==== {_now()} start {role}\n")
        log.flush()
        p = subprocess.Popen(base + [role] + passthrough, stdout=log, stderr=subprocess.STDOUT,
                             preexec_fn=_die_with_parent(os.getpid()) if sys.platform.startswith("linux")
                             else None)
        procs[role] = {"proc": p, "log": log, "pid": p.pid, "started": _now(), "exit_code": None, "ended": None,
                       "restarts": restarts[role], "clean_restarts": clean_restarts[role]}
        _append_jsonl(events, {"event": "role_started", "role": role, "pid": p.pid})

    def note() -> str:
        parts = []
        if crashes:
            parts.append("crashed: " + ", ".join(f"{r} x{n}" for r, n in sorted(crashes.items())))
        if given_up:
            parts.append("given up: " + ", ".join(sorted(given_up)))
        return "; ".join(parts)

    def write_status(finished: bool = False, extra: str = ""):
        _write_json(status_path, {"pid": os.getpid(), "updated": _now(), "finished": finished,
                                  "note": "; ".join(x for x in (note(), extra) if x),
                                  "restarts": dict(restarts), "clean_restarts": dict(clean_restarts),
                                  "pending_restarts": sorted(due),
                                  "roles": {r: {k: v for k, v in d.items() if k not in ("proc", "log")}
                                            for r, d in procs.items()}})

    def running(role: str) -> bool:
        return role in procs and procs[role]["exit_code"] is None

    def schedule(role: str, why: str, clean: bool = False):
        if stopping["flag"] or role in due:
            return
        if clean:
            due[role] = time.monotonic() + backoff0
            due_clean.add(role)
            print(f"[pipeline] {role}: {why}; restart in {backoff0:.0f}s", flush=True)
            return
        if restarts[role] >= max_restarts:
            given_up.add(role)
            _append_jsonl(events, {"event": "role_given_up", "role": role, "restarts": restarts[role],
                                   "reason": why})
            print(f"[pipeline] {role}: {why}; {restarts[role]} restarts used, not restarted again", flush=True)
            return
        delay = min(300.0, backoff0 * (2 ** restarts[role]))
        due[role] = time.monotonic() + delay
        print(f"[pipeline] {role}: {why}; restart in {delay:.0f}s (see logs/{role}.log)", flush=True)

    def on_signal(signum, _frame):
        stopping["flag"] = True
        due.clear()
        due_clean.clear()
        for d in procs.values():
            if d["proc"].poll() is None:
                d["proc"].send_signal(signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    start("fetch")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not role_alive(state, "fetch") and procs["fetch"]["proc"].poll() is None:
        time.sleep(0.5)
    start("process")
    start("monitor")
    write_status()
    while True:
        for role, d in list(procs.items()):
            if d["exit_code"] is None and d["proc"].poll() is not None:
                d["exit_code"] = d["proc"].returncode
                d["ended"] = _now()
                d["log"].close()
                ev = {"event": "role_exited", "role": role, "exit_code": d["exit_code"]}
                if d["exit_code"] != 0 and role != "monitor":
                    ev["event"] = "role_crashed"
                    crashes[role] += 1
                _append_jsonl(events, ev)
                if role == "monitor" or stopping["flag"]:
                    continue
                if d["exit_code"] != 0:
                    if role == "fetch" and "process" in given_up:
                        given_up.add("fetch")         # nothing would ever consume what it fetches
                        _append_jsonl(events, {"event": "role_given_up", "role": "fetch",
                                               "reason": "process was given up"})
                    else:
                        schedule(role, f"exited with code {d['exit_code']}")
                elif role == "process" and running("fetch"):
                    schedule(role, "ended while fetch still runs", clean=True)
                # a process that ended cleanly while fetch waits for its
                # restart is started again with fetch (below)
        now = time.monotonic()
        for role in sorted(due, key=lambda r: r != "fetch"):          # fetch first, as at the start
            if now >= due[role] and not stopping["flag"]:
                del due[role]
                clean = role in due_clean
                due_clean.discard(role)
                if clean:
                    clean_restarts[role] += 1
                else:
                    restarts[role] += 1
                start(role)
                _append_jsonl(events, {"event": "role_restarted", "role": role, "restart": restarts[role],
                                       "clean_restarts": clean_restarts[role],
                                       "reason": "clean exit" if clean else "crash"})
                if role == "fetch" and not running("process") and "process" not in due \
                        and "process" not in given_up:
                    clean_restarts["process"] += 1
                    start("process")
                    _append_jsonl(events, {"event": "role_restarted", "role": "process",
                                           "restart": restarts["process"],
                                           "clean_restarts": clean_restarts["process"],
                                           "reason": "fetch restarted"})
        write_status()
        workers_done = not due and all(procs[r]["exit_code"] is not None for r in ("fetch", "process"))
        if workers_done:
            mon = procs["monitor"]
            if mon["exit_code"] is None:
                mon["proc"].terminate()
                try:
                    mon["proc"].wait(30)
                except subprocess.TimeoutExpired:
                    mon["proc"].kill()
                mon["exit_code"] = mon["proc"].returncode
                mon["ended"] = _now()
                mon["log"].close()
            try:
                Monitor(cfg).run(interval, once=True)
            except Exception:  # noqa: BLE001 - the final snapshot is best effort
                pass
            failed = bool(given_up) or any(procs[r]["exit_code"] != 0 for r in ("fetch", "process"))
            write_status(finished=True, extra=("stopped by signal" if stopping["flag"]
                                               else ("failed" if failed else "done")))
            _append_jsonl(events, {"event": "pipeline_end", "crashes": dict(crashes), "restarts": dict(restarts),
                                   "given_up": sorted(given_up)})
            return 1 if failed else 0
        time.sleep(tick)
