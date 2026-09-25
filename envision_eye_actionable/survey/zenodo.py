"""Zenodo access for the survey: scrape loading, record metadata, downloads.

Metadata sources, in order of preference:

* legacy record JSON (``GET /api/records/<id>``): file list with md5 and
  direct links, access_right, relations.version. Read from the
  envision-discovery metadata dir when present, else fetched and cached.
* DataCite JSON (same URL, ``Accept: application/vnd.datacite.datacite+json``):
  creators with nameType/ORCID/ROR, rightsList, dates, relatedIdentifiers,
  descriptions, fundingReferences. It maps almost 1:1 onto the AI-READI
  dataset_description schema. Always fetched once and cached.

Zenodo's guest limit is about 133 requests per minute, so every request
(metadata calls, file downloads, zip listings, member fetches and range
reads, across all worker threads) goes through one shared throttle: at most
``max_per_minute`` requests in any 60 s window (default 120) and at least
``min_interval`` seconds between two requests. A 429 sets a shared cooldown
from Retry-After (seconds or HTTP-date, at least 60 s when absent) that every
worker waits on; 5xx responses and transport errors back off exponentially
per request. Zenodo's limit is per IP, so with ``shared_state_dir`` two
things are shared between the processes using that dir: the 429 cooldown
(a wall-clock not-before time in ``zenodo_not_before``, which every process
waits on) and the per-minute budget. For the budget, every request is
logged as a wall-clock time in ``zenodo_requests`` under a file lock, and a
request goes out only while all the processes together sent fewer than
``shared_per_minute`` (default 120, ``--shared-rpm``) in the last 60 s. The
per-process ``max_per_minute`` (``--rpm``) still applies on top, so it can
give one process a smaller share, but two processes left at the default no
longer add up to more than the combined budget.

Zenodo's X-RateLimit-Remaining header is read on every response: when it
drops to RATE_REMAINING_FLOOR or below, every request pauses until the
window resets (X-RateLimit-Reset), whatever the local count says.

Access token: with ``ZENODO_TOKEN`` in the environment, ``token_file``
(one line) or, when neither is given, DEFAULT_TOKEN_FILE when it exists,
requests to zenodo.org carry ``Authorization: Bearer``. The token does not
raise the rate limit; it only opens records the account was granted. It is
sent to zenodo.org hosts only (never to other hosts, never on a redirect
elsewhere) and is never logged, printed or written anywhere. As a
safeguard it is registered with redact() when loaded: result, event,
status and predictions lines pass through redact(), and
install_log_redaction() puts a RedactingFilter on the log handlers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from requests.auth import AuthBase

logger = logging.getLogger(__name__)

API = "https://zenodo.org/api/records"
MAX_PER_MINUTE = 120   # below Zenodo's 133 per 60 s window (per IP and User-Agent, token or not)
RETRY_STATUSES = (429, 500, 502, 503, 504)
SHARED_COOLDOWN_NAME = "zenodo_not_before"   # in --shared-state-dir: wall-clock end of a 429 cooldown
SHARED_REQUESTS_NAME = "zenodo_requests"     # in --shared-state-dir: wall-clock times of the last minute's requests
RATE_REMAINING_FLOOR = 3                      # pause until the window resets at this many requests left
TOKEN_ENV = "ZENODO_TOKEN"
# Read when neither ZENODO_TOKEN nor --token-file is given and the file exists.
DEFAULT_TOKEN_FILE = Path("~/.config/envision-survey/zenodo_token")
REDACTED = "[REDACTED]"

_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()


def register_secret(value: str | None):
    """Remember a secret so redact() removes it from any text."""
    if value and len(value) >= 8:
        with _SECRETS_LOCK:
            _SECRETS.add(value)


def redact(text):
    """``text`` with every registered secret replaced by [REDACTED]
    (non-strings are returned unchanged)."""
    if not isinstance(text, str) or not _SECRETS:
        return text
    for s in tuple(_SECRETS):
        if s in text:
            text = text.replace(s, REDACTED)
    return text


class RedactingFilter(logging.Filter):
    """Logging filter that removes registered secrets from the message and
    its arguments (attach it to handlers: filters on a logger do not see
    records propagated from child loggers)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _SECRETS:
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001 - never let logging fail
                return True
            clean = redact(msg)
            if clean != msg:
                record.msg, record.args = clean, None
            if record.exc_info and record.exc_info[1] is not None:
                text = logging.Formatter().formatException(record.exc_info)
                if redact(text) != text:
                    record.exc_info, record.exc_text = None, None
                    record.msg = f"{record.msg}\n{redact(text)}"
        return True


def install_log_redaction(logger_: logging.Logger | None = None):
    """Attach a RedactingFilter to every handler of ``logger_`` (default:
    the root logger), once."""
    lg = logger_ or logging.getLogger()
    for h in lg.handlers:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())


def token_source(token_file: Path | None = None) -> str:
    """Where load_token takes the token from: 'env', 'file', 'default_file'
    or 'none' (the value is never returned)."""
    if os.environ.get(TOKEN_ENV, "").strip():
        return "env"
    if token_file is not None:
        return "file"
    if DEFAULT_TOKEN_FILE.expanduser().is_file():
        return "default_file"
    return "none"


def load_token(token_file: Path | None = None, use_default: bool = True) -> str | None:
    """Zenodo access token from ``ZENODO_TOKEN``, else the first line of
    ``token_file``, else (``use_default``) of DEFAULT_TOKEN_FILE when it
    exists; None when there is none. An explicit ``token_file`` that cannot
    be read is an error. The token is registered for redaction and never
    logged."""
    tok = os.environ.get(TOKEN_ENV, "").strip()
    if tok:
        register_secret(tok)
        return tok
    path = Path(token_file).expanduser() if token_file is not None else None
    if path is None and use_default and DEFAULT_TOKEN_FILE.expanduser().is_file():
        path = DEFAULT_TOKEN_FILE.expanduser()
    if path is None:
        return None
    try:
        tok = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        raise ValueError(f"cannot read the Zenodo token file {path}") from None
    register_secret(tok)
    return tok or None


def _is_zenodo_host(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host == "zenodo.org" or host.endswith(".zenodo.org")


class ZenodoBearer(AuthBase):
    """Adds ``Authorization: Bearer`` to requests for zenodo.org hosts only.
    Its repr never shows the token."""

    def __init__(self, token: str):
        self._token = token

    def __call__(self, r):
        if _is_zenodo_host(r.url):
            r.headers["Authorization"] = f"Bearer {self._token}"
        else:
            r.headers.pop("Authorization", None)
        return r

    def __repr__(self) -> str:
        return "ZenodoBearer(<token hidden>)"

    __str__ = __repr__


class RemoteError(RuntimeError):
    """A Zenodo request failed for good (non-retryable status or retries used up)."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status
DATACITE_ACCEPT = "application/vnd.datacite.datacite+json"
USER_AGENT = "envision-eye-actionable-survey/0.1 (+https://github.com/EyeACT/envision-eye-actionable)"


def retry_after_seconds(value: str | None, fallback: float, status: int = 429) -> float:
    """Seconds to wait from a Retry-After header (delta-seconds, decimal or
    HTTP-date), capped at one hour. Without a usable value a 429 waits at
    least 60 s (Zenodo's rate window); other statuses use ``fallback``."""
    floor = max(fallback, 60.0) if status == 429 else fallback
    if not value:
        return floor
    value = value.strip()
    try:
        return min(3600.0, max(0.0, float(value)))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return min(3600.0, max(0.0, (when - datetime.now(timezone.utc)).total_seconds()))
    except (TypeError, ValueError, IndexError, OverflowError):
        return floor


def load_scrape(path: Path) -> list[dict]:
    """Load the envision-discovery scrape (a JSON list, or {"results": [...]})."""
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        data = data.get("results") or data.get("records") or []
    out = []
    for rec in data:
        sid = str(rec.get("source_id") or rec.get("id") or "").strip()
        if not sid:
            continue
        rec["source_id"] = sid
        out.append(rec)
    return out


class ZenodoClient:
    """Throttled, caching client for Zenodo record metadata."""

    def __init__(
        self,
        cache_dir: Path,
        metadata_dir: Path | None = None,
        min_interval: float = 1.0,
        offline: bool = False,
        max_per_minute: int | None = MAX_PER_MINUTE,
        shared_state_dir: Path | None = None,
        shared_per_minute: int | None = MAX_PER_MINUTE,
        token: str | None = None,
    ):
        self.cache_dir = cache_dir
        self.metadata_dir = metadata_dir
        self.min_interval = min_interval
        self.max_per_minute = max_per_minute
        self.offline = offline
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.authenticated = bool(token)
        if token:
            register_secret(token)
            self.session.auth = ZenodoBearer(token)
        self._lock = threading.Lock()
        self._last = float("-inf")
        self._not_before = 0.0
        self._window: deque[float] = deque()
        self.n_requests = 0
        self._shared_file = None
        self._shared_mtime = None
        self._requests_file = None
        self.shared_per_minute = shared_per_minute
        if shared_state_dir is not None:
            Path(shared_state_dir).mkdir(parents=True, exist_ok=True)
            self._shared_file = Path(shared_state_dir) / SHARED_COOLDOWN_NAME
            if shared_per_minute:
                self._requests_file = Path(shared_state_dir) / SHARED_REQUESTS_NAME
        (cache_dir / "legacy").mkdir(parents=True, exist_ok=True)
        (cache_dir / "datacite").mkdir(parents=True, exist_ok=True)

    # -- shared 429 cooldown ---------------------------------------------------
    def _read_shared(self):
        """Fold another process's 429 cooldown into this one (caller holds
        the lock). Re-read only when the file changed: one stat per request."""
        if self._shared_file is None:
            return
        try:
            mtime = self._shared_file.stat().st_mtime_ns
        except OSError:
            return
        if mtime == self._shared_mtime:
            return
        self._shared_mtime = mtime
        try:
            until = float(self._shared_file.read_text(encoding="ascii").strip() or 0)
        except (OSError, ValueError):
            return
        left = until - time.time()
        if left > 0:
            self._not_before = max(self._not_before, time.monotonic() + left)

    def _write_shared(self, seconds: float):
        """Publish a cooldown to the other processes (atomic replace; a
        later cooldown already in the file is kept)."""
        if self._shared_file is None:
            return
        until = time.time() + seconds
        try:
            try:
                until = max(until, float(self._shared_file.read_text(encoding="ascii").strip() or 0))
            except (OSError, ValueError):
                pass
            tmp = self._shared_file.with_name(f"{self._shared_file.name}.{os.getpid()}.tmp")
            tmp.write_text(f"{until:.3f}\n", encoding="ascii")
            tmp.replace(self._shared_file)
        except OSError as e:
            logger.warning("could not write the shared 429 cooldown %s: %s", self._shared_file, e)

    def _shared_slot(self) -> float:
        """Take a slot in the combined per-minute budget of the processes
        sharing the state dir (caller holds the thread lock): 0.0 when the
        request may go out (its time is logged), else the seconds to wait.
        A state dir that cannot be read or written is logged and not
        enforced, so a disk problem there never stops the survey."""
        from .locks import locked
        path = self._requests_file
        try:
            with locked(path.with_name(path.name + ".lock")):
                now = time.time()
                try:
                    raw = path.read_text(encoding="ascii").split()
                except FileNotFoundError:
                    raw = []
                times = []
                for x in raw:
                    try:
                        t = float(x)
                    except ValueError:
                        continue
                    if 0.0 <= now - t < 60.0:
                        times.append(t)
                if len(times) >= self.shared_per_minute:
                    return max(1e-3, 60.0 - (now - min(times)) + 1e-3)
                times.append(now)
                path.write_text("".join(f"{t:.3f}\n" for t in times), encoding="ascii")
                return 0.0
        except (OSError, TimeoutError) as e:
            logger.warning("shared request budget %s not enforced for this request: %s", path, e)
            return 0.0

    # -- HTTP -------------------------------------------------------------
    def throttle(self):
        """Wait for this client's turn. Shared by metadata calls and every
        worker thread, so all Zenodo requests together stay under
        ``max_per_minute`` per sliding 60 s window, one per ``min_interval``,
        and all of them honour a 429 cooldown.

        The lock is held only to check and reserve a slot, never while
        sleeping: a worker that gets a 429 can set the cooldown at once, and
        a sleeper re-checks it when it wakes, so it never fires into the
        cooldown. With a shared state dir, a cooldown set by another process
        (after its own 429) is honoured as well, and the request also needs a
        slot in the combined per-minute budget of all the processes sharing
        the dir (``shared_per_minute``)."""
        while True:
            with self._lock:
                self._read_shared()
                now = time.monotonic()
                wait = max(self.min_interval - (now - self._last), self._not_before - now)
                if self.max_per_minute:
                    while self._window and now - self._window[0] >= 60.0:
                        self._window.popleft()
                    if len(self._window) >= self.max_per_minute:
                        # small margin: the oldest request must be strictly
                        # out of the window, float residue included
                        wait = max(wait, 60.0 - (now - self._window[0]) + 1e-3)
                if wait <= 1e-6 and self._requests_file is not None:
                    wait = self._shared_slot()
                if wait <= 1e-6:     # float residue after a sleep must not spin
                    self._last = now
                    self._window.append(now)
                    self.n_requests += 1
                    return
            time.sleep(wait)

    _throttle = throttle  # backwards-compatible name

    def observe(self, r) -> None:
        """Read Zenodo's X-RateLimit-Remaining / X-RateLimit-Reset from a
        response: at RATE_REMAINING_FLOOR requests left or fewer, pause every
        request (of this process and, with a shared state dir, of the others)
        until the window resets."""
        try:
            remaining = r.headers.get("X-RateLimit-Remaining")
            if remaining is None or int(remaining) > RATE_REMAINING_FLOOR:
                return
            reset = r.headers.get("X-RateLimit-Reset")
            if reset:
                v = float(reset)
                wait = v - time.time() if v > 1e9 else v      # epoch seconds, or seconds left
            else:
                wait = 60.0
            wait = min(max(wait, 1.0), 120.0)
        except (TypeError, ValueError, AttributeError):
            return
        logger.info("Zenodo rate limit: %s requests left, pausing %.0fs", remaining, wait)
        self.backoff(wait)

    def backoff(self, seconds: float):
        """Pause every request of this client for ``seconds`` (after a 429),
        and of every process sharing its state dir. Never waits on a sleeping
        worker: throttle() releases the lock while it sleeps."""
        with self._lock:
            self._not_before = max(self._not_before, time.monotonic() + seconds)
            self._write_shared(seconds)

    def get_json(self, url: str, accept: str = "application/json", max_tries: int = 6) -> dict | None:
        """GET a JSON document with throttling and backoff. None on 4xx/give-up."""
        if self.offline:
            return None
        delay = 5.0
        for attempt in range(max_tries):
            self._throttle()
            try:
                r = self.session.get(url, headers={"Accept": accept}, timeout=(20, 60))
            except requests.RequestException as e:
                logger.warning("GET %s failed (%s), attempt %d", url, type(e).__name__, attempt + 1)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            self.observe(r)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return None
            if r.status_code in (429, 500, 502, 503, 504):
                sleep = retry_after_seconds(r.headers.get("Retry-After"), delay, r.status_code)
                logger.warning("GET %s -> %d, backing off %.0fs", url, r.status_code, sleep)
                if r.status_code == 429:
                    self.backoff(sleep)   # everyone waits, throttle() sleeps
                else:
                    time.sleep(sleep)
                delay = min(delay * 2, 120)
                continue
            logger.warning("GET %s -> %d (not retried)", url, r.status_code)
            return None
        return None

    def get(self, url: str, headers: dict | None = None, stream: bool = False, max_tries: int = 6,
            timeout=(20, 120), base_delay: float = 2.0, max_delay: float = 120.0) -> requests.Response:
        """Throttled GET with retries. Returns a 2xx response (the caller
        closes it when ``stream``). 429 honours Retry-After through the shared
        cooldown; 5xx and transport errors back off exponentially
        (``base_delay`` doubling up to ``max_delay``). Raises RemoteError for
        any other status or when the tries are used up."""
        if self.offline:
            raise RemoteError("offline")
        delay = base_delay
        last = "no attempt"
        status = None
        for attempt in range(max_tries):
            self.throttle()
            try:
                r = self.session.get(url, headers=headers or {}, stream=stream, timeout=timeout)
            except requests.RequestException as e:
                last, status = f"{type(e).__name__}", None
                logger.warning("GET %s failed (%s), attempt %d", url, type(e).__name__, attempt + 1)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            self.observe(r)
            if 200 <= r.status_code < 300:
                return r
            status = r.status_code
            retry_after = r.headers.get("Retry-After")
            r.close()
            if status in RETRY_STATUSES:
                sleep = retry_after_seconds(retry_after, delay, status)
                last = f"HTTP {status}"
                logger.warning("GET %s -> %d, backing off %.0fs", url, status, sleep)
                if status == 429:
                    self.backoff(sleep)
                else:
                    time.sleep(sleep)
                delay = min(delay * 2, max_delay)
                continue
            raise RemoteError(f"HTTP {status}", status)
        raise RemoteError(f"gave up after {max_tries} tries ({last})", status)

    # -- Records ----------------------------------------------------------
    def legacy(self, record_id: str) -> dict | None:
        """Legacy record JSON: discovery metadata dir, then cache, then API."""
        if self.metadata_dir:
            p = self.metadata_dir / f"{record_id}.json"
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as fp:
                        return json.load(fp)
                except (OSError, ValueError) as e:
                    logger.warning("Bad metadata file %s: %s", p, e)
        return self._cached("legacy", record_id, f"{API}/{record_id}", "application/json")

    def datacite(self, record_id: str) -> dict | None:
        """DataCite 4.x JSON for the record (cached)."""
        return self._cached("datacite", record_id, f"{API}/{record_id}", DATACITE_ACCEPT)

    def _cached(self, kind: str, record_id: str, url: str, accept: str) -> dict | None:
        p = self.cache_dir / kind / f"{record_id}.json"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as fp:
                    return json.load(fp)
            except (OSError, ValueError):
                pass
        doc = self.get_json(url, accept=accept)
        if doc is not None:
            tmp = p.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(doc, fp)
            tmp.replace(p)
        return doc


def record_files(legacy: dict | None) -> list[dict]:
    """Normalise the legacy file list to [{key, size, md5, url}]."""
    out = []
    for f in (legacy or {}).get("files") or []:
        key = f.get("key") or f.get("filename") or ""
        if not key:
            continue
        checksum = f.get("checksum") or ""
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        links = f.get("links") or {}
        url = links.get("self") or links.get("content") or links.get("download")
        out.append({"key": key, "size": int(f.get("size") or f.get("filesize") or 0),
                    "md5": md5, "url": url})
    return out


def safe_filename(name: str) -> str:
    """Same rule as envision-discovery's downloader: keep the basename only."""
    name = name.replace("\x00", "").strip()
    name = name.replace("\\", "/").split("/")[-1]
    return name or "unnamed"


def _md5_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stream_download(
    session: requests.Session,
    url: str,
    dest: Path,
    expected_size: int | None = None,
    md5: str | None = None,
    max_retries: int = 6,
    throttle=None,
    max_md5_restarts: int = 2,
    room_check=None,
    room_every: int = 64 << 20,
) -> tuple[bool, str | None]:
    """Resumable download to ``dest`` (via ``dest.part``), size and md5 checked.

    Mirrors envision-discovery's ``_stream_download`` (Range resume, backoff)
    and adds md5 verification. Returns (ok, error).

    ``throttle`` (a ZenodoClient) is called before every request, retries
    included, and a 429 pauses all of its users through ``backoff``. Every
    retry waits with exponential backoff, and an md5 mismatch restarts the
    whole file at most ``max_md5_restarts`` times.

    ``room_check()`` (optional) is called before the first write and after
    every ``room_every`` bytes; when it returns False the download stops, the
    partial file is deleted and the error is "disk floor reached" (not
    retried), so a file never keeps filling a disk that another process
    has used up since the check made before the download.
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 5.0
    last_err = "retries exhausted"
    md5_restarts = 0

    def pause():
        nonlocal delay
        time.sleep(delay)
        delay = min(delay * 2, 300)

    for attempt in range(max_retries):
        if throttle is not None:
            throttle.throttle()
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                if throttle is not None and hasattr(throttle, "observe"):
                    throttle.observe(r)
                if r.status_code == 416 and expected_size and have >= expected_size:
                    pass  # already complete
                elif r.status_code in (200, 206):
                    mode = "ab" if (have and r.status_code == 206) else "wb"
                    since_check = room_every
                    with open(part, mode) as fp:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            if room_check is not None and since_check >= room_every:
                                since_check = 0
                                if not room_check():
                                    fp.close()
                                    part.unlink(missing_ok=True)
                                    return False, "disk floor reached"
                            fp.write(chunk)
                            since_check += len(chunk)
                elif r.status_code in (429, 500, 502, 503, 504):
                    wait = retry_after_seconds(r.headers.get("Retry-After"), delay, r.status_code)
                    last_err = f"HTTP {r.status_code}"
                    if r.status_code == 429 and throttle is not None:
                        throttle.backoff(wait)   # all workers wait, not just this one
                    else:
                        time.sleep(wait)
                    delay = min(delay * 2, 300)
                    continue
                else:
                    return False, f"HTTP {r.status_code}"
        except requests.RequestException as e:
            logger.warning("download %s interrupted (%s), attempt %d", url, type(e).__name__, attempt + 1)
            last_err = f"{type(e).__name__}"
            pause()
            continue

        size = part.stat().st_size if part.exists() else 0
        if expected_size and size < expected_size:
            last_err = "truncated"
            pause()      # resume on the next attempt, but not back to back
            continue
        if expected_size and size > expected_size:
            part.unlink(missing_ok=True)
            last_err = "oversize body"
            pause()
            continue
        if md5 and _md5_of(part) != md5:
            # A corrupt resume is not fixable by resuming again: start over,
            # but never fetch a large file more than a couple of extra times.
            part.unlink(missing_ok=True)
            md5_restarts += 1
            if md5_restarts > max_md5_restarts:
                return False, "md5 mismatch"
            last_err = "md5 mismatch"
            pause()
            continue
        part.replace(dest)
        return True, None
    return False, last_err
