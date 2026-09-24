"""Sample images from Zenodo zip files without downloading the zips.

Zenodo serves the members of a stored zip one at a time:

* ``GET /api/records/<id>/files/<name>.zip/container`` lists the members as
  ``{"entries": [{key, size, compressed_size, mimetype, crc, links}],
  "truncated": bool, "total": int, "directories": [...]}``. The listing is
  capped near 1000 items (files plus directories) with no pagination
  (``page``, ``size``, ``limit``, ``offset`` and ``from`` are ignored), and
  ``truncated`` says when the cap was hit. Since the cap and the flag are
  empirical, a listing is also treated as truncated when ``total`` is above
  the items returned or when it holds CONTAINER_LISTING_CAP items or more.
  RAR and other non-zip files answer HTTP 500.
* ``GET .../container/<member path>`` serves one member, decompressed.

When a listing is truncated (or the container API fails) the zip's central
directory is read with HTTP range requests instead: ``zipfile`` runs over a
seekable file object whose reads become ``Range: bytes=a-b`` requests with a
small block cache, so a zip of any size is listed in a few requests (end
record, then the whole central directory in one read). Members are then
fetched through the container endpoint, or with range reads of the member's
own bytes when the endpoint refuses them (straight to range reads when the
container API already failed for that zip).

Memory stays bounded on a small VM: both fetch paths stream to disk (a
stored or deflated member is one streamed range request inflated on the
fly), the range reader's cache is capped at CACHE_MAX_BYTES, single
prefetches at PREFETCH_MAX_BYTES, and a range body that breaks off mid-read
is retried.

Bytes stay bounded per record: members over a per-member cap
(--remote-max-member-mb, default 200 MB) are never sampled, another member
being drawn in their place, and a per-record byte budget
(--remote-record-budget-gb, default 2 GB) covers direct members and nested
archives together; sampled members past it are not fetched. The budget is
enforced on the bytes actually received (ByteMeter), failed transfers and
retries included, not only on the listed sizes of the fetches that worked:
a transfer that would pass it is stopped, so a record receives at most the
budget plus one read chunk (1 MB) per fetch worker. Listing requests
(container listings, central directories) are not counted.

Every request goes through the survey's ZenodoClient: one shared throttle
(at most 110 requests per minute by default, --rpm) with retries and
exponential backoff on 429 and 5xx.
"""

from __future__ import annotations

import hashlib
import logging
import random
import shutil
import struct
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests

from .constants import IMAGE_KINDS, detect_ext, file_kind, nested_worth
from .zenodo import API, RemoteError, ZenodoClient

logger = logging.getLogger(__name__)

# Default per-member cap of the library functions (a single image this large
# would be inventoried, not decoded, anyway); the survey passes its own,
# smaller --remote-max-member-mb.
MAX_MEMBER_BYTES = 1 << 30
# Nested archives (zip of zips) may be larger; they are streamed to disk in
# both the container and the range path, never held in memory. Each nested
# archive is fetched whole, so the per-archive cap and the per-record nested
# budget (sample_remote_zips) are kept small: at 2 to 7.5 MB/s, 500 MB is one
# to four minutes.
NESTED_MAX_BYTES = 500_000_000
# Stop fetching nested archives after this many in a row added no image.
NESTED_EMPTY_STREAK = 5
# Container listings are capped near this many items (files plus
# directories); a listing this long is re-read from the central directory.
CONTAINER_LISTING_CAP = 1000
# Memory bounds of the range reader: bytes kept in its block cache, and the
# longest single prefetch. Everything longer streams in block-sized reads.
CACHE_MAX_BYTES = 64 << 20
PREFETCH_MAX_BYTES = 8 << 20
_JUNK_PREFIXES = ("__MACOSX/",)


def file_url(record_id: str, key: str) -> str:
    return f"{API}/{record_id}/files/{quote(key, safe='')}/content"


def container_url(record_id: str, key: str) -> str:
    return f"{API}/{record_id}/files/{quote(key, safe='')}/container"


def is_junk_member(name: str) -> bool:
    """macOS resource forks and metadata (__MACOSX/, ._x.png, .DS_Store)."""
    base = PurePosixPath(name).name
    return name.startswith(_JUNK_PREFIXES) or "/__MACOSX/" in name or base.startswith("._") \
        or base == ".DS_Store"


@dataclass
class Member:
    name: str
    size: int
    compressed_size: int = 0
    crc: int | None = None
    kind: str = "other"


@dataclass
class ZipListing:
    key: str                      # Zenodo file key of the zip
    source: str                   # container | range
    members: list[Member] = field(default_factory=list)
    truncated_listing: bool = False
    total_reported: int | None = None
    error: str | None = None
    # The container listing itself failed (HTTP 500 for a zip the API cannot
    # open, or retries used up), as opposed to returning a truncated list:
    # member fetches then go straight to range reads.
    container_failed: bool = False

    @property
    def failed(self) -> bool:
        """Neither listing worked: the zip contributes nothing."""
        return bool(self.error) and not self.members

    @property
    def images(self) -> list[Member]:
        return [m for m in self.members if m.kind in IMAGE_KINDS]

    @property
    def nested(self) -> list[Member]:
        return [m for m in self.members if m.kind in ("archive", "compressed") and nested_worth(m.name, m.kind)]

    @property
    def noext(self) -> list[Member]:
        return [m for m in self.members if m.kind == "noext"]


# ---------------------------------------------------------------------------
# HTTP range file
# ---------------------------------------------------------------------------
class HttpRangeFile:
    """Read-only, seekable file over HTTP range requests (for ``zipfile``).

    Each cache miss is one ``Range`` request of at least ``block`` bytes;
    the last ``max_blocks`` blocks are kept. ``prefetch`` loads a known span
    in one request (a member's local header plus its data).
    """

    def __init__(self, client: ZenodoClient, url: str, size: int, block: int = 256 << 10,
                 max_blocks: int = 8, max_cache_bytes: int = CACHE_MAX_BYTES, max_tries: int = 4):
        self.client, self.url, self.size = client, url, int(size)
        self.block, self.max_blocks, self.max_cache_bytes = block, max_blocks, max_cache_bytes
        self.max_tries = max_tries
        self.pos = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_bytes = 0
        self.n_requests = 0
        self.bytes_fetched = 0

    # file protocol -------------------------------------------------------
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            pos = offset
        elif whence == 1:
            pos = self.pos + offset
        else:
            pos = self.size + offset
        if pos < 0:
            raise OSError("negative seek position")
        self.pos = pos
        return pos

    def close(self):
        self._cache.clear()
        self._cache_bytes = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        out = self._from_cache(self.pos, n)
        if out is None:
            # Use the fetched bytes directly: a block larger than the cache
            # bound (a big central directory) is returned but never kept.
            data = self._fetch(self.pos, max(n, self.block))
            out = data[:n]
            if len(out) < n:
                raise OSError("range read returned fewer bytes than requested")
        self.pos += len(out)
        return out

    def prefetch(self, start: int, length: int):
        """Load ``[start, start + length)`` in one request, at most
        PREFETCH_MAX_BYTES (longer reads go block by block)."""
        length = min(length, PREFETCH_MAX_BYTES, self.size - start)
        if length > 0 and self._from_cache(start, length) is None:
            self._fetch(start, length)

    # internals -----------------------------------------------------------
    def _from_cache(self, start: int, n: int) -> bytes | None:
        for bstart, data in self._cache.items():
            if bstart <= start and start + n <= bstart + len(data):
                self._cache.move_to_end(bstart)
                return data[start - bstart:start - bstart + n]
        return None

    def _fetch(self, start: int, length: int) -> bytes:
        """One range request (retried on a broken or short body), cached
        when it fits the cache bound. Returns the bytes."""
        end = min(self.size, start + length)            # exclusive
        delay = 2.0
        for attempt in range(self.max_tries):
            try:
                data = self._fetch_once(start, end)
                break
            except (requests.RequestException, _ShortRead) as e:
                # client.get retries until a response arrives; a body that
                # breaks off mid-read (ChunkedEncodingError, read timeout) or
                # comes back short is retried here, with the same backoff.
                if attempt + 1 >= self.max_tries:
                    raise OSError(f"range read failed after {self.max_tries} tries: "
                                  f"{type(e).__name__}: {e}"[:300]) from e
                logger.warning("range read %s [%d, %d) failed (%s), attempt %d",
                               self.url, start, end, type(e).__name__, attempt + 1)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        self.n_requests += 1
        self.bytes_fetched += len(data)
        if len(data) <= self.max_cache_bytes:
            self._cache[start] = data
            self._cache_bytes += len(data)
            while self._cache and (len(self._cache) > self.max_blocks or self._cache_bytes > self.max_cache_bytes):
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= len(old)
        return data

    def _fetch_once(self, start: int, end: int) -> bytes:
        # stream=True: the status is checked before any body is read, so a
        # server that ignores Range cannot make us pull the whole file.
        r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end - 1}"}, stream=True)
        try:
            if r.status_code != 206 and not (r.status_code == 200 and start == 0 and end == self.size):
                raise RemoteError(f"server ignored the Range header (HTTP {r.status_code})", r.status_code)
            data = r.content
        finally:
            r.close()
        data = data[: end - start]
        if len(data) < end - start:
            raise _ShortRead(f"short range read: {len(data)} of {end - start} bytes")
        return data


class _ShortRead(OSError):
    """A range response body shorter than asked for (retried)."""


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
def _member(name: str, size: int, csize: int = 0, crc=None) -> Member:
    return Member(name=name, size=int(size or 0), compressed_size=int(csize or 0),
                  crc=int(crc) if isinstance(crc, int) else None, kind=file_kind(name))


def container_listing_short(doc: dict) -> bool:
    """True when a container listing may not hold every member: it says
    ``truncated``, it reports a ``total`` above the items it returned, or it
    reached the observed cap of about CONTAINER_LISTING_CAP items (the cap
    and the flag are empirical, so a full page is never trusted)."""
    if doc.get("truncated"):
        return True
    n_items = len(doc.get("entries") or []) + len(doc.get("directories") or [])
    total = doc.get("total")
    if isinstance(total, int) and not isinstance(total, bool) and total > n_items:
        return True
    return n_items >= CONTAINER_LISTING_CAP


def list_zip(client: ZenodoClient, record_id: str, key: str, size: int) -> ZipListing:
    """Members of a remote zip: container API first, range-read central
    directory when the listing is truncated or the API fails."""
    listing = ZipListing(key=key, source="container")
    try:
        r = client.get(container_url(record_id, key), max_tries=3)
        try:
            doc = r.json()
        finally:
            r.close()
        entries = doc.get("entries") or []
        listing.total_reported = doc.get("total")
        listing.truncated_listing = container_listing_short(doc)
        if not listing.truncated_listing:
            listing.members = [_member(e.get("key") or "", e.get("size"), e.get("compressed_size"), e.get("crc"))
                               for e in entries if e.get("key") and not e["key"].endswith("/")]
            listing.members = [m for m in listing.members if not is_junk_member(m.name)]
            return listing
    except (RemoteError, ValueError) as e:
        listing.error = f"container listing: {e}"
        listing.container_failed = True
    # Truncated or failed: read the central directory with range requests.
    try:
        members = list_zip_by_range(client, record_id, key, size)
    except Exception as e:  # noqa: BLE001 - a bad zip must not stop the record
        msg = f"range listing: {type(e).__name__}: {e}"[:300]
        listing.error = f"{listing.error}; {msg}" if listing.error else msg
        return listing
    listing.source = "range"
    listing.members = [m for m in members if not is_junk_member(m.name)]
    return listing


def list_zip_by_range(client: ZenodoClient, record_id: str, key: str, size: int) -> list[Member]:
    import zipfile
    fp = HttpRangeFile(client, file_url(record_id, key), size)
    with zipfile.ZipFile(fp) as zf:
        return [_member(i.filename, i.file_size, i.compress_size, i.CRC)
                for i in zf.infolist() if not i.is_dir() and not (i.flag_bits & 0x1)]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def allocate(counts: dict[str, int], cap: int) -> dict[str, int]:
    """Split ``cap`` over groups in proportion to their counts (largest
    remainder, ties by key order); never more than a group holds."""
    total = sum(counts.values())
    if total <= cap:
        return dict(counts)
    quotas = {k: cap * v / total for k, v in counts.items()}
    out = {k: min(counts[k], int(q)) for k, q in quotas.items()}
    left = cap - sum(out.values())
    for k in sorted(counts, key=lambda k: (-(quotas[k] - int(quotas[k])), list(counts).index(k))):
        if left <= 0:
            break
        if out[k] < counts[k]:
            out[k] += 1
            left -= 1
    # Groups that were capped by their own size leave room for the others.
    while left > 0:
        grew = False
        for k in counts:
            if left and out[k] < counts[k]:
                out[k] += 1
                left -= 1
                grew = True
        if not grew:
            break
    return out


def n_oversize(listings: list[ZipListing], pool: str, max_bytes: int) -> int:
    """Members of ``pool`` that are never sampled for being over ``max_bytes``."""
    return sum(1 for lst in listings for m in getattr(lst, pool) if m.size > max_bytes)


def n_zero_size(listings: list[ZipListing], pool: str) -> int:
    """Members of ``pool`` listed with size 0 (empty files), never sampled."""
    return sum(1 for lst in listings for m in getattr(lst, pool) if m.size <= 0)


def sample_members(listings: list[ZipListing], cap: int, rng: random.Random,
                   pool: str = "images", max_bytes: int = MAX_MEMBER_BYTES) -> list[tuple[ZipListing, Member]]:
    """Seeded sample of up to ``cap`` members spread over the zips in
    proportion to their candidate counts. ``pool`` is images or noext.
    Members over ``max_bytes`` are not candidates, so another member of the
    same zip is drawn in their place when the zip has one."""
    groups = {}
    for i, lst in enumerate(listings):
        cands = sorted(getattr(lst, pool), key=lambda m: m.name)
        cands = [m for m in cands if 0 < m.size <= max_bytes]
        if cands:
            groups[str(i)] = (lst, cands)
    quota = allocate({k: len(v[1]) for k, v in groups.items()}, cap)
    out = []
    for k, (lst, cands) in groups.items():
        n = quota.get(k, 0)
        picked = cands if n >= len(cands) else rng.sample(cands, n)
        out += [(lst, m) for m in picked]
    return out


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
_SPLIT_FIRST_SUFFIXES = (".7z.001", ".zip.001")


def scratch_suffix(name: str) -> str:
    """File suffix a scratch copy keeps so the walker reads it as the same
    kind (``.png``, ``.tar.gz``, ``.7z.001``; '' for extensionless DICOM)."""
    low = name.lower()
    for s in _SPLIT_FIRST_SUFFIXES:
        if low.endswith(s):
            return s
    ext = detect_ext(name)
    # Keep only a plain, short extension: anything else could carry a path
    # separator or blow the 255-byte name limit.
    if ext and len(ext) <= 16 and all(ch.isalnum() or ch == "." for ch in ext):
        return ext
    return ""


def safe_member_path(root: Path, zip_index: int, name: str, n: int = 0) -> Path:
    """Scratch path for the ``n``-th sampled member of zip ``zip_index``.

    The member's own path is not mirrored: a flat, unique name
    (index + name hash + suffix) cannot exceed the file name limit, cannot
    clash with a file named like one of its directories, and two members
    whose names sanitise alike never share a file. The original name is kept
    in the display string only."""
    digest = hashlib.sha1(name.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return root / str(zip_index) / f"{n:05d}_{digest}{scratch_suffix(name)}"


class RangeZips:
    """Range-read ZipFile objects, opened once per zip and shared by the
    member fetches of a record (the central directory is read once).
    ``zipfile`` over one file object is not thread-safe, hence the lock."""

    def __init__(self, client: ZenodoClient, record_id: str):
        self.client, self.record_id = client, record_id
        self._open: dict[str, tuple[HttpRangeFile, object]] = {}
        self.lock = threading.Lock()

    def get(self, key: str, size: int):
        import zipfile
        if key not in self._open:
            fp = HttpRangeFile(self.client, file_url(self.record_id, key), size, max_blocks=4)
            self._open[key] = (fp, zipfile.ZipFile(fp))
        return self._open[key]

    def close(self):
        for fp, zf in self._open.values():
            zf.close()
            fp.close()
        self._open.clear()


class BudgetExceededError(RemoteError):
    """The record byte budget was reached during a transfer."""


class ByteMeter:
    """Thread-safe count of the member and nested archive bytes received for
    a record, failed transfers and retries included. ``add`` raises
    BudgetExceededError once the count passes ``limit`` (None: no limit), which
    stops the transfer in progress."""

    def __init__(self, limit: int | None = None):
        self.limit = limit
        self.n = 0
        self._lock = threading.Lock()

    def add(self, k: int):
        with self._lock:
            self.n += k
            over = self.limit is not None and self.n > self.limit
        if over:
            raise BudgetExceededError("record byte budget reached during the transfer")

    def fits(self, k: int) -> bool:
        with self._lock:
            return self.limit is None or self.n + k <= self.limit


def _metered(chunks, meter: ByteMeter | None):
    """Pass ``chunks`` through, counting their bytes in ``meter`` first."""
    for chunk in chunks:
        if meter is not None and chunk:
            meter.add(len(chunk))
        yield chunk


def fetch_member(client: ZenodoClient, record_id: str, lst: ZipListing, m: Member, dest: Path,
                 zip_size: int = 0, range_zips: RangeZips | None = None,
                 max_bytes: int = MAX_MEMBER_BYTES, meter: ByteMeter | None = None) -> tuple[bool, str | None]:
    """Fetch one member into ``dest``: container endpoint, else range read.
    Size and CRC (when known) are checked. Both paths stream to disk, so
    memory stays bounded whatever the member size; members over
    ``max_bytes`` are refused.

    When the container API already failed for the whole zip (its listing
    errored), the container endpoint is not tried for its members: that
    would cost a full retry cycle per member before the range fallback.

    ``meter`` counts every byte received (both paths, failed attempts
    included); BudgetExceededError from it propagates to the caller."""
    if m.size and m.size > max_bytes:
        return False, f"member of {m.size} bytes is over the fetch cap of {max_bytes}"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:          # a scratch path problem fails this member only
        return False, f"scratch dir: {type(e).__name__}: {e}"[:300]
    tmp = dest.with_name(dest.name + ".part")
    can_range = bool(zip_size) and range_zips is not None
    if lst.container_failed and can_range:
        err = "container: skipped (container API failed for this zip)"
    else:
        try:
            url = f"{container_url(record_id, lst.key)}/{quote(m.name, safe='/')}"
            r = client.get(url, stream=True, max_tries=4)
            try:
                n, crc = _stream_to(tmp, _metered(r.iter_content(chunk_size=1 << 20), meter), max_bytes)
            finally:
                r.close()
            _check(n, crc, m.size, m.crc)
            tmp.replace(dest)
            return True, None
        except BudgetExceededError:
            tmp.unlink(missing_ok=True)
            raise
        except (RemoteError, OSError, requests.RequestException) as e:
            err = f"container: {e}"
            tmp.unlink(missing_ok=True)
    if not can_range:
        return False, err
    try:
        _range_member(client, record_id, lst, m, tmp, zip_size, range_zips, max_bytes, meter=meter)
        tmp.replace(dest)
        return True, None
    except BudgetExceededError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return False, f"{err}; range: {type(e).__name__}: {e}"[:300]


def _stream_to(path: Path, chunks, max_bytes: int, decompress=None) -> tuple[int, int]:
    """Write ``chunks`` (optionally through ``decompress``) to ``path``;
    returns (bytes written, CRC-32). Raises past ``max_bytes``."""
    crc = n = 0
    step = 1 << 20          # at most 1 MB of inflated output per call (no zip-bomb blow-up)
    with open(path, "wb") as out:
        def write(b: bytes):
            nonlocal crc, n
            if b:
                out.write(b)
                crc = zlib.crc32(b, crc)
                n += len(b)
                if n > max_bytes:
                    raise RemoteError("member larger than the fetch cap")

        for chunk in chunks:
            if not chunk:
                continue
            if decompress is None:
                write(chunk)
                continue
            write(decompress.decompress(chunk, step))
            while decompress.unconsumed_tail:
                write(decompress.decompress(decompress.unconsumed_tail, step))
        if decompress is not None:
            write(decompress.flush())
    return n, crc


def _check(n: int, crc: int, size: int, want_crc: int | None):
    if size and n != size:
        raise RemoteError(f"size mismatch {n} != {size}")
    if want_crc is not None and crc != want_crc:
        raise RemoteError("crc mismatch")


def _range_member(client: ZenodoClient, record_id: str, lst: ZipListing, m: Member, tmp: Path,
                  zip_size: int, range_zips: RangeZips, max_bytes: int, max_tries: int = 3,
                  meter: ByteMeter | None = None):
    """Read one member with range requests, into ``tmp``.

    Stored and deflated members (nearly all zips) are streamed: the local
    header is read through the shared ZipFile, then the member's compressed
    bytes come in one streamed range request and are inflated on the fly, so
    memory stays flat for any member size. Other methods (bzip2, lzma,
    deflate64) go through ``zipfile`` with block-sized reads."""
    with range_zips.lock:
        fp, zf = range_zips.get(lst.key, zip_size)
        info = zf.getinfo(m.name)
        if info.flag_bits & 0x1:
            raise RemoteError("encrypted member")
        if info.file_size > max_bytes:
            raise RemoteError("member larger than the fetch cap")
        if info.compress_type not in (0, 8):
            if meter is not None:     # counted up front: these reads bypass the streamed path
                meter.add(30 + len(info.filename.encode("utf-8")) + 1024 + info.compress_size)
            fp.prefetch(info.header_offset, 30 + len(info.filename.encode("utf-8")) + 1024 + info.compress_size)
            with zf.open(info) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            return
        fp.seek(info.header_offset)
        head = fp.read(30)
    if len(head) != 30 or head[:4] != b"PK\x03\x04":
        raise RemoteError("bad local file header")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    end = start + info.compress_size                       # exclusive
    if end > zip_size:
        raise RemoteError("member data runs past the end of the zip")
    url = file_url(record_id, lst.key)
    delay = 2.0
    for attempt in range(max_tries):
        try:
            if info.compress_size == 0:
                n, crc = _stream_to(tmp, [], max_bytes)
            else:
                r = client.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, stream=True)
                try:
                    if r.status_code != 206:
                        raise RemoteError(f"server ignored the Range header (HTTP {r.status_code})",
                                          r.status_code)
                    dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
                    n, crc = _stream_to(tmp, _metered(r.iter_content(chunk_size=1 << 20), meter),
                                        max_bytes, dec)
                finally:
                    r.close()
            _check(n, crc, info.file_size, info.CRC)
            return
        except requests.RequestException as e:
            if attempt + 1 >= max_tries:
                raise
            logger.warning("range member %s!/%s failed (%s), attempt %d", lst.key, m.name,
                           type(e).__name__, attempt + 1)
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------
@dataclass
class RemoteSample:
    listings: list[ZipListing] = field(default_factory=list)
    pool: str = ""                                   # images | noext | ""
    n_images_listed: int = 0                         # image members over all listings
    n_noext_listed: int = 0
    n_nested_listed: int = 0
    n_nested_fetched: int = 0
    n_nested_over_cap: int = 0                       # nested archives over the per-archive cap
    n_nested_over_budget: int = 0                    # nested archives skipped: record budget used up
    nested_bytes: int = 0                            # bytes of the nested archives fetched
    nested_stop: str = ""                            # why nested fetching stopped early, if it did
    nested_images: int = 0                          # image entries found in fetched nested archives
    sampled: list[tuple[ZipListing, Member]] = field(default_factory=list)
    fetched: list[tuple[Path, str]] = field(default_factory=list)       # (scratch path, display)
    failures: list[str] = field(default_factory=list)
    unfetched_image_paths: list[str] = field(default_factory=list)      # displays, for laterality
    other_member_names: list[str] = field(default_factory=list)
    member_kind_counts: dict = field(default_factory=dict)              # kinds of members not fetched
    unfetched_image_ext_counts: dict = field(default_factory=dict)
    n_requests: int = 0
    nested_done: set = field(default_factory=set)                       # (listing index, member name)
    n_members_skipped_oversize: int = 0              # pool members over the per-member cap, never sampled
    n_members_zero_size: int = 0                     # pool members listed with size 0, never sampled
    n_members_over_budget: int = 0                   # sampled members dropped: record byte budget reached
    members_bytes: int = 0                           # listed bytes of the direct members fetched
    budget_stop: str = ""                            # why direct member fetching stopped early, if it did
    bytes_received: int = 0                          # member and nested bytes received, failures included

    @property
    def bytes_fetched(self) -> int:
        """Listed sizes of the members and nested archives fetched
        successfully (listings not included). ``bytes_received`` is the
        measured count, failed transfers and retries included."""
        return self.members_bytes + self.nested_bytes

    @property
    def n_listing_failures(self) -> int:
        """Zips whose container and range listings both failed."""
        return sum(1 for lst in self.listings if lst.failed)

    def detail(self) -> str:
        parts = []
        for lst in self.listings:
            s = f"{lst.key}: {len(lst.images)} images / {len(lst.members)} files ({lst.source}"
            if lst.truncated_listing:
                s += (f", container listing truncated (total {lst.total_reported})"
                      if lst.total_reported is not None else ", container listing possibly truncated")
            if lst.error:
                s += f", {lst.error}"
            parts.append(s + ")")
        return "; ".join(parts)[:2000]


def sample_remote_zips(
    client: ZenodoClient, record_id: str, zips: list[dict], root: Path, cap: int, rng: random.Random,
    workers: int = 3, nested_max: int = 20, nested_budget_bytes: int = 2_000_000_000,
    room_for=None, on_nested=None, nested_max_bytes: int = NESTED_MAX_BYTES,
    nested_empty_streak: int = NESTED_EMPTY_STREAK, max_member_bytes: int = MAX_MEMBER_BYTES,
    record_budget_bytes: int | None = None,
) -> RemoteSample:
    """List every zip, sample up to ``cap`` image members over all of them
    (proportional to their image counts, seeded) and fetch only those into
    ``root``. ``zips`` are Zenodo file dicts {key, size}.

    Byte bounds: members over ``max_member_bytes`` are never sampled (another
    member is drawn instead where the zip has one; counted in
    ``n_members_skipped_oversize``). ``record_budget_bytes`` caps the bytes
    received for the record, nested archives and direct members together,
    failed transfers and retries included (a ByteMeter; ``bytes_received``):
    nested archives that no longer fit are skipped, the sampled direct
    members are taken in seeded random order until the next one would pass
    the budget, and during the fetches a member is not started when it no
    longer fits the bytes left, and a transfer is stopped when it would pass
    the budget. Those members are not fetched (``n_members_over_budget``,
    ``budget_stop``) and count as listed, unfetched images.

    Zips whose image members sit in nested archives (zip of zips) get their
    nested archives fetched whole instead, in seeded random order, until
    ``nested_max`` archives are opened, ``nested_budget_bytes`` (a budget of
    its own, not the record's download budget) is used,
    ``nested_empty_streak`` archives in a row added no image, or
    ``on_nested(path, display) -> images added`` has found ``cap`` images.
    Archives over ``nested_max_bytes`` are never fetched. Records whose zips
    hold only extensionless files (often DICOM) get a sample of those,
    sniffed later. ``room_for(nbytes)`` guards the disk.
    """
    from concurrent.futures import ThreadPoolExecutor

    out = RemoteSample()
    meter = ByteMeter(record_budget_bytes)
    start_requests = client.n_requests
    sizes = {z["key"]: int(z.get("size") or 0) for z in zips}
    for z in zips:
        out.listings.append(list_zip(client, record_id, z["key"], sizes[z["key"]]))
    out.n_images_listed = sum(len(lst.images) for lst in out.listings)
    out.n_noext_listed = sum(len(lst.noext) for lst in out.listings)
    out.n_nested_listed = sum(len(lst.nested) for lst in out.listings)
    range_zips = RangeZips(client, record_id)
    try:
        # ---- nested archives (zip of zips) when direct images are scarce
        if out.n_nested_listed and out.n_images_listed < cap and on_nested is not None:
            nested = [(i, lst, m) for i, lst in enumerate(out.listings) for m in lst.nested]
            nested.sort(key=lambda t: (t[1].key, t[2].name))
            rng.shuffle(nested)
            budget = nested_budget_bytes
            empty = 0
            for k, (i, lst, m) in enumerate(nested):
                if out.n_nested_fetched >= nested_max:
                    out.nested_stop = f"--remote-nested-max {nested_max} archives opened"
                    break
                if out.nested_images >= cap:
                    break
                if nested_empty_streak and empty >= nested_empty_streak:
                    out.nested_stop = f"{empty} nested archives in a row held no image"
                    break
                if m.size > nested_max_bytes:
                    out.n_nested_over_cap += 1        # a size limit, not a transient failure
                    continue
                if m.size > budget:
                    out.n_nested_over_budget += 1
                    out.nested_stop = out.nested_stop or "nested byte budget used up"
                    continue
                if not meter.fits(m.size):
                    out.n_nested_over_budget += 1
                    out.nested_stop = out.nested_stop or "record byte budget used up"
                    continue
                if room_for is not None and not room_for(m.size):
                    continue
                dest = safe_member_path(root / "nested", i, m.name, k)
                try:
                    ok, err = fetch_member(client, record_id, lst, m, dest, sizes[lst.key], range_zips,
                                           max_bytes=nested_max_bytes, meter=meter)
                except BudgetExceededError:
                    out.n_nested_over_budget += 1
                    out.nested_stop = out.nested_stop or "record byte budget used up"
                    continue
                if not ok:
                    out.failures.append(f"{lst.key}!/{m.name}: {err}"[:300])
                    continue
                budget -= m.size
                out.nested_bytes += m.size
                out.n_nested_fetched += 1
                out.nested_done.add((i, m.name))
                added = int(on_nested(dest, f"{lst.key}!/{m.name}") or 0)
                out.nested_images += added
                empty = 0 if added else empty + 1

        # ---- direct members
        if out.n_images_listed:
            out.pool = "images"
        elif out.n_noext_listed and not out.n_nested_fetched:
            out.pool = "noext"
        if out.pool:
            out.n_members_skipped_oversize = n_oversize(out.listings, out.pool, max_member_bytes)
            out.n_members_zero_size = n_zero_size(out.listings, out.pool)
            out.sampled = sample_members(out.listings, cap, rng, out.pool, max_bytes=max_member_bytes)
        if record_budget_bytes is not None and out.sampled:
            # Seeded random order, so a budget cut keeps a random subsample
            # rather than the first zips; fetch order stays as sampled.
            left = record_budget_bytes - meter.n          # measured: failed nested fetches count too
            order = list(range(len(out.sampled)))
            rng.shuffle(order)
            keep: set[int] = set()
            for j in order:
                size = out.sampled[j][1].size
                if size > left:
                    out.budget_stop = (f"record byte budget reached after {len(keep)} of "
                                       f"{len(out.sampled)} sampled members")
                    break
                keep.add(j)
                left -= size
            out.n_members_over_budget = len(out.sampled) - len(keep)
            out.sampled = [item for j, item in enumerate(out.sampled) if j in keep]
        picked = {(id(lst), m.name) for lst, m in out.sampled}
        index = {id(lst): i for i, lst in enumerate(out.listings)}
        for i, lst in enumerate(out.listings):
            for m in lst.members:
                if (id(lst), m.name) in picked or (i, m.name) in out.nested_done:
                    continue
                disp = f"{lst.key}!/{m.name}"
                out.member_kind_counts[m.kind] = out.member_kind_counts.get(m.kind, 0) + 1
                if m.kind in IMAGE_KINDS:
                    if len(out.unfetched_image_paths) < 50_000:
                        out.unfetched_image_paths.append(disp)
                    ext = detect_ext(m.name) or ".dcm"
                    out.unfetched_image_ext_counts[ext] = out.unfetched_image_ext_counts.get(ext, 0) + 1
                elif len(out.other_member_names) < 5000:
                    out.other_member_names.append(disp)

        over_budget = object()      # marker: member not fetched, record byte budget reached

        def one(numbered):
            n, item = numbered
            lst, m = item
            dest = safe_member_path(root / "members", index[id(lst)], m.name, n)
            try:
                if not meter.fits(m.size):
                    return item, dest, over_budget
                if room_for is not None and not room_for(m.size):
                    return item, dest, "skipped (disk floor)"
                ok, err = fetch_member(client, record_id, lst, m, dest, sizes[lst.key], range_zips,
                                       max_bytes=max_member_bytes, meter=meter)
            except BudgetExceededError:
                return item, dest, over_budget
            except OSError as e:      # one bad member never aborts the record
                ok, err = False, f"{type(e).__name__}: {e}"
            return item, dest, (None if ok else err)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for (lst, m), dest, err in ex.map(one, enumerate(out.sampled)):
                if err is over_budget:
                    # Failed transfers or retries used the budget up: listed,
                    # unfetched, like the members cut before the fetches.
                    out.n_members_over_budget += 1
                    out.budget_stop = out.budget_stop or "record byte budget reached during the fetches"
                    out.member_kind_counts[m.kind] = out.member_kind_counts.get(m.kind, 0) + 1
                    if m.kind in IMAGE_KINDS:
                        if len(out.unfetched_image_paths) < 50_000:
                            out.unfetched_image_paths.append(f"{lst.key}!/{m.name}")
                        ext = detect_ext(m.name) or ".dcm"
                        out.unfetched_image_ext_counts[ext] = out.unfetched_image_ext_counts.get(ext, 0) + 1
                elif err:
                    out.failures.append(f"{lst.key}!/{m.name}: {err}"[:300])
                else:
                    out.fetched.append((dest, f"{lst.key}!/{m.name}"))
                    out.members_bytes += m.size
    finally:
        range_zips.close()
        out.n_requests = client.n_requests - start_requests
        out.bytes_received = meter.n
    return out
