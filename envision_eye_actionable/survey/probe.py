"""Look inside a remote archive before downloading it whole.

Remote zips are sampled member by member (remote.py). Every other archive
(tar, tar.gz, 7z, rar, single-file .gz) used to be downloaded whole before
the walker could list it, even when it held only tables or code. The probe
lists such an archive with a few HTTP range requests first and the survey
downloads it only when the listing shows a member the walker would use:

* a pixel-bearing file of any kind the survey converts
  (constants.IMAGE_KINDS: images, SVG, DICOM, volumes and volume headers,
  vendor OCT, video, numeric arrays, microscopy),
* a nested archive, or a single-file compressed member worth reading
  (constants.nested_worth; split archive parts too),
* an extensionless member that may be DICOM or an image. The walker sniffs
  those in zip and tar archives only (7z and rar members are listed without
  reading them), so they count for zip and tar, and there the probe sniffs
  the magic bytes itself (constants.sniff_kind) when the bytes are cheap to
  read. Members under 8 bytes or of 64 MB and more are never sniffed by the
  walker, so they never count.
  Common text names (README, LICENSE, Makefile, ...) never count.

macOS junk (``__MACOSX/``, ``._x.png``, ``.DS_Store``) never counts.

Outcomes: ``images`` (a member above was listed: download as before),
``no_images`` (the whole listing was read and holds none: not downloaded,
its listing catalogued), ``unknown`` (the probe could not read the whole
listing within its limits, or failed: download as before) and
``not_probed`` (small files, split sets, probing off).

Methods, by format:

* ``.tar``: ``tarfile`` over a range-read file object hops from header to
  header (one range request per jump over a large member; small members
  are covered by read blocks that grow while the reads stay sequential).
* ``.tar.gz``, ``.tgz``, ``.tar.bz2``, ``.tar.xz``, ``.tar.zst``: one streamed range
  request of the first ``stream_bytes`` (default 64 MB), decompressed and
  listed on the fly. A compressed tar has no index, so a listing that stops
  at the limit without an image member is unknown.
* ``.7z``: ``py7zr`` over the range-read file object reads the signature
  header and the end header (encoded headers included; they sit at the end
  too), a few requests whatever the archive size. An encrypted header is
  unknown.
* ``.rar`` (RAR 4 and 5): the block headers are walked with range reads,
  jumping over member data, and the walk stops at the first member that
  counts (``_walk_rar``); a layout the walker does not parse goes to
  ``rarfile``, which reads every header. Encrypted headers are unknown. The
  first part of a multi-volume set can show images but never proves their
  absence.
* ``.zip`` (only when a zip is downloaded whole: --no-remote-zip): the
  central directory by range reads.
* single-file ``.gz``/``.bz2``/``.xz``: decided by the inner name (x.tif.gz
  is an image) with no request; an extensionless inner name gets its first
  bytes decompressed and sniffed for DICM, as the walker would.

Each archive probe stops at ``stream_bytes`` bytes and ``max_requests``
requests; a record stops probing at RECORD_PROBE_BYTES bytes and
RECORD_PROBE_REQUESTS requests (the rest is not probed). Archives of at
most MIN_PROBE_BYTES are not probed: downloading them costs about the same.
"""

from __future__ import annotations

import bz2
import io
import logging
import lzma
import re
import struct
import tarfile
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import requests

from .constants import (IMAGE_KINDS, RAR_EXTS, SEVEN_ZIP_EXTS, SNIFF_BYTES, TAR_EXTS, ZIP_EXTS, detect_ext,
                        file_kind, is_fragment, nested_worth, sniff_kind, split_first, split_stem)
from .remote import HttpRangeFile, file_url, is_junk_member
from .zenodo import RemoteError, ZenodoClient

logger = logging.getLogger(__name__)

STREAM_BYTES = 64 << 20          # per archive: bytes read at most (--archive-probe-mb)
MAX_REQUESTS = 50                # per archive: range requests at most (--archive-probe-max-requests)
RECORD_PROBE_BYTES = 1 << 30     # per record: probe bytes, all archives together
RECORD_PROBE_REQUESTS = 400      # per record: probe requests, all archives together
MIN_PROBE_BYTES = 1 << 20        # archives this small are downloaded without a probe
SNIFF_REQUESTS = 5               # extra range reads per archive to sniff extensionless members
NAMES_KEPT = 2000                # member names kept per negative archive (catalogue, manufacturer hints)
DICM_OFFSET = 128
SNIFF_MIN, SNIFF_MAX = 8, 64 << 20       # the walker sniffs extensionless members in this size range

# Extensionless names that are never DICOM.
_TEXT_NAMES = {
    "readme", "license", "licence", "copying", "copyright", "notice", "authors", "contributors",
    "changelog", "changes", "history", "news", "install", "makefile", "dockerfile", "version",
    "manifest", "todo", "citation", "gemfile", "procfile", "vagrantfile", "jenkinsfile",
}

_TAR_STREAM_EXTS = TAR_EXTS - {".tar"}
def _zstd_obj():
    import zstandard
    return zstandard.ZstdDecompressor().decompressobj()


_SINGLE_OPENERS = {".gz": lambda: zlib.decompressobj(16 + zlib.MAX_WBITS), ".bz2": bz2.BZ2Decompressor,
                   ".xz": lzma.LZMADecompressor, ".zst": _zstd_obj}


class _ProbeLimit(Exception):
    """The probe reached its byte or request limit."""


@dataclass
class ProbeResult:
    key: str
    size: int
    fmt: str = ""                  # tar | tar_stream | 7z | rar | zip | single | split | other
    outcome: str = "not_probed"    # images | no_images | unknown | not_probed
    method: str = ""
    complete: bool = False         # the whole listing was read
    n_members: int = 0             # regular-file members listed
    n_images: int = 0              # image, DICOM or volume members (plus sniffed DICOM)
    n_nested: int = 0              # nested archives or compressed members worth reading
    n_noext: int = 0               # extensionless members counted as possible DICOM
    bytes_used: int = 0
    n_requests: int = 0
    hit: str = ""                  # the member that decided images
    note: str = ""
    kind_counts: Counter = field(default_factory=Counter)
    ext_counts: Counter = field(default_factory=Counter)      # every listed member, by extension
    names: list[str] = field(default_factory=list)

    @property
    def positive(self) -> bool:
        return self.n_images + self.n_nested + self.n_noext > 0

    def summary(self) -> dict:
        """Compact per-archive record for the results row."""
        return {"key": self.key, "size": self.size, "format": self.fmt, "outcome": self.outcome,
                "method": self.method, "complete": self.complete, "n_members": self.n_members,
                "n_images": self.n_images, "n_nested": self.n_nested, "n_noext": self.n_noext,
                "bytes": self.bytes_used, "requests": self.n_requests, "hit": self.hit[:200],
                "note": self.note[:300]}


def probe_format(key: str) -> str:
    """Probe method family of a Zenodo file name ('' when not an archive)."""
    if split_first(key) or is_fragment(key):
        return "split"
    kind = file_kind(key)
    ext = detect_ext(key)
    if kind == "archive":
        if ext == ".tar":
            return "tar"
        if ext in _TAR_STREAM_EXTS:
            return "tar_stream"
        if ext in SEVEN_ZIP_EXTS:
            return "7z"
        if ext in RAR_EXTS:
            return "rar"
        if ext in ZIP_EXTS:
            return "zip"
        return "other"
    if kind == "compressed":
        return "single"
    return ""


def member_signal(name: str, size: int, noext_readable: bool, head: bytes | None = None) -> str:
    """What one listed member says: image | nested | noext | '' (nothing).

    ``noext_readable``: the walker sniffs extensionless members of this
    archive format (zip, tar). ``head``: the member's first bytes when the
    probe could read them (decides an extensionless member)."""
    if is_junk_member(name):
        return ""
    kind = file_kind(name)
    if kind in IMAGE_KINDS:
        return "image"
    if kind == "fragment" or (kind in ("archive", "compressed") and nested_worth(name, kind)):
        return "nested"
    if kind == "noext" and noext_readable:
        base = PurePosixPath(name).name.lower()
        if base in _TEXT_NAMES or not (SNIFF_MIN <= size < SNIFF_MAX):
            return ""
        if head is not None:
            return "image" if sniff_kind(head) is not None else ""
        return "noext"
    return ""


class _Listing:
    """Accumulates members into a ProbeResult; ``add`` returns True on the
    first positive member (the probe may stop there)."""

    def __init__(self, res: ProbeResult, noext_readable: bool):
        self.res, self.noext_readable = res, noext_readable

    def add(self, name: str, size: int, head: bytes | None = None) -> bool:
        res = self.res
        name = name.replace("\\", "/")
        res.n_members += 1
        res.kind_counts[file_kind(name)] += 1
        res.ext_counts[detect_ext(name) or "(none)"] += 1
        sig = member_signal(name, size, self.noext_readable, head)
        if sig == "image":
            res.n_images += 1
        elif sig == "nested":
            res.n_nested += 1
        elif sig == "noext":
            res.n_noext += 1
        elif len(res.names) < NAMES_KEPT:
            res.names.append(name)
        if sig and not res.hit:
            res.hit = name
        return bool(sig)


class ProbeRangeFile(HttpRangeFile):
    """HttpRangeFile with per-probe limits and read blocks that grow while
    reads stay sequential (dense small members) and drop back after a jump
    (over a large member), so a header walk costs few bytes and requests."""

    def __init__(self, client: ZenodoClient, url: str, size: int, max_bytes: int, max_requests: int,
                 base_block: int = 64 << 10, max_block: int = 8 << 20):
        super().__init__(client, url, size, block=base_block, max_blocks=16,
                         max_cache_bytes=max(32 << 20, 2 * max_block), max_tries=3)
        self.base_block, self.max_block = base_block, max_block
        self.max_bytes, self.max_requests = max_bytes, max_requests
        self._last_end = 0

    def _fetch(self, start: int, length: int) -> bytes:
        need = min(length, self.size - start)
        if start <= self._last_end + self.block:
            self.block = min(self.block * 2, self.max_block)      # sequential: grow
        else:
            self.block = self.base_block                          # a jump: start small again
        if self.n_requests >= self.max_requests:
            raise _ProbeLimit(f"request limit ({self.max_requests}) reached")
        allowed = self.max_bytes - self.bytes_fetched
        if need > allowed:
            raise _ProbeLimit(f"byte limit ({self.max_bytes / 2**20:g} MB) reached")
        length = min(max(need, self.block), self.size - start, allowed)
        data = super()._fetch(start, length)
        self._last_end = start + len(data)
        return data

    def cached(self, start: int, n: int) -> bytes | None:
        return self._from_cache(start, n)


class _RawIO(io.RawIOBase):
    """io.RawIOBase view of a range file (py7zr and rarfile want a real
    file object)."""

    def __init__(self, fp: HttpRangeFile):
        super().__init__()
        self.fp = fp

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        return self.fp.seek(offset, whence)

    def tell(self):
        return self.fp.tell()

    def read(self, n=-1):
        return self.fp.read(n)

    def readinto(self, b):
        data = self.fp.read(len(b))
        b[:len(data)] = data
        return len(data)

    def readall(self):
        return self.fp.read(-1)


class _LimitedStream:
    """Read-only stream over a response body that raises _ProbeLimit once
    ``limit`` bytes are used and more is asked for while the file goes on."""

    def __init__(self, resp, limit: int, total: int):
        self._it = resp.iter_content(chunk_size=256 << 10)
        self.limit, self.total = limit, total
        self.buf = b""
        self.n = 0              # bytes taken from the network

    def read(self, k: int = -1) -> bytes:
        while k < 0 or len(self.buf) < k:
            try:
                chunk = next(self._it)
            except StopIteration:
                if self.n < self.total and self.n >= self.limit:
                    if self.buf:
                        break
                    raise _ProbeLimit(f"byte limit ({self.limit >> 20} MB) reached") from None
                break
            self.n += len(chunk)
            self.buf += chunk
        if k < 0:
            k = len(self.buf)
        out, self.buf = self.buf[:k], self.buf[k:]
        return out


# ---------------------------------------------------------------------------
# Format probes
# ---------------------------------------------------------------------------
def _probe_tar(client, url, res: ProbeResult, stream_bytes: int, max_requests: int):
    """Uncompressed tar: tarfile hops from header to header over range reads."""
    res.method = "tar_headers"
    fp = ProbeRangeFile(client, url, res.size, stream_bytes, max_requests)
    lst = _Listing(res, noext_readable=True)
    sniffs = 0
    try:
        with tarfile.open(fileobj=fp, mode="r:") as tf:
            while True:
                m = tf.next()
                if m is None:
                    res.complete = True
                    break
                if not m.isfile():
                    continue
                head = None
                if file_kind(m.name) == "noext" and SNIFF_MIN <= m.size < SNIFF_MAX:
                    n_head = min(SNIFF_BYTES, m.size)
                    head = fp.cached(m.offset_data, n_head)
                    if head is None and sniffs < SNIFF_REQUESTS:
                        sniffs += 1
                        pos = fp.tell()
                        fp.seek(m.offset_data)
                        head = fp.read(n_head)
                        fp.seek(pos)
                if lst.add(m.name, m.size, head):
                    break
    finally:
        res.bytes_used += fp.bytes_fetched
        res.n_requests += fp.n_requests


def _probe_tar_stream(client, url, res: ProbeResult, stream_bytes: int):
    """Compressed tar: the first ``stream_bytes`` streamed and listed."""
    res.method = "tar_stream"
    limit = min(stream_bytes, res.size)
    r = client.get(url, headers={"Range": f"bytes=0-{limit - 1}"}, stream=True, max_tries=3)
    res.n_requests += 1
    stream = _LimitedStream(r, limit, res.size)
    lst = _Listing(res, noext_readable=True)
    try:
        if r.status_code != 206 and not (r.status_code == 200 and limit == res.size):
            raise RemoteError(f"server ignored the Range header (HTTP {r.status_code})", r.status_code)
        if res.key.lower().endswith((".tar.zst", ".tzst")):
            import zstandard
            opened = tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(stream), mode="r|")
        else:
            opened = tarfile.open(fileobj=stream, mode="r|*")
        with opened as tf:
            for m in tf:
                if not m.isfile():
                    continue
                head = None
                if file_kind(m.name) == "noext" and SNIFF_MIN <= m.size < SNIFF_MAX:
                    src = tf.extractfile(m)
                    head = src.read(SNIFF_BYTES) if src is not None else None
                if lst.add(m.name, m.size, head):
                    break
            else:
                res.complete = True
    finally:
        res.bytes_used += stream.n
        r.close()


def _probe_7z(client, url, res: ProbeResult, stream_bytes: int, max_requests: int):
    """7z: signature header, then the end header (py7zr, range reads)."""
    import py7zr
    res.method = "7z_header"
    fp = ProbeRangeFile(client, url, res.size, stream_bytes, max_requests, base_block=256 << 10)
    lst = _Listing(res, noext_readable=False)
    try:
        try:
            with py7zr.SevenZipFile(_RawIO(fp), mode="r") as z:
                infos = z.list()
        except py7zr.exceptions.PasswordRequired as e:
            raise _Unknown(f"encrypted header: {e}") from e
        # The whole index is in memory: count every member.
        for info in infos:
            if not info.is_directory:
                lst.add(info.filename, int(info.uncompressed or 0))
        res.complete = True
    finally:
        res.bytes_used += fp.bytes_fetched
        res.n_requests += fp.n_requests


def _probe_rar(client, url, res: ProbeResult, stream_bytes: int, max_requests: int, volume: bool):
    """RAR 4/5: block headers walked with range reads, stopping at the
    first member that counts (the survey's own walker, _walk_rar); a
    layout it does not parse goes to rarfile, which reads every header."""
    res.method = "rar_headers"
    fp = ProbeRangeFile(client, url, res.size, stream_bytes, max_requests)
    lst = _Listing(res, noext_readable=False)
    try:
        seen: list[tuple[str, int]] = []

        def on_member(name: str, size: int) -> bool:
            seen.append((name, size))
            return bool(member_signal(name, size, False))
        try:
            complete, is_volume = _walk_rar(fp, res.size, on_member)
        except _RarFormat as e:
            logger.debug("rar walker gave up (%s); rarfile walks it", e)
            _probe_rar_rarfile(fp, res, lst, volume)
            return
        for name, size in seen:
            lst.add(name, size)
        # A volume set continues in the next part: never proven empty.
        res.complete = complete and not volume and not is_volume
        if not res.complete and not res.positive and (volume or is_volume):
            res.note = "multi-volume set: the first part lists only its own members"
    finally:
        res.bytes_used += fp.bytes_fetched
        res.n_requests += fp.n_requests


def _probe_rar_rarfile(fp, res: ProbeResult, lst: "_Listing", volume: bool):
    import rarfile
    res.method = "rar_headers (rarfile)"
    rf = rarfile.RarFile(_RawIO(fp), part_only=True)
    infos = rf.infolist()
    if not infos and rf.needs_password():
        raise _Unknown("encrypted headers")
    for info in infos:                  # rarfile has walked every header already
        if not info.is_dir():
            lst.add(info.filename, int(info.file_size or 0))
    res.complete = not volume and not _rar_is_volume(rf)
    if not res.complete and not res.positive:
        res.note = "multi-volume set: the first part lists only its own members"


class _RarFormat(Exception):
    """The survey's own RAR header walker cannot read this layout (rarfile
    then walks it)."""


def _vint(buf: bytes, i: int) -> tuple[int, int]:
    """RAR 5 variable-length integer at ``buf[i:]``: (value, next index)."""
    v = shift = 0
    while True:
        if i >= len(buf) or shift > 63:
            raise _RarFormat("truncated vint")
        b = buf[i]
        v |= (b & 0x7F) << shift
        i += 1
        if not b & 0x80:
            return v, i
        shift += 7


def _walk_rar(fp, size: int, on_member) -> tuple[bool, bool]:
    """Walk RAR 4 or RAR 5 block headers with seeks over member data,
    calling ``on_member(name, unpacked_size)`` for each file; stops early
    when it returns True. Returns (whole listing read, archive is a volume
    of a multi-part set). Raises _Unknown for encrypted headers and
    _RarFormat for anything it does not parse (rarfile takes over)."""
    fp.seek(0)
    sig = fp.read(8)
    volume = False
    if sig[:7] == b"Rar!\x1a\x07\x00":
        pos = 7
        while pos + 7 <= size:
            fp.seek(pos)
            head = fp.read(min(512, size - pos))
            if len(head) < 7:
                raise _RarFormat("short block header")
            htype, flags, hsize = head[2], struct.unpack("<H", head[3:5])[0], struct.unpack("<H", head[5:7])[0]
            if hsize < 7:
                raise _RarFormat("bad block size")
            add = 0
            if htype == 0x73:                                    # main header
                volume = bool(flags & 0x0001)
                if flags & 0x0080:
                    raise _Unknown("encrypted headers")
            elif htype == 0x74:                                  # file header
                if len(head) < 32:
                    raise _RarFormat("short file header")
                add = struct.unpack("<I", head[7:11])[0]
                unp = struct.unpack("<I", head[11:15])[0]
                name_len = struct.unpack("<H", head[26:28])[0]
                off = 32
                if flags & 0x0100:                               # 64-bit sizes
                    add |= struct.unpack("<I", head[32:36])[0] << 32
                    unp |= struct.unpack("<I", head[36:40])[0] << 32
                    off = 40
                if off + name_len > len(head):
                    fp.seek(pos + off)
                    raw_name = fp.read(name_len)
                else:
                    raw_name = head[off:off + name_len]
                raw_name = raw_name.split(b"\0", 1)[0]
                name = raw_name.decode("utf-8", "replace") if flags & 0x0200 else raw_name.decode("latin-1")
                is_dir = (flags & 0x00E0) == 0x00E0
                if not is_dir and on_member(name.replace("\\", "/"), unp):
                    return False, volume
            elif htype == 0x7B:                                  # end of archive
                return True, volume
            elif flags & 0x8000:                                 # other block with data
                add = struct.unpack("<I", head[7:11])[0]
            pos += hsize + add
        return True, volume
    if sig == b"Rar!\x1a\x07\x01\x00":
        pos = 8
        while pos + 4 < size:
            fp.seek(pos)
            head = fp.read(min(512, size - pos))
            hsize, i = _vint(head, 4)
            start = i
            if hsize > 1 << 21:
                raise _RarFormat("implausible header size")
            if start + hsize > len(head):
                fp.seek(pos)
                head = fp.read(start + hsize)
            htype, i = _vint(head, i)
            hflags, i = _vint(head, i)
            data_size = 0
            if hflags & 0x1:
                _, i = _vint(head, i)                            # extra area size
            if hflags & 0x2:
                data_size, i = _vint(head, i)
            if htype == 1:                                       # main archive header
                aflags, i = _vint(head, i)
                volume = bool(aflags & 0x1)
            elif htype == 4:                                     # archive encryption header
                raise _Unknown("encrypted headers")
            elif htype == 2:                                     # file header
                fflags, i = _vint(head, i)
                unp, i = _vint(head, i)
                _, i = _vint(head, i)                            # attributes
                if fflags & 0x2:
                    i += 4
                if fflags & 0x4:
                    i += 4
                _, i = _vint(head, i)                            # compression info
                _, i = _vint(head, i)                            # host OS
                name_len, i = _vint(head, i)
                name = head[i:i + name_len].decode("utf-8", "replace")
                if not fflags & 0x1 and on_member(name.replace("\\", "/"), unp):
                    return False, volume
            elif htype == 5:                                     # end of archive
                return True, volume
            pos += start + hsize + data_size
        return True, volume
    raise _RarFormat("not a RAR 4 or RAR 5 signature")



def _rar_is_volume(rf) -> bool:
    """Main header volume flag (RAR 4 and 5 alike). When it cannot be read
    the archive counts as a volume: never proven empty."""
    import rarfile
    try:
        main = rf._file_parser._main
        return main is None or bool(main.flags & rarfile.RAR_MAIN_VOLUME)
    except Exception:  # noqa: BLE001
        return True


def _probe_zip(client, url, res: ProbeResult, stream_bytes: int, max_requests: int):
    """Zip downloaded whole (--no-remote-zip): the central directory."""
    import zipfile
    res.method = "zip_central_directory"
    fp = ProbeRangeFile(client, url, res.size, stream_bytes, max_requests, base_block=256 << 10)
    lst = _Listing(res, noext_readable=True)
    try:
        with zipfile.ZipFile(fp) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir() and not (i.flag_bits & 0x1)]
        for i in infos:
            lst.add(i.filename, i.file_size)
        res.complete = True
    finally:
        res.bytes_used += fp.bytes_fetched
        res.n_requests += fp.n_requests


def _probe_single(client, url, res: ProbeResult):
    """x.tif.gz: by the inner name; extensionless inner: DICM sniff."""
    ext = detect_ext(res.key)
    inner = res.key[: -len(ext)] if ext else res.key
    ik = file_kind(inner)
    res.method = "inner_name"
    res.n_members = 1
    res.kind_counts[ik] += 1
    res.ext_counts[detect_ext(inner) or "(none)"] += 1
    if ik in IMAGE_KINDS:
        res.n_images, res.hit, res.complete = 1, inner, True
        return
    if ik == "archive":
        raise _Unknown("compressed archive: contents not listed")
    if ik != "noext" or ext not in _SINGLE_OPENERS:
        res.complete = True
        res.names.append(inner)
        return
    res.method = "inner_sniff"
    n = min(res.size, 64 << 10)
    r = client.get(url, headers={"Range": f"bytes=0-{n - 1}"}, stream=True, max_tries=3)
    try:
        if r.status_code not in (200, 206):
            raise RemoteError(f"HTTP {r.status_code}", r.status_code)
        data = b"".join(r.iter_content(chunk_size=n))[:n]
    finally:
        r.close()
    res.n_requests += 1
    res.bytes_used += len(data)
    try:
        dec = _SINGLE_OPENERS[ext]()
        head = dec.decompress(data, SNIFF_BYTES) if ext != ".zst" else dec.decompress(data)[:SNIFF_BYTES]
    except Exception as e:  # noqa: BLE001 - zlib, lzma, bz2, zstd errors alike
        raise _Unknown(f"decompress failed: {e}") from e
    if len(head) < SNIFF_BYTES and len(data) < res.size:
        raise _Unknown("too few decompressed bytes to sniff")
    res.complete = True
    if sniff_kind(head) is not None:
        res.n_images, res.hit = 1, inner
    else:
        res.names.append(inner)


class _Unknown(Exception):
    """The probe cannot decide (encrypted, unsupported layout)."""


def probe_archive(client: ZenodoClient, record_id: str, key: str, size: int, *, split: bool = False,
                  stream_bytes: int = STREAM_BYTES, max_requests: int = MAX_REQUESTS,
                  min_bytes: int | None = None) -> ProbeResult:
    """Probe one Zenodo file. ``split``: the file is part of a split
    archive set (x.part1.rar, x.r00, x.7z.001, x.zip + x.z01)."""
    res = ProbeResult(key=key, size=int(size or 0), fmt=probe_format(key))
    min_bytes = MIN_PROBE_BYTES if min_bytes is None else min_bytes
    if not res.fmt:
        res.note = "not an archive"
        return res
    if res.fmt == "single":
        pass                                   # decided by name, or one small read
    elif res.size <= min_bytes:
        res.note = f"small archive (at most {min_bytes >> 20} MB): downloaded without a probe"
        return res
    url = file_url(record_id, key)
    try:
        if res.fmt == "tar":
            _probe_tar(client, url, res, stream_bytes, max_requests)
        elif res.fmt == "tar_stream":
            _probe_tar_stream(client, url, res, stream_bytes)
        elif res.fmt == "7z" and not split:
            _probe_7z(client, url, res, stream_bytes, max_requests)
        elif res.fmt == "rar":
            _probe_rar(client, url, res, stream_bytes, max_requests, volume=split)
        elif res.fmt == "zip" and not split:
            _probe_zip(client, url, res, stream_bytes, max_requests)
        elif res.fmt == "single":
            _probe_single(client, url, res)
        else:
            res.note = "split archive set or format without an index: not probed"
            return res
    except _ProbeLimit as e:
        res.note = f"listing not finished: {e}"
    except _Unknown as e:
        res.note = str(e)[:300]
    except Exception as e:  # noqa: BLE001 - a failed probe falls back to the download
        res.note = f"probe failed: {type(e).__name__}: {e}"[:300]
        logger.info("archive probe %s/%s failed: %s", record_id, key, res.note)
    if res.positive:
        res.outcome = "images"
    elif res.complete and not res.note:
        res.outcome = "no_images"
    else:
        res.outcome = "unknown"
        if res.complete and res.note:
            res.complete = False
    if split and res.outcome == "no_images":
        res.outcome, res.complete = "unknown", False
        res.note = res.note or "split archive set: one part cannot prove the others empty"
    return res


_SPLIT_RE = [
    re.compile(r"^(?P<stem>.+)\.(?:z\d{2,3}|zip)$"),
    re.compile(r"^(?P<stem>.+)\.part\d+\.rar$"),
    re.compile(r"^(?P<stem>.+)\.(?:r\d{2,3}|rar)$"),
    re.compile(r"^(?P<stem>.+\.(?:7z|zip))\.\d{3}$"),
]


def split_members(keys: list[str]) -> set[str]:
    """Keys that belong to a split archive set of two or more parts
    (rar, spanned zip and numbered 7z sets, and the generic x.tar.001,
    x.zip.partaa, x.part01 sets the survey joins)."""
    units: dict[str, list[str]] = {}
    for k in keys:
        stem = split_stem(k)
        if stem is not None:
            units.setdefault(f"g:{stem.lower()}", []).append(k)
    for k in keys:
        low = k.lower()
        for i, rx in enumerate(_SPLIT_RE):
            m = rx.match(low)
            if m:
                units.setdefault(f"{i}:{m.group('stem')}", []).append(k)
                break
    out = {k for ks in units.values() if len(ks) > 1 for k in ks}
    # x.part1.rar alone still announces a volume set.
    out |= {k for k in keys if re.search(r"\.part\d+\.rar$", k.lower())}
    return out


@dataclass
class RecordProbe:
    results: list[ProbeResult] = field(default_factory=list)
    stop: str = ""

    @property
    def bytes_used(self) -> int:
        return sum(r.bytes_used for r in self.results)

    @property
    def n_requests(self) -> int:
        return sum(r.n_requests for r in self.results)

    def by_outcome(self, outcome: str) -> list[ProbeResult]:
        return [r for r in self.results if r.outcome == outcome]

    def detail(self) -> str:
        parts = []
        for r in self.results:
            s = f"{r.key}: {r.outcome}"
            if r.method:
                s += f" ({r.method}, {r.n_members} members, {r.n_images} images"
                if r.n_nested:
                    s += f", {r.n_nested} nested"
                if r.n_noext:
                    s += f", {r.n_noext} extensionless"
                s += f", {r.bytes_used} B, {r.n_requests} req)"
            if r.note:
                s += f" [{r.note[:120]}]"
            parts.append(s)
        return "; ".join(parts)[:4000]


def probe_record_archives(client: ZenodoClient, record_id: str, files: list[dict], all_keys: list[str], *,
                          stream_bytes: int = STREAM_BYTES, max_requests: int = MAX_REQUESTS,
                          record_bytes: int = RECORD_PROBE_BYTES, record_requests: int = RECORD_PROBE_REQUESTS,
                          min_bytes: int | None = None) -> RecordProbe:
    """Probe the archives among ``files`` (Zenodo file dicts {key, size})
    that would be downloaded whole. ``all_keys``: every file name of the
    record (split sets). Fragments (x.r00, x.z01, x.7z.002) are never
    probed: their set follows the first part, which never tests negative."""
    out = RecordProbe()
    split = split_members(all_keys)
    for f in files:
        key = f["key"]
        fmt = probe_format(key)
        if not fmt or file_kind(key) == "fragment":
            continue
        free = fmt != "single" and int(f.get("size") or 0) <= (MIN_PROBE_BYTES if min_bytes is None else min_bytes)
        if not free and (out.bytes_used >= record_bytes or out.n_requests >= record_requests):
            out.stop = out.stop or "record probe limit reached"
            res = ProbeResult(key=key, size=int(f.get("size") or 0), fmt=fmt, outcome="not_probed",
                              note="record probe limit reached")
        else:
            left_bytes = record_bytes - out.bytes_used
            left_req = record_requests - out.n_requests
            try:
                res = probe_archive(client, record_id, key, f.get("size") or 0, split=key in split,
                                    stream_bytes=min(stream_bytes, left_bytes),
                                    max_requests=min(max_requests, left_req), min_bytes=min_bytes)
            except requests.RequestException as e:      # belt and braces: probe_archive catches
                res = ProbeResult(key=key, size=int(f.get("size") or 0), fmt=fmt, outcome="unknown",
                                  note=f"probe failed: {type(e).__name__}")
        out.results.append(res)
    return out
