"""Find image files in a record without unpacking whole archives.

A full ``unpack_tree`` would double disk use on records up to 80 GB, so the
survey lists archive members instead and later reads only the sampled ones:

* zip: ``zipfile`` central directory (instant). Members using compression
  Python cannot read (deflate64, bzip2 variants in old zips, spanned sets)
  are read through the ``7z`` binary instead.
* tar family (zstd too): one streaming pass to list, one streaming pass to
  read the sampled members (no seeking, no member table kept for reading).
* 7z, rar, spanned zip (.zip + .z01..): ``7z l -slt`` to list, ``7z x`` with a
  list file to extract only the sampled members into a temp dir. RAR members
  are extracted with ``unrar`` because many 7z builds cannot decode RAR.
  Without a 7z binary the archive is unpacked whole with the package's
  ``unpack_archive`` and walked as a directory.

Nested archives (zip in zip, tar in zip, ...) are extracted one member at a
time into the record's scratch dir and walked recursively up to ``max_depth``.
Single-file compressed members (x.tif.gz) are decompressed the same way.

Extensionless members (and files) are sniffed by their first bytes
(constants.sniff_kind: DICOM with or without the preamble, PNG, JPEG, TIFF,
BMP). A volume header member (x.mhd, x.hdr) is read together with its data
member (x.raw, x.img) from the same directory: both are extracted under
their own names into one scratch directory.

Large zips written without ZIP64 records (32-bit offsets that wrapped past
4 GiB) make ``zipfile`` shift every member's offset; a member whose local
header is not where the central directory says is looked up at the offsets
4 GiB apart (``open_zip_member``). Deflate64 members are read through
zipfile-deflate64 when it is installed, else through 7z.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import logging
import os
import shutil
import struct
import subprocess
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterator

from ..unpack import unpack_archive
from .constants import (
    IMAGE_KINDS, PATH_KINDS, RAR_EXTS, SEVEN_ZIP_EXTS, SNIFF_BYTES, TAR_EXTS, ZIP_EXTS,
    detect_ext, file_kind, nested_worth, sniff_kind, split_first, volume_companion,
)

try:                    # registers Deflate64 (method 9) with zipfile
    import zipfile_deflate64  # noqa: F401
    _DEFLATE64 = True
except Exception:  # noqa: BLE001 - optional; 7z reads Deflate64 too
    _DEFLATE64 = False

logger = logging.getLogger(__name__)

SEVEN_ZIP = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
# Many 7z builds (e.g. Ubuntu's 7zip) list RAR archives but cannot decode them
# ("Unsupported Method"), so RAR members are extracted with unrar when present.
UNRAR = shutil.which("unrar")
# zipfile can read stored(0), deflate(8), bzip2(12), lzma(14).
_PY_ZIP_METHODS = {0, 8, 12, 14} | ({9} if _DEFLATE64 else set())
DICM_MAGIC_OFFSET = 128


@dataclass(slots=True)
class Entry:
    """One image-bearing file, either on disk or inside an archive."""
    container: int          # index into RecordWalker.containers (-1 = plain file)
    member: str             # member name inside the container, or absolute path
    display: str            # record-relative display path, "a.zip!/x/y.png"
    ext: str
    kind: str               # an IMAGE_KINDS kind
    size: int
    companion: str | None = None   # volume_pair: the data member (same directory)


@dataclass
class Container:
    kind: str               # zip | tar | 7z | rar
    path: Path
    display: str
    depth: int
    use_7z: bool = False
    n_members: int = 0
    n_images: int = 0
    error: str | None = None


@dataclass
class RecordWalker:
    """Collects image entries (and counts of everything else) for one record."""

    scratch: Path
    max_depth: int = 3
    disk_floor_bytes: int = 0
    reserved_by_others: object = None      # callable -> bytes other survey processes reserved on this disk
    noext_sniff_limit: int = 2000
    containers: list[Container] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    kind_counts: Counter = field(default_factory=Counter)
    image_ext_counts: Counter = field(default_factory=Counter)
    member_ext_counts: Counter = field(default_factory=Counter)   # every archive member listed, by extension
    other_names: list[str] = field(default_factory=list)   # sample, for text hints
    errors: list[str] = field(default_factory=list)
    _nested_n: int = 0
    _sniffed: int = 0

    # ------------------------------------------------------------------ add
    def add_file(self, path: Path, display: str, depth: int = 0):
        """Register a top-level (or decompressed) file."""
        kind = file_kind(path.name)
        ext = detect_ext(path.name)
        if kind in ("noext", "other"):
            sniffed = _sniff_file(path)
            if sniffed is not None and (kind == "noext" or sniffed[0] == "dicom"):
                kind, ext = sniffed
        self.kind_counts[kind] += 1
        if kind in IMAGE_KINDS:
            self._add_entry(Entry(-1, str(path), display, ext or ".dcm", kind, _size(path)))
        elif kind == "archive":
            self._walk_archive(path, display, depth)
        elif kind == "compressed":
            self._decompress_single(path, display, depth)
        else:
            self._note_other(display)

    def add_tree(self, root: Path, display_prefix: str, depth: int):
        """Register every file under a directory (used for fallback unpacks)."""
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                p = Path(dirpath) / fn
                rel = p.relative_to(root).as_posix()
                self.add_file(p, f"{display_prefix}{rel}", depth)

    def _add_entry(self, e: Entry):
        self.entries.append(e)
        self.image_ext_counts[e.ext] += 1
        if e.container >= 0:
            self.containers[e.container].n_images += 1

    def _note_other(self, display: str):
        if len(self.other_names) < 5000:
            self.other_names.append(display)

    # ------------------------------------------------------------- archives
    def _walk_archive(self, path: Path, display: str, depth: int):
        ext = detect_ext(path.name)
        low = path.name.lower()
        try:
            if ext in ZIP_EXTS or low.endswith(".zip.001"):
                self._walk_zip(path, display, depth)
            elif ext in TAR_EXTS:
                self._walk_tar(path, display, depth)
            elif ext in SEVEN_ZIP_EXTS or ext in RAR_EXTS or split_first(low):
                # 7z opens any numbered split set (x.7z.001, x.tar.001, x.001)
                self._walk_7z(path, display, depth)
            else:
                self._note_other(display)
        except EncryptedArchiveError as e:
            self.kind_counts["encrypted"] += 1
            self.errors.append(f"{display}: encrypted archive skipped ({e})"[:300])
        except Exception as e:  # noqa: BLE001 - one bad archive must not stop the record
            msg = f"{display}: {type(e).__name__}: {e}"[:300]
            logger.warning("archive walk failed %s", msg)
            self.errors.append(msg)

    def _new_container(self, kind: str, path: Path, display: str, depth: int) -> int:
        self.containers.append(Container(kind, path, display, depth))
        return len(self.containers) - 1

    def _member(self, ci: int, name: str, size: int, depth: int, opener=None, siblings=None):
        """Classify one archive member; recurse into nested archives.
        ``siblings``: every member name of the container (volume pairs)."""
        c = self.containers[ci]
        c.n_members += 1
        disp = f"{c.display}!/{name}"
        kind = file_kind(name)
        ext = detect_ext(name)
        self.member_ext_counts[ext or "(none)"] += 1
        if kind == "noext" and opener is not None and self._sniffed < self.noext_sniff_limit \
                and 8 <= size < (64 << 20):
            self._sniffed += 1
            try:
                with opener() as fp:
                    head = fp.read(SNIFF_BYTES)
                sniffed = sniff_kind(head)
                if sniffed is not None:
                    kind, ext = sniffed
            except Exception:  # noqa: BLE001
                pass
        self.kind_counts[kind] += 1
        if kind in IMAGE_KINDS:
            comp = None
            if kind == "volume_pair" and siblings is not None:
                comp = volume_companion(name, siblings)
            self._add_entry(Entry(ci, name, disp, ext or ".dcm", kind, size, comp))
        elif (kind in ("archive", "compressed") and depth < self.max_depth
              and _nested_worth(name, kind)):
            nested = self._extract_nested(ci, name, size)
            if nested is not None:
                if kind == "archive":
                    self._walk_archive(nested, disp, depth + 1)
                else:
                    self._decompress_single(nested, disp, depth + 1)
                self._drop_if_unused(nested)
        else:
            self._note_other(disp)

    def _nested_path(self, name: str) -> Path | None:
        self._nested_n += 1
        d = self.scratch / "nested" / str(self._nested_n)
        d.mkdir(parents=True, exist_ok=True)
        return d / PurePosixPath(name).name

    def _drop_if_unused(self, path: Path):
        """Delete an extracted/decompressed scratch file nothing will read.

        A nested archive stays only while one of its members is an image
        entry; a decompressed file stays only when it became an image entry.
        Deeper nested archives were extracted to their own files, so they do
        not depend on this one. Without this, per-patient inner zips with no
        images would fill the disk to the floor during listing and hide the
        images of every archive listed after that.
        """
        sp = str(path)
        if any(e.container < 0 and e.member == sp for e in self.entries):
            return
        if any(c.path == path and c.n_images > 0 for c in self.containers):
            return
        try:
            path.unlink(missing_ok=True)
            if path.parent.parent == self.scratch / "nested":
                path.parent.rmdir()  # only succeeds when empty
        except OSError:
            pass

    def _room_for(self, nbytes: int) -> bool:
        free = shutil.disk_usage(self.scratch).free
        if self.reserved_by_others is not None:
            free -= self.reserved_by_others()
        return free - nbytes >= self.disk_floor_bytes

    def _extract_nested(self, ci: int, name: str, size: int) -> Path | None:
        c = self.containers[ci]
        if not self._room_for(size):
            self.errors.append(f"{c.display}!/{name}: nested archive skipped (disk floor)")
            return None
        dest = self._nested_path(name)
        try:
            if c.kind == "zip" and not c.use_7z:
                with zipfile.ZipFile(c.path) as zf, zf.open(name) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            elif c.kind == "tar":
                return None  # tar members are extracted inline while streaming
            else:
                got = _extract_members(c, [name], dest.parent)
                src = got.get(name)
                if src is None:
                    raise RuntimeError("7z did not produce the member")
                if src != dest:
                    shutil.move(str(src), dest)
            return dest
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"{c.display}!/{name}: nested extract failed: {e}"[:300])
            dest.unlink(missing_ok=True)
            return None

    def _walk_zip(self, path: Path, display: str, depth: int):
        multipart = (path.with_suffix(".z01").exists()
                     or path.name.lower().endswith(".zip.001"))
        if multipart:
            if not SEVEN_ZIP:
                self.errors.append(f"{display}: spanned zip needs the 7z binary")
                return
            self._walk_7z(path, display, depth)
            return
        try:
            zf = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, NotImplementedError, OSError) as e:
            if SEVEN_ZIP:
                self._walk_7z(path, display, depth)
                return
            raise RuntimeError(f"unreadable zip: {e}") from e
        with zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            use_7z = any(i.compress_type not in _PY_ZIP_METHODS for i in infos) and bool(SEVEN_ZIP)
            ci = self._new_container("zip" if not use_7z else "7z", path, display, depth)
            self.containers[ci].use_7z = use_7z
            names = [i.filename for i in infos]
            for info in infos:
                if info.flag_bits & 0x1:
                    self.kind_counts["encrypted"] += 1
                    continue
                opener = None
                if not use_7z:
                    opener = (lambda i=info: open_zip_member(zf, i))
                self._member(ci, info.filename, info.file_size, depth, opener, siblings=names)

    def _walk_tar(self, path: Path, display: str, depth: int):
        ci = self._new_container("tar", path, display, depth)
        names: list[str] = []
        n0 = len(self.entries)
        with open_tar_stream(path) as tf:
            for m in tf:
                if m.isfile():
                    names.append(m.name)
                if not m.isfile():
                    continue
                kind = file_kind(m.name)
                if (kind in ("archive", "compressed") and depth < self.max_depth
                        and _nested_worth(m.name, kind)):
                    # Streaming mode: extract the nested archive now, while it
                    # is the current member.
                    c = self.containers[ci]
                    c.n_members += 1
                    self.kind_counts[kind] += 1
                    if not self._room_for(m.size):
                        self.errors.append(f"{display}!/{m.name}: nested skipped (disk floor)")
                        continue
                    dest = self._nested_path(m.name)
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    with open(dest, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
                    disp = f"{display}!/{m.name}"
                    if kind == "archive":
                        self._walk_archive(dest, disp, depth + 1)
                    else:
                        self._decompress_single(dest, disp, depth + 1)
                    self._drop_if_unused(dest)
                    continue
                opener = None
                if kind == "noext":
                    opener = (lambda mm=m: _Closing(tf.extractfile(mm)))
                self._member(ci, m.name, m.size, depth, opener)
        # A tar is listed in one streaming pass: volume headers learn their
        # data member once every name is known.
        for e in self.entries[n0:]:
            if e.container == ci and e.kind == "volume_pair" and e.companion is None:
                e.companion = volume_companion(e.member, names)

    def _walk_7z(self, path: Path, display: str, depth: int):
        if not SEVEN_ZIP:
            # Fallback: full unpack with the package's own unpacker.
            if not self._room_for(_size(path) * 2):
                self.errors.append(f"{display}: no 7z binary and no room to unpack")
                return
            dest = self.scratch / "unpacked" / str(len(self.containers))
            if unpack_archive(path, dest):
                self.add_tree(dest, f"{display}!/", depth + 1)
            else:
                self.errors.append(f"{display}: unpack failed")
            return
        is_rar = detect_ext(path.name) in RAR_EXTS
        kind = "rar" if (is_rar and UNRAR) else "7z"
        ci = self._new_container(kind, path, display, depth)
        self.containers[ci].use_7z = True
        try:
            listing = _7z_list(path, self.kind_counts)
        except EncryptedArchiveError:
            raise
        except RuntimeError:
            if not is_rar:
                raise
            listing = _rar_list(path)
        names = [n for n, _ in listing]
        for name, size in listing:
            self._member(ci, name, size, depth, None, siblings=names)

    def _decompress_single(self, path: Path, display: str, depth: int):
        """x.tif.gz -> x.tif (only when the inner file is something we read)."""
        inner = path.name[: -len(detect_ext(path.name))]
        if file_kind(inner) not in IMAGE_KINDS | {"archive", "noext"} or depth >= self.max_depth:
            self._note_other(display)
            return
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open, ".zst": _zstd_open}.get(detect_ext(path.name))
        if opener is None:
            return
        # Budget: assume up to 10x expansion when checking the disk floor.
        if not self._room_for(_size(path) * 10):
            self.errors.append(f"{display}: decompress skipped (disk floor)")
            return
        dest = self._nested_path(inner)
        try:
            with opener(path, "rb") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"{display}: decompress failed: {e}"[:300])
            dest.unlink(missing_ok=True)
            return
        self.add_file(dest, display[: -len(detect_ext(path.name))], depth + 1)
        self._drop_if_unused(dest)

    # ---------------------------------------------------------------- read
    def read_entries(
        self, entries: list[Entry], max_inmem_bytes: int = 256 << 20,
    ) -> Iterator[tuple[Entry, bytes | None, Path | None, str | None]]:
        """Yield (entry, data, path, error) for each requested entry.

        Exactly one of data/path is set on success. Volumes always come as a
        path (their readers need a file). Temp files are removed when the
        caller advances the iterator, so use the data before the next step.
        """
        by_container: dict[int, list[Entry]] = defaultdict(list)
        for e in entries:
            by_container[e.container].append(e)

        for ci, group in by_container.items():
            if ci < 0:
                for e in group:
                    yield e, None, Path(e.member), None
                continue
            c = self.containers[ci]
            done: set[int] = set()   # id() of entries already yielded from this group
            try:
                if c.kind == "zip":
                    reader = self._read_zip(c, group, max_inmem_bytes)
                elif c.kind == "tar":
                    reader = self._read_tar(c, group, max_inmem_bytes)
                else:
                    reader = self._read_7z(c, group)
                for item in reader:
                    done.add(id(item[0]))
                    yield item
            except Exception as ex:  # noqa: BLE001
                msg = f"{type(ex).__name__}: {ex}"[:200]
                self.errors.append(f"{c.display}: read failed: {msg}")
                # Only entries not yet yielded: the others were already
                # classified (or reported) and must not be counted twice.
                for e in group:
                    if id(e) not in done:
                        yield e, None, None, msg

    def _spill(self, data_src, e: Entry) -> Path:
        tmp = self.scratch / "spill"
        tmp.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=tmp, suffix=e.ext)
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(data_src, out, 1 << 20)
        return Path(name)

    def _pair_dir(self) -> Path:
        """Fresh scratch dir for a volume header and its data file (both keep
        their own base names there, as the header refers to the data file)."""
        (self.scratch / "spill").mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(dir=self.scratch / "spill", prefix="pair_"))

    @staticmethod
    def _save_as(src, dest: Path) -> Path:
        with open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        return dest

    def _read_zip(self, c: Container, group: list[Entry], max_inmem: int):
        with zipfile.ZipFile(c.path) as zf:
            for e in group:
                try:
                    if e.kind == "volume_pair" and e.companion:
                        d = self._pair_dir()
                        try:
                            for name in (e.member, e.companion):
                                with open_zip_member(zf, zf.getinfo(name)) as src:
                                    self._save_as(src, d / PurePosixPath(name).name)
                            yield e, None, d / PurePosixPath(e.member).name, None
                        finally:
                            shutil.rmtree(d, ignore_errors=True)
                        continue
                    with open_zip_member(zf, zf.getinfo(e.member)) as src:
                        if e.kind in PATH_KINDS or e.size > max_inmem:
                            p = self._spill(src, e)
                            try:
                                yield e, None, p, None
                            finally:
                                p.unlink(missing_ok=True)
                        else:
                            yield e, src.read(), None, None
                except (NotImplementedError, zipfile.BadZipFile, OSError, KeyError) as ex:
                    yield e, None, None, f"{type(ex).__name__}: {ex}"[:200]

    def _read_tar(self, c: Container, group: list[Entry], max_inmem: int):
        wanted = {e.member: e for e in group}
        # Volume pairs: both members are saved as they stream past, and the
        # header is yielded once its data member is there too.
        pair_of = {e.companion: e for e in group if e.kind == "volume_pair" and e.companion}
        pair_dirs: dict[int, Path] = {}
        pair_got: dict[int, int] = {}
        seen = set()
        try:
            with open_tar_stream(c.path) as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    e = wanted.get(m.name)
                    if e is not None and m.name in seen:
                        e = None
                    pe = pair_of.get(m.name)
                    if e is None and pe is None:
                        continue
                    src = tf.extractfile(m)
                    if src is None:
                        if e is not None:
                            seen.add(m.name)
                            yield e, None, None, "not a regular file"
                        continue
                    pair_e = e if (e is not None and e.kind == "volume_pair" and e.companion) else pe
                    if pair_e is not None:
                        k = id(pair_e)
                        if k not in pair_dirs:
                            pair_dirs[k] = self._pair_dir()
                        d = pair_dirs[k]
                        self._save_as(src, d / PurePosixPath(m.name).name)
                        pair_got[k] = pair_got.get(k, 0) + 1
                        if e is not None:
                            seen.add(m.name)
                        if pair_got[k] == 2:
                            try:
                                yield pair_e, None, d / PurePosixPath(pair_e.member).name, None
                            finally:
                                shutil.rmtree(d, ignore_errors=True)
                                pair_dirs.pop(k, None)
                        continue
                    seen.add(m.name)
                    if e.kind in PATH_KINDS or e.size > max_inmem:
                        p = self._spill(src, e)
                        try:
                            yield e, None, p, None
                        finally:
                            p.unlink(missing_ok=True)
                    else:
                        yield e, src.read(), None, None
                    if len(seen) == len(wanted) and not pair_dirs:
                        break
        finally:
            for d in pair_dirs.values():
                shutil.rmtree(d, ignore_errors=True)
        for name, e in wanted.items():
            if name not in seen:
                yield e, None, None, "member not found on second pass"
            elif e.kind == "volume_pair" and e.companion and pair_got.get(id(e), 0) < 2:
                yield e, None, None, "volume data member not found on second pass"

    def _read_7z(self, c: Container, group: list[Entry], chunk: int = 400,
                 max_chunk_bytes: int = 20_000_000_000):
        """Extract sampled members in chunks bounded by count AND bytes.

        Each chunk's byte budget is min(max_chunk_bytes, free - disk floor),
        re-measured per chunk, so 400 large volumes can never be written at
        once. A member that alone exceeds the room left is reported as
        skipped instead of filling the disk.
        """
        i = 0
        while i < len(group):
            free = shutil.disk_usage(self.scratch).free
            budget = min(max_chunk_bytes, free - self.disk_floor_bytes)
            part: list[Entry] = []
            total = 0
            while i < len(group) and len(part) < chunk:
                size = max(0, group[i].size)
                if part and total + size > budget:
                    break
                part.append(group[i])
                total += size
                i += 1
            if total > budget or not self._room_for(total):
                self.errors.append(f"{c.display}: {len(part)} member(s) skipped (disk floor)")
                for e in part:
                    yield e, None, None, "skipped (disk floor)"
                continue
            out_dir = Path(tempfile.mkdtemp(dir=self.scratch, prefix="x7z_"))
            try:
                # 7z and unrar keep member paths: a volume header and its data
                # member land in the same directory.
                names = [e.member for e in part] + [e.companion for e in part
                                                    if e.kind == "volume_pair" and e.companion]
                got = _extract_members(c, names, out_dir)
                for e in part:
                    p = got.get(e.member)
                    if p is None or not p.exists() or (e.size and p.stat().st_size == 0):
                        yield e, None, None, "archive tool did not extract member (unsupported method?)"
                    else:
                        yield e, None, p, None
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Closing:
    """Context-manager wrapper for tarfile.extractfile results (may be None)."""

    def __init__(self, fp):
        self.fp = fp

    def __enter__(self):
        return self.fp if self.fp is not None else io.BytesIO(b"")

    def __exit__(self, *exc):
        if self.fp is not None:
            self.fp.close()


_nested_worth = nested_worth   # shared with constants.wanted_for_download


# ---------------------------------------------------------------------------
# zip members at wrapped offsets, zstd tars, sniffing
# ---------------------------------------------------------------------------
_4GIB = 1 << 32


def wrapped_offsets(header_offset: int, archive_size: int) -> list[int]:
    """Candidate local header offsets of a member whose central directory
    offset may be off by a multiple of 4 GiB: zips over 4 GiB written
    without ZIP64 records store offsets modulo 2**32, and zipfile then adds
    the same 4 GiB multiple to every member. Candidates are the given
    offset shifted down and up by whole 4 GiB steps inside the archive."""
    out = []
    k = header_offset - _4GIB
    while k >= 0:
        out.append(k)
        k -= _4GIB
    k = header_offset + _4GIB
    while k + 30 <= archive_size:
        out.append(k)
        k += _4GIB
    return out


def local_header_matches(head: bytes, info) -> bool:
    """The bytes at a candidate offset are the local header of ``info``:
    signature and file name (the name is compared when it was read)."""
    if len(head) < 30 or head[:4] != b"PK\x03\x04":
        return False
    n_len = struct.unpack("<H", head[26:28])[0]
    name = head[30:30 + n_len]
    if len(name) < n_len:
        return False
    want = info.orig_filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437", "replace")
    return name.replace(b"\\", b"/") == want.replace(b"\\", b"/")


def open_zip_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo):
    """zf.open(info), looking the member up at 4 GiB-shifted offsets when
    its local header is not where the central directory says (see
    wrapped_offsets)."""
    try:
        return zf.open(info)
    except zipfile.BadZipFile as first:
        fp = zf.fp
        try:
            fp.seek(0, 2)
            size = fp.tell()
        except Exception:  # noqa: BLE001
            raise first from None
        n = 30 + len(info.orig_filename.encode("utf-8", "replace")) + 8
        for cand in wrapped_offsets(info.header_offset, size):
            fp.seek(cand)
            if local_header_matches(fp.read(n), info):
                import copy
                fixed = copy.copy(info)
                fixed.header_offset = cand
                return zf.open(fixed)
        raise first from None


def _zstd_open(path, mode="rb"):
    import zstandard
    return zstandard.open(path, mode)


def open_tar_stream(path: Path):
    """tarfile in streaming mode for every tar compression, zstd included
    (tarfile itself reads zstd only from Python 3.14)."""
    low = str(path).lower()
    if low.endswith((".tar.zst", ".tzst")):
        import zstandard
        fp = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fp)
        tf = tarfile.open(fileobj=reader, mode="r|")
        _close = tf.close

        def close():
            try:
                _close()
            finally:
                reader.close()
                fp.close()
        tf.close = close
        return tf
    return tarfile.open(path, mode="r|*")


def _sniff_file(p: Path):
    try:
        with open(p, "rb") as fp:
            return sniff_kind(fp.read(SNIFF_BYTES))
    except OSError:
        return None


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _is_dicom_file(p: Path) -> bool:
    try:
        with open(p, "rb") as fp:
            head = fp.read(132)
        return head[DICM_MAGIC_OFFSET:DICM_MAGIC_OFFSET + 4] == b"DICM"
    except OSError:
        return False


class EncryptedArchiveError(RuntimeError):
    """The archive (or its header) is password protected."""


# A dummy password makes 7z/unrar fail fast on encrypted data instead of
# prompting on the terminal (p7zip reads the prompt from the tty, so
# stdin=DEVNULL alone does not stop it waiting until the timeout).
_7Z_NO_PASSWORD = "-pnone"
_UNRAR_NO_PASSWORD = "-p-"
_ENCRYPTED_MARKERS = ("wrong password", "can not open encrypted archive", "cannot open encrypted archive",
                      "encrypted", "password")


def _looks_encrypted(*texts: str) -> bool:
    low = " ".join(texts).lower()
    return any(m in low for m in _ENCRYPTED_MARKERS)


def _7z_list(path: Path, counts: Counter | None = None) -> list[tuple[str, int]]:
    """List regular-file members of anything 7z can open: [(name, size)].

    Encrypted members are left out (they cannot be read without a password)
    and counted under ``counts["encrypted"]``. An archive with an encrypted
    header raises EncryptedArchiveError.
    """
    proc = subprocess.run(
        [SEVEN_ZIP, "l", "-slt", "-sccUTF-8", _7Z_NO_PASSWORD, str(path)],
        capture_output=True, timeout=3600, stdin=subprocess.DEVNULL,
    )
    text = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode not in (0, 1) and "----------" not in text:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        err = stderr.strip().splitlines()
        if _looks_encrypted(stderr, text):
            raise EncryptedArchiveError(f"7z l: encrypted archive ({err[-1] if err else proc.returncode})")
        raise RuntimeError(f"7z l failed: {err[-1] if err else proc.returncode}")
    body = text.split("----------", 1)[1] if "----------" in text else ""
    out = []
    for block in body.split("\n\n"):
        props = {}
        for line in block.strip().splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                props[k.strip()] = v.strip()
        name = props.get("Path")
        if not name:
            continue
        if props.get("Folder") == "+" or props.get("Attributes", "").startswith("D"):
            continue
        if props.get("Encrypted") == "+":
            if counts is not None:
                counts["encrypted"] += 1
            continue
        try:
            size = int(props.get("Size") or 0)
        except ValueError:
            size = 0
        out.append((name.replace("\\", "/"), size))
    return out


def _extract_members(c: Container, members: list[str], out_dir: Path) -> dict[str, Path]:
    """Extract selected members of a 7z/rar/spanned container."""
    if c.kind == "rar" and UNRAR:
        return _unrar_extract(c.path, members, out_dir)
    return _7z_extract(c.path, members, out_dir)


def _rar_list(path: Path) -> list[tuple[str, int]]:
    """RAR listing fallback through the rarfile module (uses unrar)."""
    import rarfile
    with rarfile.RarFile(str(path)) as rf:
        return [(i.filename.replace("\\", "/"), int(i.file_size or 0))
                for i in rf.infolist() if not i.is_dir()]


def _unrar_extract(archive: Path, members: list[str], out_dir: Path) -> dict[str, Path]:
    """unrar x with a list file; keeps member paths under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, listfile = tempfile.mkstemp(dir=out_dir.parent, suffix=".lst")
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write("\n".join(members) + "\n")
    try:
        # -scfl: the list file is UTF-8.
        subprocess.run(
            [UNRAR, "x", "-o+", "-y", "-idq", _UNRAR_NO_PASSWORD, "-scfl", str(archive),
             f"@{listfile}", f"{out_dir}/"],
            capture_output=True, timeout=6 * 3600, stdin=subprocess.DEVNULL,
        )
    finally:
        os.unlink(listfile)
    return {m: out_dir / m for m in members if (out_dir / m).exists()}


def _7z_extract(archive: Path, members: list[str], out_dir: Path) -> dict[str, Path]:
    """Extract only ``members`` (keeping paths) into out_dir. Returns name->path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, listfile = tempfile.mkstemp(dir=out_dir.parent, suffix=".lst")
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write("\n".join(members) + "\n")
    try:
        subprocess.run(
            [SEVEN_ZIP, "x", "-y", "-bd", "-scsUTF-8", "-sccUTF-8", _7Z_NO_PASSWORD,
             f"-o{out_dir}", str(archive), f"@{listfile}"],
            capture_output=True, timeout=6 * 3600, stdin=subprocess.DEVNULL,
        )
    finally:
        os.unlink(listfile)
    return {m: out_dir / m for m in members if (out_dir / m).exists()}
