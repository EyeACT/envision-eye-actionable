"""Format helpers for images.py: true file type by content, and readers of
array and volume formats that have no (safe, memory-bounded) library reader.

* ``sniff_content``: what a file really holds, from its first bytes. A file
  named .jpg can be an XML error page, a PDF, a git-lfs pointer or a
  git-annex link (a symlink committed as text); those are reported as
  ``wrong_format`` (not pixel-bearing) instead of a decode error. Image
  codecs Pillow lacks (lossless and 12-bit JPEG, JPEG XL, JPEG XR) are
  recognised here and decoded through imagecodecs.
* ``jpeg_frame_info``: the start-of-frame header of a JPEG (process,
  precision, size), read without decoding.
* ``safe_unpickle``: NumPy object arrays (``np.save`` of a dict or a list of
  arrays) unpickled with an allow-list of NumPy reconstruction globals only:
  no other callable can run, so an untrusted file cannot execute code.
* ``walk_arrays``: numeric arrays inside nested dicts, lists, object arrays
  and MATLAB structs, with a dotted name for each.
* ``mat5_variables``: variable headers of a MATLAB v5 file (name, shape,
  class, complex), parsed directly. scipy's whosmat fails on files that hold
  a MATLAB object (table, string, classdef: "'NoneType' object is not
  iterable"); this reader skips such variables.
* ``octave_variables`` / ``octave_text_variables``: GNU Octave binary
  (``Octave-1-L``) and text (``# Created by Octave``) save files.
* ``read_envi`` / ``read_bart``: ENVI hyperspectral and BART (``.hdr`` +
  ``.cfl``) headers, data memory-mapped.
* ``image_like``: rejects tables, signals and noise that pass the shape
  test: a real image has neighbouring pixels far more alike than random
  pixel pairs.

Only numpy and the standard library are imported at module level (images.py
is also loaded standalone by export_onnx).
"""

from __future__ import annotations

import io
import pickle
import re
import struct
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# content sniffing
# ---------------------------------------------------------------------------
# Content kinds that are never pixel-bearing, whatever the extension says.
NON_IMAGE_CONTENT = {"empty", "html page", "XML error page", "XML document", "JSON", "PDF", "git-lfs pointer",
                     "git-annex link", "text"}
_PRINTABLE = set(range(32, 127)) | {9, 10, 12, 13}


def _is_text(b: bytes) -> bool:
    if not b:
        return False
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        # a multi-byte character cut at the end of the sample is still text
        try:
            s = b[:-3].decode("utf-8")
        except UnicodeDecodeError:
            return False
    bad = sum(1 for ch in s if ord(ch) < 32 and ord(ch) not in _PRINTABLE)
    return bad <= len(s) // 100


def sniff_content(head: bytes) -> str | None:
    """Content kind of a file from its first bytes (a few KB): an image
    codec ('jpeg', 'jpegxl', 'jpeg2000', 'jpegxr', 'png', 'gif', 'bmp',
    'tiff', 'webp', 'svg', 'dicom'), an array container ('hdf5', 'npy',
    'mat5', 'octave', 'octave text', 'mrc'), a non-image kind (see
    NON_IMAGE_CONTENT) or None when unknown."""
    if not head:
        return "empty"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:2] == b"\xff\x0a" or head[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n":
        return "jpegxl"
    if head[:12] == b"\x00\x00\x00\x0cjP  \r\n\x87\n" or head[:4] == b"\xff\x4f\xff\x51":
        return "jpeg2000"
    if head[:3] == b"II\xbc":
        return "jpegxr"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:2] == b"BM" and len(head) >= 26:
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    if head[:8] == b"\x89HDF\r\n\x1a\n" or head[512:520] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if head[:6] == b"\x93NUMPY":
        return "npy"
    if head[:10] == b"MATLAB 5.0" or head[:10] == b"MATLAB 7.3":
        return "mat5"
    if head[:9] == b"Octave-1-":
        return "octave"
    if len(head) >= 212 and head[208:212] in (b"MAP ", b"MAP\x00"):
        return "mrc"
    if head[:4] == b"%PDF":
        return "PDF"
    if head[:4] == b"PK\x03\x04":
        return "zip archive"
    if head[:2] == b"\x1f\x8b":
        return "gzip stream"
    sample = head[:2048]
    if not _is_text(sample):
        return None
    low = sample.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if low.startswith(b"nrrd000"):
        return "nrrd"
    if low.startswith((b"objecttype", b"ndims")):
        return "metaimage"
    if low.startswith(b"# created by octave"):
        return "octave text"
    if low.startswith(b"version https://git-lfs"):
        return "git-lfs pointer"
    if re.match(rb"^(\.\./)*\.git/annex/", low) or b"/.git/annex/objects/" in low[:300]:
        return "git-annex link"
    if b"<svg" in low[:2048] and low.startswith((b"<?xml", b"<svg", b"<!doctype svg", b"<!--")):
        return "svg"
    if low.startswith(b"<!doctype html") or low.startswith(b"<html") or b"<html" in low[:512]:
        return "html page"
    if low.startswith(b"<?xml") or low.startswith(b"<"):
        return "XML error page" if b"<error>" in low[:1024] else "XML document"
    if low[:1] in (b"{", b"["):
        return "JSON"
    return "text"


def jpeg_frame_info(buf: bytes) -> dict | None:
    """Start-of-frame of a JPEG stream: {'sof': marker, 'process':
    'baseline' | 'extended' | 'progressive' | 'lossless' | 'arithmetic',
    'precision', 'rows', 'cols', 'components'}; None when no SOF is found
    in ``buf``."""
    names = {0xC0: "baseline", 0xC1: "extended", 0xC2: "progressive", 0xC3: "lossless",
             0xC5: "differential", 0xC6: "differential", 0xC7: "lossless",
             0xC9: "arithmetic", 0xCA: "arithmetic", 0xCB: "arithmetic lossless",
             0xCD: "arithmetic", 0xCE: "arithmetic", 0xCF: "arithmetic lossless"}
    i = 2
    n = len(buf)
    while i + 4 <= n:
        if buf[i] != 0xFF:
            i += 1                      # tolerate padding between segments
            continue
        m = buf[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xDA or m == 0xD9:
            return None
        length = int.from_bytes(buf[i + 2:i + 4], "big")
        if m in names and i + 10 <= n:
            p = buf[i + 4]
            rows = int.from_bytes(buf[i + 5:i + 7], "big")
            cols = int.from_bytes(buf[i + 7:i + 9], "big")
            return {"sof": m, "process": names[m], "precision": int(p), "rows": rows, "cols": cols,
                    "components": int(buf[i + 9])}
        i += 2 + length
    return None


# ---------------------------------------------------------------------------
# NumPy object arrays: restricted unpickling
# ---------------------------------------------------------------------------
# The only globals a pickled NumPy array (and dicts or lists of them) needs.
# Anything else (os.system, builtins.eval, arbitrary classes) is refused.
_PICKLE_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"), ("_codecs", "encode"), ("collections", "OrderedDict"),
    ("numpy.core.records", "recarray"), ("numpy._core.records", "recarray"), ("numpy", "recarray"),
    ("numpy.core.records", "record"), ("numpy._core.records", "record"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _PICKLE_ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def safe_unpickle(fp) -> object:
    """Unpickle with the NumPy allow-list (raises UnpicklingError on any
    other global)."""
    return _SafeUnpickler(fp, encoding="latin1").load()


def walk_arrays(obj, prefix: str = "", limit: int = 5000, depth: int = 0, out: list | None = None) -> list:
    """(name, ndarray) of every numeric array inside nested dicts, lists,
    tuples, object arrays and MATLAB structs (scipy mat_struct, record
    arrays), at most ``limit``."""
    out = [] if out is None else out
    if len(out) >= limit or depth > 8:
        return out
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            if obj.dtype.names:
                for f in obj.dtype.names:
                    for k in range(min(obj.size, 64)):
                        walk_arrays(obj[f].flat[k], f"{prefix}.{f}" if obj.size == 1 else f"{prefix}.{f}[{k}]",
                                    limit, depth + 1, out)
            else:
                for k in range(min(obj.size, 256)):     # a 0-d array (np.save of a dict) keeps its name
                    walk_arrays(obj.flat[k], prefix if obj.ndim == 0 else f"{prefix}[{k}]", limit, depth + 1, out)
        elif obj.dtype.names:
            for f in obj.dtype.names:
                walk_arrays(obj[f], f"{prefix}.{f}", limit, depth + 1, out)
        elif np.issubdtype(obj.dtype, np.number) or obj.dtype == np.bool_:
            out.append((prefix or "array", obj))
        return out
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:1000]:
            if str(k).startswith("__"):          # scipy __header__, __globals__
                continue
            walk_arrays(v, f"{prefix}.{k}" if prefix else str(k), limit, depth + 1, out)
    elif isinstance(obj, (list, tuple)):
        for k, v in enumerate(obj[:256]):
            walk_arrays(v, f"{prefix}[{k}]", limit, depth + 1, out)
    elif hasattr(obj, "_fieldnames"):              # scipy.io.matlab mat_struct
        for f in obj._fieldnames:
            walk_arrays(getattr(obj, f, None), f"{prefix}.{f}", limit, depth + 1, out)
    return out


# ---------------------------------------------------------------------------
# MATLAB v5 variable headers
# ---------------------------------------------------------------------------
MAT_CLASSES = {1: "cell", 2: "struct", 3: "object", 4: "char", 5: "sparse", 6: "double", 7: "single",
               8: "int8", 9: "uint8", 10: "int16", 11: "uint16", 12: "int32", 13: "uint32", 14: "int64",
               15: "uint64", 16: "function_handle", 17: "opaque"}
_MI_MATRIX, _MI_COMPRESSED = 14, 15
_MAT_HEADER_BYTES = 4096           # inflated bytes read to parse one variable's header


def _mat5_sub(b: bytes, off: int, e: str):
    """(type, data, next offset) of one data element (small or normal)."""
    t, n = struct.unpack(e + "II", b[off:off + 8])
    if t >> 16:                                   # small data element: data in the tag
        return t & 0xFFFF, b[off + 4:off + 4 + (t >> 16)], off + 8
    return t, b[off + 8:off + 8 + n], off + 8 + n + ((-n) % 8)


def _mat5_matrix_header(b: bytes, e: str):
    """(name, dims, class name, complex) of a miMATRIX element body."""
    _t, flags_b, off = _mat5_sub(b, 0, e)
    flags = struct.unpack(e + "I", flags_b[:4])[0]
    cls = MAT_CLASSES.get(flags & 0xFF, f"class{flags & 0xFF}")
    if cls == "opaque":                           # MATLAB objects: name first, no dims
        _t, name_b, off = _mat5_sub(b, off, e)
        return name_b.decode("latin1"), (), cls, False
    _t, dims_b, off = _mat5_sub(b, off, e)
    dims = struct.unpack(e + f"{len(dims_b) // 4}i", dims_b[: 4 * (len(dims_b) // 4)])
    _t, name_b, off = _mat5_sub(b, off, e)
    return name_b.decode("latin1"), tuple(int(d) for d in dims), cls, bool(flags & 0x0800)


def mat5_variables(path: Path, limit: int = 5000) -> list[tuple[str, tuple, str, bool]]:
    """Top-level variables of a MATLAB v5 file: (name, shape, class,
    complex). Only each variable's header is read (a compressed one is
    inflated for its first few KB); a variable whose header cannot be
    parsed is skipped."""
    out = []
    with open(path, "rb") as fp:
        hdr = fp.read(128)
        if len(hdr) < 128:
            raise ValueError("not a MAT v5 file (short header)")
        e = "<" if hdr[126:128] == b"IM" else ">"
        size = Path(path).stat().st_size
        pos = 128
        while pos + 8 <= size and len(out) < limit:
            fp.seek(pos)
            t, n = struct.unpack(e + "II", fp.read(8))
            body = pos + 8
            pos = body + n + (0 if t == _MI_COMPRESSED else (-n) % 8)
            try:
                if t == _MI_COMPRESSED:
                    d = zlib.decompressobj()
                    buf, left = b"", n
                    fp.seek(body)
                    while len(buf) < _MAT_HEADER_BYTES and left > 0:
                        chunk = fp.read(min(1 << 16, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                        buf += d.decompress(chunk, _MAT_HEADER_BYTES - len(buf))
                        if d.unconsumed_tail:
                            break
                    t2, n2 = struct.unpack(e + "II", buf[:8])
                    if t2 != _MI_MATRIX:
                        continue
                    out.append(_mat5_matrix_header(buf[8:], e))
                elif t == _MI_MATRIX:
                    out.append(_mat5_matrix_header(fp.read(min(n, _MAT_HEADER_BYTES)), e))
            except (struct.error, zlib.error, ValueError, UnicodeDecodeError):
                continue
    return out


# ---------------------------------------------------------------------------
# GNU Octave save files
# ---------------------------------------------------------------------------
# Octave save_type codes of the binary format (ls-utils.h).
_OCT_SAVE_TYPES = {0: "u1", 1: "u2", 2: "u4", 3: "i1", 4: "i2", 5: "i4", 6: "f4", 7: "f8", 8: "u8", 9: "i8"}
_OCT_INT_TYPES = {"int8": "i1", "int16": "i2", "int32": "i4", "int64": "i8",
                  "uint8": "u1", "uint16": "u2", "uint32": "u4", "uint64": "u8"}


def octave_variables(path: Path, limit: int = 1000) -> list[tuple[str, tuple, np.dtype, int, bool]]:
    """Numeric matrices of a GNU Octave binary save file (``Octave-1-L`` /
    ``Octave-1-B``), also inside structs and cells: (dotted name, shape,
    dtype, data offset, complex), data in column-major order. Nothing is
    read but headers: data is skipped by its size. Parsing stops at the
    first value of a type it cannot size (range, function handle, class
    object); the matrices listed before it are kept."""
    out: list = []
    with open(path, "rb") as fp:
        magic = fp.read(10)
        if not magic.startswith(b"Octave-1-"):
            raise ValueError("not an Octave binary file")
        e = "<" if magic[9:10] == b"L" else ">"
        fp.read(1)                                   # float format
        size = Path(path).stat().st_size

        def i32() -> int:
            b = fp.read(4)
            if len(b) < 4:
                raise EOFError
            return struct.unpack(e + "i", b)[0]

        def dims_() -> tuple:
            nd = i32()
            if nd < 0:
                return tuple(i32() for _ in range(-nd))
            return (nd, i32())                       # old 2D format: rows, columns

        def record(name_override: str | None, prefix: str, depth: int) -> bool:
            n = i32()
            if not 0 < n < 4096:
                return False
            name = fp.read(n).decode("latin1")
            doc = i32()
            if not 0 <= doc < (1 << 20):
                return False
            fp.seek(doc, 1)
            fp.read(1)                               # global flag
            code = fp.read(1)
            if not code or code[0] != 255:
                return False                         # pre-3.0 numeric type codes: not handled
            typ = fp.read(i32()).decode("latin1")
            full = name_override if name_override is not None else (f"{prefix}.{name}" if prefix else name)
            return value(typ, full, depth)

        def value(typ: str, full: str, depth: int) -> bool:
            if depth > 8 or len(out) >= limit:
                return False
            base = typ[:-7] if typ.endswith(" matrix") else ""
            if typ in ("matrix", "complex matrix", "float matrix", "float complex matrix", "bool matrix") \
                    or base in _OCT_INT_TYPES:
                dims = dims_()
                count = int(np.prod(dims, dtype=np.int64))
                cplx = "complex" in typ
                if typ == "bool matrix":
                    dt = np.dtype("u1")
                elif base in _OCT_INT_TYPES:
                    dt = np.dtype(e + _OCT_INT_TYPES[base])
                else:
                    st = fp.read(1)[0]
                    dt = np.dtype(e + _OCT_SAVE_TYPES.get(st, "f8"))
                off = fp.tell()
                nbytes = count * dt.itemsize * (2 if cplx else 1)
                if count < 0 or off + nbytes > size:
                    return False
                out.append((full, dims, dt, off, cplx))
                fp.seek(nbytes, 1)
                return True
            if typ in ("scalar", "complex scalar", "float scalar", "float complex scalar"):
                fp.read(1)                           # save type
                fp.seek((4 if "float" in typ else 8) * (2 if "complex" in typ else 1), 1)
                return True
            if typ == "bool":
                fp.seek(1, 1)
                return True
            if typ.endswith(" scalar") and typ[:-7] in _OCT_INT_TYPES:
                fp.seek(np.dtype(_OCT_INT_TYPES[typ[:-7]]).itemsize, 1)
                return True
            if typ in ("string", "sq_string", "char matrix"):
                fp.seek(int(np.prod(dims_(), dtype=np.int64)), 1)
                return True
            if typ == "cell":
                count = int(np.prod(dims_(), dtype=np.int64))
                return all(record(f"{full}[{k}]", "", depth + 1) for k in range(count))
            if typ == "scalar struct":
                nfields = i32()
                return all(record(None, full, depth + 1) for _ in range(nfields))
            if typ == "struct":
                dims_()
                nfields = i32()
                return all(record(None, full, depth + 1) for _ in range(nfields))
            return False

        try:
            while len(out) < limit and fp.tell() < size:
                if not record(None, "", 0):
                    break
        except (EOFError, struct.error, IndexError, UnicodeDecodeError, ValueError):
            pass
    return out


def octave_array(path: Path, var: tuple) -> np.ndarray:
    """Memory map of one variable from octave_variables (magnitude for a
    complex one is the caller's job: a complex variable comes back as
    complex)."""
    name, dims, dt, off, cplx = var
    if cplx:
        ct = np.dtype(dt.byteorder + ("c8" if dt.itemsize == 4 else "c16")) if dt.byteorder in "<>" else \
            np.dtype("c8" if dt.itemsize == 4 else "c16")
        return np.memmap(path, dtype=ct, mode="r", offset=off, shape=dims, order="F")
    return np.memmap(path, dtype=dt, mode="r", offset=off, shape=dims, order="F")


def octave_text_variables(path: Path, max_bytes: int = 64 << 20) -> list[tuple[str, np.ndarray]]:
    """Numeric matrices of an Octave text save file (``# name:``,
    ``# type: matrix``, ``# rows:``, ``# columns:`` or ``# ndims:``, then
    the values). Files over ``max_bytes`` are read up to that point."""
    with open(path, "rb") as fp:
        text = fp.read(max_bytes).decode("latin1", "replace")
    out = []
    for m in re.finditer(r"# name: (\S+)\n# type: ((?:float |bool |int\d+ |uint\d+ )?matrix)\n"
                         r"(?:# rows: (\d+)\n# columns: (\d+)\n|# ndims: (\d+)\n\s*([\d ]+)\n)", text):
        name = m.group(1)
        if m.group(3):
            dims = (int(m.group(3)), int(m.group(4)))
        else:
            dims = tuple(int(x) for x in m.group(6).split())
        count = int(np.prod(dims, dtype=np.int64))
        body = text[m.end(): m.end() + count * 32]
        vals = np.array(body.split()[:count], dtype=np.float64) if count else np.zeros(0)
        if vals.size != count:
            continue
        out.append((name, vals.reshape(dims, order="F" if m.group(5) else "C")))
    return out


# ---------------------------------------------------------------------------
# ENVI and BART headers
# ---------------------------------------------------------------------------
_ENVI_TYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 6: "c8", 9: "c16", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def parse_envi_header(text: str) -> dict:
    """``key = value`` pairs of an ENVI header (braced values may span
    lines), keys lower case."""
    out: dict = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, re.M):
        out[m.group(1).strip().lower()] = m.group(2).strip().strip("{}").strip()
    return out


def read_envi(header: Path, data: Path) -> tuple[np.ndarray, dict]:
    """Memory map of an ENVI cube shaped (lines, samples, bands) whatever
    its interleave (a view, nothing read yet), and the parsed header."""
    h = parse_envi_header(header.read_text(encoding="latin1", errors="replace"))
    samples, lines = int(h["samples"]), int(h["lines"])
    bands = int(h.get("bands", 1) or 1)
    code = int(h.get("data type", 4))
    if code not in _ENVI_TYPES:
        raise ValueError(f"ENVI data type {code} not supported")
    dt = np.dtype((">" if h.get("byte order", "0").strip() == "1" else "<") + _ENVI_TYPES[code])
    offset = int(h.get("header offset", 0) or 0)
    inter = h.get("interleave", "bsq").strip().lower()
    shape = {"bsq": (bands, lines, samples), "bil": (lines, bands, samples), "bip": (lines, samples, bands)}[inter]
    need = offset + int(np.prod(shape, dtype=np.int64)) * dt.itemsize
    if data.stat().st_size < need:
        raise ValueError(f"ENVI data file shorter than its header says ({data.stat().st_size} < {need} bytes)")
    mm = np.memmap(data, dtype=dt, mode="r", offset=offset, shape=shape)
    cube = {"bsq": lambda a: a.transpose(1, 2, 0), "bil": lambda a: a.transpose(0, 2, 1),
            "bip": lambda a: a}[inter](mm)
    return cube, h


def read_bart(header: Path, data: Path) -> np.ndarray:
    """Memory map of a BART .cfl (complex64, column-major) shaped by the
    ``# Dimensions`` line of its .hdr."""
    lines = header.read_text(encoding="latin1", errors="replace").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip().lower() == "# dimensions")
    dims = tuple(int(x) for x in lines[i + 1].split())
    need = int(np.prod(dims, dtype=np.int64)) * 8
    if data.stat().st_size < need:
        raise ValueError(f"BART data file shorter than its header says ({data.stat().st_size} < {need} bytes)")
    return np.memmap(data, dtype=np.complex64, mode="r", shape=dims, order="F")


# ---------------------------------------------------------------------------
# image-likeness of a 2D plane taken out of a numeric array
# ---------------------------------------------------------------------------
IMAGE_LIKE_CROP = 256              # side of the centre crop the test runs on
IMAGE_LIKE_MAX_RATIO = 0.9         # neighbour / random-pair difference at or over this: not an image


def _column_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks of the values within each column (ties share a rank)."""
    n = a.shape[0]
    order = np.argsort(a, axis=0, kind="stable")
    srt = np.take_along_axis(a, order, axis=0)
    out = np.empty(a.shape, dtype=np.float64)
    for j in range(a.shape[1]):
        cut = np.r_[0, np.flatnonzero(np.diff(srt[:, j])) + 1, n]
        out[order[:, j], j] = np.repeat((cut[:-1] + cut[1:] - 1) / 2.0, np.diff(cut))
    return out


def neighbour_ratio(plane: np.ndarray) -> float | None:
    """Mean absolute difference of neighbouring pixels (the worse of rows
    and columns) over that of random pixel pairs, on a 2x2 block mean of a
    centre crop (block means damp per-pixel speckle), with each column
    replaced by its ranks first: a table whose columns have different
    scales would otherwise look smooth next to its widest column, and a
    monotone per-column transform leaves an image an image. About 1 for
    white noise, tables, sparse count matrices and stacked signals; far
    below 1 for photographs, scans and masks. None when the plane is
    constant or mostly non-finite."""
    a = np.asarray(plane)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)                       # channels first
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1) if a.shape[-1] in (3, 4) else a[..., 0]
    if a.ndim != 2:
        return None
    h, w = a.shape
    c = IMAGE_LIKE_CROP
    r0, c0 = max(0, (h - c) // 2), max(0, (w - c) // 2)
    a = np.asarray(a[r0:r0 + c, c0:c0 + c], dtype=np.float64)
    if np.iscomplexobj(a):
        a = np.abs(a)
    fin = np.isfinite(a)
    if fin.mean() < 0.5:
        return None
    if not fin.all():
        a = np.where(fin, a, np.median(a[fin]))
    # block means: 4 x 4 on planes of 128 px or more (a 64-px grid at least),
    # else 2 x 2; they average speckle down and leave tables random
    b = 4 if min(a.shape) >= 128 else 2 if min(a.shape) >= 16 else 1
    h2, w2 = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    if b > 1:
        a = a[:h2, :w2].reshape(h2 // b, b, w2 // b, b).mean(axis=(1, 3))
    a = _column_ranks(a)
    d_r = np.abs(np.diff(a, axis=0)).mean()
    d_c = np.abs(np.diff(a, axis=1)).mean()
    rng = np.random.default_rng(0)
    flat = a.ravel()
    i, j = rng.integers(0, flat.size, 20000), rng.integers(0, flat.size, 20000)
    d_rand = np.abs(flat[i] - flat[j]).mean()
    if not d_rand > 0:
        return None
    return float(max(d_r, d_c) / d_rand)


def image_like(plane: np.ndarray) -> tuple[bool, float | None]:
    """(is image like, neighbour ratio): see neighbour_ratio. A constant or
    unmeasurable plane counts as image like (the blank check reports it)."""
    r = neighbour_ratio(plane)
    return (r is None or r < IMAGE_LIKE_MAX_RATIO), r


def bytes_head(data, path, n: int = 4096) -> bytes:
    """First ``n`` bytes of in-memory data or a file ('' when unreadable)."""
    try:
        if data is not None:
            if isinstance(data, io.BytesIO):
                return bytes(data.getbuffer()[:n])
            return bytes(data[:n])
        with open(path, "rb") as fp:
            return fp.read(n)
    except (OSError, TypeError):
        return b""
