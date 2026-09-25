"""Decode one frame per image file and collect DICOM-like header facts.

Every loader returns ``Loaded(image, facts, error, frames, views)``:

* ``image`` is an 8-bit RGB ``PIL.Image`` ready for preprocessing, or None
  when the file cannot be turned into an image (no reader, undecodable, no
  image-shaped content). Size never refuses a file: large rasters are read
  at a reduced resolution instead (pyramid level, strided strip or tile
  decode, JPEG DCT scaling, JPEG 2000 resolution levels).
* ``frames``: more frames of the same file classified together with
  ``image`` (video: the frames at 25% and 75% of the duration next to the
  one at 50%); the runner averages their class probabilities.
* ``views``: other images of the same file classified on their own, as
  ``(suffix, image, facts)`` (vendor OCT: the fundus or SLO image next to the
  middle B-scan).
* ``facts`` is a small dict of header facts that map to DICOM attributes
  (Rows, Columns, PhotometricInterpretation, BitsStored, NumberOfFrames,
  PixelSpacing, Manufacturer, ...), plus ``source_format`` and
  ``conversion`` (how the pixels were turned into the classified image).
  Real DICOM files carry their own tags under ``facts["dicom"]``.
  Identifying tags are never read into facts.

Which frame is classified:
    TIFF (all variants):     series 0; a pyramid gives the smallest level whose
                             long side is at least PYRAMID_TARGET px; a stack of
                             same-shape pages (Z, time, ImageJ, OME, LSM) its
                             middle plane, first channel; over the pixel or byte
                             budget a strided decode, strip or tile at a time
    other rasters:           first frame (JPEG at a reduced DCT scale, JPEG 2000
                             at a reduced resolution level when large)
    SVG:                     largest embedded raster image, else rendered at
                             SVG_RENDER_PX with external references removed
    DICOM multi-frame:       middle frame
    NIfTI / NRRD / MHA /
    MHD / Analyze pairs:     middle slice along the shortest spatial axis
    numeric arrays:          image-shaped dataset (see _choose_dataset), middle
                             slice, strided when large
    microscopy:              middle Z plane, first channel, time 0
    vendor OCT:              middle B-scan, plus the fundus / SLO image as a view
    video:                   frames at 25, 50 and 75% of the duration

Pixel conversion: 8-bit data is kept as is; deeper integers and floats are
windowed between robust percentiles (0.5 and 99.5, on a subsample) unless
they hold few distinct values (label maps keep exact levels, min-max). An
alpha channel is composited over white. The mask check of deeper integer
single-channel data runs on its native values (a hot pixel cannot squeeze a
real 16-bit image into a few gray levels), and a constant frame is flagged
``blank`` (not classified).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from .constants import IMG_SIZE, MEAN, STD, TIFF_EXTS, TRAIN_STORED_SIZE

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # the caller enforces its own pixel budget

try:                                    # HEIC / HEIF / AVIF through Pillow
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001 - optional
    pass

# Decode budgets. Nothing is refused for size: a file over these is read at
# a reduced resolution (strided) so the decoded frame stays within them.
STRIDED_TARGET_PIXELS = 16_000_000      # pixels of a reduced (strided) decode
DECODE_MAX_BYTES = 1 << 30              # bytes one plane may take when decoded whole
PYRAMID_TARGET = 1024                   # a pyramid level whose long side is at least this
SVG_RENDER_PX = 512                     # SVG render width
SVG_MAX_BYTES = 64 << 20
VENDOR_MAX_BYTES = 3 << 29              # vendor OCT readers load the whole file (1.5 GiB; 7 GB VM)
ARRAY_MIN_SIDE = 64                     # an image-shaped array plane has both sides at least this
ARRAY_MAX_ASPECT = 32                   # ... and no more elongated (signals, tables)
ARRAY_NAME_HINT = re.compile(r"(^|[^a-z])(img|image|images|oct|bscan|b_scan|vol|volume|fundus|slo|frame|frames)"
                             r"($|[^a-z])", re.I)
# Neural network parameters are 2D/4D float arrays that pass the shape test
# but are not images: tensors named like layer parameters (Keras kernel:0,
# PyTorch weight / running_mean, ...) are never candidates, and an HDF5 that
# Keras wrote (root attribute keras_version or model_config, group
# model_weights) is a model file, not image data.
ARRAY_PARAM_NAME = re.compile(r"(^|[/._-])(kernel|bias|weight|weights|gamma|beta|moving_mean|moving_variance|"
                              r"running_mean|running_var|embeddings?|recurrent_kernel|depthwise_kernel|"
                              r"pointwise_kernel)(:\d+)?$", re.I)
H5_MODEL_ATTRS = ("keras_version", "model_config", "training_config")
WINDOW_MIN_LEVELS = 256                 # more distinct values than this: percentile window

_MODE_PHOTOMETRIC = {
    "1": "MONOCHROME2", "L": "MONOCHROME2", "LA": "MONOCHROME2", "I;16": "MONOCHROME2",
    "I;16B": "MONOCHROME2", "I;16L": "MONOCHROME2", "I": "MONOCHROME2", "F": "MONOCHROME2",
    "P": "PALETTE COLOR", "RGB": "RGB", "RGBA": "RGB", "YCbCr": "YBR_FULL", "CMYK": "RGB",
}
_MODE_BITS = {"1": 1, "L": 8, "LA": 8, "P": 8, "RGB": 8, "RGBA": 8, "YCbCr": 8, "CMYK": 8,
              "I;16": 16, "I;16B": 16, "I;16L": 16, "I": 32, "F": 32}
_TIFF_PHOTOMETRIC = {0: "MONOCHROME1", 1: "MONOCHROME2", 2: "RGB", 3: "PALETTE COLOR", 6: "YBR_FULL"}
_SCREEN_DPI = {72.0, 96.0, 150.0, 300.0}

# DICOM attributes kept from real files (research: dicom_mapping.json). Never
# add patient, institution, UID or serial-number tags here: outputs are public.
DICOM_KEEP = [
    "SOPClassUID", "Modality", "ImageType", "Manufacturer", "ManufacturerModelName",
    "SoftwareVersions", "Rows", "Columns", "NumberOfFrames", "SamplesPerPixel",
    "PhotometricInterpretation", "BitsAllocated", "BitsStored", "PixelSpacing",
    "ImageLaterality", "Laterality", "BodyPartExamined", "IlluminationWaveLength",
    "HorizontalFieldOfView", "AcquisitionDateTime", "StudyDate", "SeriesDescription",
    "ProtocolName", "LossyImageCompression", "LossyImageCompressionMethod",
]
DICOM_KEEP_SEQUENCES = [
    "AnatomicRegionSequence", "AcquisitionDeviceTypeCodeSequence",
    "OphthalmicImageTypeCodeSequence",
]


@dataclass
class Loaded:
    image: Image.Image | None
    facts: dict = field(default_factory=dict)
    error: str | None = None
    frames: list = field(default_factory=list)      # more frames, probabilities averaged with image
    views: list = field(default_factory=list)       # (suffix, image) classified on their own


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def load_frame(kind: str, ext: str, data: bytes | None, path: Path | None,
               max_pixels: int = 80_000_000, companion: Path | None = None) -> Loaded:
    """Decode one representative frame of an image file (see module doc).
    ``companion``: the data file of a volume header pair (x.raw of x.mhd)."""
    tmp = None
    try:
        if data is not None and path is None and kind in ("volume", "volume_pair", "vendor_oct", "video",
                                                          "array", "microscopy"):
            # These readers need a file; the walker normally hands them one.
            fd, name = tempfile.mkstemp(suffix=ext or ".bin")
            with os.fdopen(fd, "wb") as fp:
                fp.write(data)
            tmp = path = Path(name)
            data = None
        if kind == "dicom":
            return _load_dicom(data, path, max_pixels)
        if kind in ("volume", "volume_pair"):
            if path is None:
                return Loaded(None, {"format": ext, "source_format": ext.lstrip(".").upper()},
                              "volume needs a file path")
            return _load_volume(path, ext, max_pixels, companion=companion)
        if kind == "vector":
            return _load_svg(data, path, ext, max_pixels)
        if kind == "vendor_oct":
            return _load_vendor(path, ext, max_pixels)
        if kind == "video":
            return _load_video(path, ext)
        if kind == "array":
            return _load_array(path, ext, max_pixels)
        if kind == "microscopy":
            return _load_microscopy(path, ext, max_pixels)
        return _load_raster(data, path, ext, max_pixels)
    except MemoryError:
        return Loaded(None, {"format": ext, "source_format": ext.lstrip(".").upper()}, "MemoryError")
    except Exception as e:  # noqa: BLE001 - one bad file never stops a record
        return Loaded(None, {"format": ext, "source_format": ext.lstrip(".").upper()},
                      f"{type(e).__name__}: {e}"[:200])
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _finish_image(img: Image.Image, facts: dict, conversion: str, native: np.ndarray | None = None) -> Loaded:
    """Pixel flags, conversion note, blank check; ``native``: the decoded
    array before 8-bit conversion (mask test on native values)."""
    facts["conversion"] = conversion
    _pixel_flags(img, facts)
    if native is not None:
        nm = native_mask_like(native)
        if nm is not None:
            facts["mask_like"] = nm
            facts["mask_test"] = "native"
    return Loaded(img, facts)


def _from_array(arr: np.ndarray, facts: dict, conversion: str) -> Loaded:
    img = array_to_rgb8(arr)
    if img is None:
        return Loaded(None, facts, f"unsupported array shape {tuple(np.shape(arr))} ({conversion})"[:200])
    return _finish_image(img, facts, conversion, native=arr)


# ---------------------------------------------------------------------------
# raster (TIFF through tifffile, everything else through PIL)
# ---------------------------------------------------------------------------
def _is_tiff_bytes(data, path) -> bool:
    try:
        if data is not None:
            head = bytes(data[:4])
        else:
            with open(path, "rb") as fp:
                head = fp.read(4)
    except OSError:
        return False
    return head in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _load_raster(data, path, ext, max_pixels) -> Loaded:
    if ext in TIFF_EXTS or _is_tiff_bytes(data, path):
        try:
            return _load_tiff(data, path, ext, max_pixels)
        except Exception as e:  # noqa: BLE001 - PIL may still read an odd TIFF
            tiff_error = f"{type(e).__name__}: {e}"
            logger.debug("tifffile failed (%s); trying PIL", tiff_error)
    else:
        tiff_error = None
    src = io.BytesIO(data) if data is not None else path
    im = Image.open(src)
    with im:
        facts = _raster_facts(im, ext)
        facts["source_format"] = facts.get("format") or ext.lstrip(".").upper()
        w, h = im.size
        conversion = f"PIL {im.format or ext}"
        if tiff_error:
            conversion += " (tifffile failed)"
        if im.format == "JPEG" and max(w, h) > 4 * IMG_SIZE:
            # Decode at a reduced DCT scale: far less RAM and CPU, and the
            # image is resized to 224 anyway.
            im.draft(im.mode, (2 * IMG_SIZE, 2 * IMG_SIZE))
            if im.size != (w, h):
                conversion += f" DCT-scaled to {im.size[0]}x{im.size[1]}"
        elif im.format == "JPEG2000" and w * h > max_pixels:
            k = 0
            while (w >> k) * (h >> k) > max_pixels and k < 8:
                k += 1
            try:
                im.reduce = k          # decode at a lower resolution level
                conversion += f" at resolution level 1/{1 << k}"
            except Exception:  # noqa: BLE001
                pass
        elif w * h > max_pixels:
            bands = len(im.getbands())
            bpp = max(1, (_MODE_BITS.get(im.mode, 8) + 7) // 8)
            if w * h * bands * bpp > DECODE_MAX_BYTES * 2:
                return Loaded(None, facts, f"too_large ({w}x{h} {im.format}: no reduced decode for this format)")
        im.seek(0)
        im.load()
        if im.size[0] * im.size[1] > max_pixels:
            factor = math.ceil(math.sqrt(im.size[0] * im.size[1] / STRIDED_TARGET_PIXELS))
            im = Image.Image.reduce(im, factor)
            conversion += f" reduced 1/{factor}"
        native = None
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            native = np.asarray(im)
        rgb = to_rgb8(im)
    return _finish_image(rgb, facts, conversion, native=native)


def _pick_level(series):
    """Pyramid level to decode: the smallest level whose long side is at
    least PYRAMID_TARGET (level 0 when every level is smaller)."""
    levels = list(getattr(series, "levels", None) or [series])
    best = levels[0]
    for lv in levels:
        shape = dict(zip(lv.axes, lv.shape))
        long_side = max(shape.get("Y", 0), shape.get("X", 0))
        if long_side >= PYRAMID_TARGET:
            best = lv
    return best, levels.index(best), len(levels)


def _plane_index(level) -> tuple[int, dict]:
    """Page index of the representative plane of a series level: middle of
    the Z / T / I / Q axes (a stack of same-shape pages), channel 0, and the
    plane axes it picked."""
    axes, shape = level.axes, level.shape
    page_axes = [(a, n) for a, n in zip(axes, shape) if a not in "YXS"]
    # Axes inside one page (SGI depth) are rare; tifffile keeps them in
    # shape too, so only as many leading axes as there are pages count.
    n_pages = len(level.pages)
    idx, dims, picked = [], [], {}
    prod = 1
    for a, n in page_axes:
        if prod * n > n_pages:
            break
        i = 0 if a in "CE" else n // 2
        idx.append(i)
        dims.append(n)
        picked[a] = f"{i}/{n}"
        prod *= n
    flat = int(np.ravel_multi_index(idx, dims)) if idx else 0
    return min(flat, max(0, n_pages - 1)), picked


def _tiff_page_facts(page, facts: dict):
    rows, cols = int(page.imagelength), int(page.imagewidth)
    compression = int(page.compression)
    facts.update({
        "rows": rows, "cols": cols, "bits": int(page.bitspersample),
        "samples": int(getattr(page, "samplesperpixel", 1) or 1),
        "photometric": _TIFF_PHOTOMETRIC.get(int(page.photometric), str(page.photometric)),
        "lossy": "01" if compression in (6, 7) else "00",
        "compression": compression,
    })
    if compression in (6, 7):
        facts["lossy_method"] = "ISO_10918_1"
    elif compression in (33003, 33005, 34712):
        facts["lossy_method"] = "ISO_15444_1"
        facts.pop("lossy", None)   # JPEG 2000 may be lossless: unknown
    if int(getattr(page, "sampleformat", 1) or 1) == 3:
        facts["float"] = True
    try:
        desc = page.description
        if desc:
            facts["description"] = str(desc)[:300]
        tags = {t.code: t.value for t in page.tags.values()}
        sp = _tiff_spacing(tags, str(desc or ""))
        if sp:
            facts["spacing_mm"] = sp
        for code, key in ((271, "make"), (272, "model"), (305, "software")):
            v = tags.get(code)
            if v:
                facts[key] = str(v).strip("\x00 ")[:80]
    except Exception:  # noqa: BLE001 - malformed tags are common
        pass


def _load_tiff(data, path, ext, max_pixels, max_bytes: int = DECODE_MAX_BYTES) -> Loaded:
    """Every TIFF variant through tifffile (BigTIFF, tiled, pyramids,
    OME, ImageJ and LSM stacks, 12/16-bit, float, planar)."""
    import tifffile
    src = io.BytesIO(data) if data is not None else path
    facts: dict = {"format": "TIFF"}
    with tifffile.TiffFile(src) as tf:
        kinds = [k for k in ("is_ome", "is_imagej", "is_lsm", "is_svs", "is_ndpi", "is_bigtiff", "is_scn")
                 if getattr(tf, k, False)]
        facts["source_format"] = "TIFF" + (f" ({', '.join(k[3:].upper() for k in kinds)})" if kinds else "")
        facts["frames"] = len(tf.pages)
        series = tf.series[0]
        level, li, n_levels = _pick_level(series)
        pi, picked = _plane_index(level)
        page = level.pages[pi]
        if page is None:
            page = tf.pages[0]
        kf = getattr(page, "keyframe", page)
        _tiff_page_facts(kf, facts)
        if len(series.pages) > 1:
            facts["frames"] = len(series.pages)
        rows, cols = facts["rows"], facts["cols"]
        conv = ["tifffile"]
        if n_levels > 1:
            conv.append(f"pyramid level {li}/{n_levels}")
            base = dict(zip(series.levels[0].axes, series.levels[0].shape))
            facts["full_rows"], facts["full_cols"] = int(base.get("Y", rows)), int(base.get("X", cols))
        if picked:
            conv.append("plane " + ",".join(f"{a}={v}" for a, v in picked.items()))
            facts["frame_index"] = pi
        depth = int(getattr(kf, "imagedepth", 1) or 1)
        nbytes = rows * cols * depth * facts["samples"] * max(1, facts["bits"]) // 8
        if rows * cols > max_pixels or nbytes > max_bytes:
            step = math.ceil(math.sqrt(rows * cols / min(max_pixels, STRIDED_TARGET_PIXELS)))
            seg_bytes = _tiff_segment_bytes(kf)
            if seg_bytes > max_bytes:
                # one strip or tile is itself over the budget (ImageJ and
                # some scanners write the whole plane as one strip):
                # decoding it would hold the whole plane. Uncompressed
                # contiguous data is strided through a view instead.
                arr = _contiguous_strided(tf, page, step, data, path)
                if arr is None:
                    return Loaded(None, facts, f"too_large: one TIFF strip or tile of {seg_bytes >> 20} MB is "
                                               f"over the decode budget of {max_bytes >> 20} MB (compressed)")
                conv.append("uncompressed view")
            else:
                arr = _strided_page(page, step)
            conv.append(f"strided 1/{step}")
            facts["decoded_rows"], facts["decoded_cols"] = int(arr.shape[0]), int(arr.shape[1])
        else:
            arr = page.asarray()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return _from_array(arr, facts, " ".join(conv))


def _tiff_segment_bytes(kf) -> int:
    """Decoded bytes of the largest strip or tile of a TIFF page."""
    W = int(kf.imagewidth)
    H = int(kf.imagelength)
    spp = int(getattr(kf, "samplesperpixel", 1) or 1)
    contig = spp if int(getattr(kf, "planarconfig", 1) or 1) != 2 else 1
    bits = getattr(kf, "bitspersample", 8)
    bits = max(bits) if isinstance(bits, (tuple, list)) else int(bits or 8)
    bps = max(1, -(-bits // 8))
    if getattr(kf, "is_tiled", False):
        depth = int(getattr(kf, "tiledepth", 1) or 1)
        return int(kf.tilelength) * int(kf.tilewidth) * depth * contig * bps
    rps = int(getattr(kf, "rowsperstrip", 0) or 0) or H
    depth = int(getattr(kf, "imagedepth", 1) or 1)
    return min(rps, H) * W * depth * contig * bps


def _contiguous_strided(tf, page, step: int, data, path) -> np.ndarray | None:
    """Every ``step``-th row and column of an uncompressed page whose data
    is one contiguous block, through a memory map of the file (or a view of
    the bytes already in memory): only the strided rows are read. None when
    the page is compressed or not contiguous."""
    kf = getattr(page, "keyframe", page)
    try:
        if not page.is_contiguous or int(kf.compression) != 1 or int(getattr(kf, "fillorder", 1) or 1) != 1 \
                or int(getattr(kf, "predictor", 1) or 1) != 1 or getattr(kf, "is_subsampled", False):
            return None
        bits = kf.bitspersample
        bits = max(bits) if isinstance(bits, (tuple, list)) else int(bits)
        if bits % 8 or kf.dtype is None:
            return None
        dtype = np.dtype(kf.dtype).newbyteorder(tf.byteorder)
        shaped = tuple(int(x) for x in kf.shaped)           # (separate samples, depth, rows, cols, contig samples)
        count = int(np.prod(shaped, dtype=np.int64))
        offset = int(page.dataoffsets[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if data is not None:
        buf = data.getbuffer() if isinstance(data, io.BytesIO) else data
        if offset + count * dtype.itemsize > len(buf):
            return None
        view = np.frombuffer(buf, dtype=dtype, count=count, offset=offset).reshape(shaped)
    else:
        view = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shaped)
    sub = np.ascontiguousarray(view[:, 0, ::step, ::step, :])  # (separate, rows, cols, contig)
    del view
    if sub.shape[0] > 1:                                     # planar: samples first
        sub = np.moveaxis(sub[..., 0], 0, -1)
    else:
        sub = sub[0]
    sub = sub.astype(sub.dtype.newbyteorder("="), copy=False)
    return sub[..., 0] if sub.shape[-1] == 1 else sub


def _strided_page(page, step: int) -> np.ndarray:
    """Every ``step``-th row and column of a TIFF page, decoded one strip
    or tile at a time (memory: one segment plus the output; the caller
    sends pages whose one segment is over the budget elsewhere, see
    _tiff_segment_bytes)."""
    kf = getattr(page, "keyframe", page)
    H, W = int(kf.imagelength), int(kf.imagewidth)
    spp = int(getattr(kf, "samplesperpixel", 1) or 1)
    planar = int(getattr(kf, "planarconfig", 1) or 1) == 2
    oh, ow = -(-H // step), -(-W // step)
    out = None
    for seg, idx, shape in page.segments(maxworkers=1):
        if seg is None:
            continue
        seg = np.asarray(seg)
        # idx: (separate sample, depth, row, column, contig sample) offsets
        s_i, _d, r0, c0 = int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3])
        seg = seg.reshape(shape) if seg.shape != tuple(shape) else seg
        # shape: (depth, length, width, contig samples)
        plane = seg[0]
        n_r = min(plane.shape[0], H - r0)
        n_c = min(plane.shape[1], W - c0)
        if n_r <= 0 or n_c <= 0:
            continue
        rs = (-r0) % step
        cs = (-c0) % step
        if rs >= n_r or cs >= n_c:
            continue
        sub = plane[rs:n_r:step, cs:n_c:step]
        if out is None:
            nch = spp if (spp > 1) else 1
            out = np.zeros((oh, ow, nch), dtype=sub.dtype)
        orr, occ = (r0 + rs) // step, (c0 + cs) // step
        if planar:
            out[orr:orr + sub.shape[0], occ:occ + sub.shape[1], s_i] = sub[..., 0]
        else:
            out[orr:orr + sub.shape[0], occ:occ + sub.shape[1], :] = sub.reshape(sub.shape[0], sub.shape[1], -1)
    if out is None:
        raise ValueError("no decodable segments")
    return out[..., 0] if out.shape[-1] == 1 else out


def _raster_facts(im: Image.Image, ext: str) -> dict:
    """Header facts from a PIL image without decoding pixels."""
    f: dict = {
        "format": im.format or ext.lstrip(".").upper(),
        "rows": im.height, "cols": im.width, "mode": im.mode,
        "frames": int(getattr(im, "n_frames", 1) or 1),
        "photometric": _MODE_PHOTOMETRIC.get(im.mode, im.mode),
        "bits": _MODE_BITS.get(im.mode),
    }
    if im.mode == "F":
        f["float"] = True
    fmt = (im.format or "").upper()
    if fmt == "JPEG" or ext in (".jpg", ".jpeg"):
        f["lossy"] = "01"
        f["lossy_method"] = "ISO_10918_1"
    elif fmt == "JPEG2000" or ext in (".jp2", ".j2k"):
        f["lossy_method"] = "ISO_15444_1"   # lossy or reversible: not known from the header
    elif fmt in ("PNG", "BMP", "GIF", "PPM"):
        f["lossy"] = "00"

    # TIFF tags: photometric (262), bits (258), compression (259), samples
    # (277), sample format (339), resolution.
    tags = getattr(im, "tag_v2", None)
    if tags is not None:
        try:
            if 262 in tags:
                f["photometric"] = _TIFF_PHOTOMETRIC.get(int(tags[262]), f["photometric"])
            if 258 in tags:
                bps = tags[258]
                f["bits"] = int(bps[0] if isinstance(bps, (tuple, list)) else bps)
            if 277 in tags:
                f["samples"] = int(tags[277])
            if 339 in tags:
                sf = tags[339]
                if int(sf[0] if isinstance(sf, (tuple, list)) else sf) == 3:
                    f["float"] = True
            if 259 in tags:
                comp = int(tags[259])
                f["lossy"] = "01" if comp in (6, 7) else "00"
                if comp in (6, 7):
                    f["lossy_method"] = "ISO_10918_1"
                elif comp in (33003, 33005, 34712):
                    f["lossy_method"] = "ISO_15444_1"
                    f.pop("lossy", None)
            desc = tags.get(270)
            if desc:
                f["description"] = str(desc)[:300]
            sp = _tiff_spacing(tags, str(desc or ""))
            if sp:
                f["spacing_mm"] = sp
        except Exception:  # noqa: BLE001 - malformed tags are common
            pass

    # EXIF / TIFF make, model, software, acquisition time.
    try:
        exif = im.getexif()
        if exif:
            if exif.get(271):
                f["make"] = str(exif.get(271)).strip("\x00 ")[:80]
            if exif.get(272):
                f["model"] = str(exif.get(272)).strip("\x00 ")[:80]
            if exif.get(305):
                f["software"] = str(exif.get(305)).strip("\x00 ")[:80]
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            dt = sub.get(36867) or sub.get(36868) or exif.get(306)
            if dt:
                f["acq_datetime"] = _exif_dt(str(dt))
    except Exception:  # noqa: BLE001
        pass
    # PNG text chunks.
    info = getattr(im, "info", {}) or {}
    if "Software" in info and "software" not in f:
        f["software"] = str(info["Software"])[:80]
    return {k: v for k, v in f.items() if v not in (None, "")}


def _tiff_spacing(tags, desc: str) -> list[float] | None:
    """PixelSpacing [row, col] in mm from TIFF resolution or ImageJ/OME text."""
    try:
        m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', desc)
        if m:
            x = float(m.group(1)) / 1000.0  # OME default unit is micrometre
            my = re.search(r'PhysicalSizeY="([\d.eE+-]+)"', desc)
            y = float(my.group(1)) / 1000.0 if my else x
            return [round(y, 6), round(x, 6)]
        xres, yres = tags.get(282), tags.get(283)
        if not xres or not yres:
            return None
        xres, yres = _ratio(xres), _ratio(yres)
        if xres <= 0 or yres <= 0:
            return None
        unit_m = re.search(r"unit=(\S+)", desc)  # ImageJ
        if unit_m:
            unit = unit_m.group(1).lower()
            scale = {"micron": 1e-3, "um": 1e-3, "µm": 1e-3, "mm": 1.0, "nm": 1e-6}.get(unit)
            if scale:
                return [round(scale / yres, 6), round(scale / xres, 6)]
            return None
        unit = int(tags.get(296, 2))
        if unit == 1 or xres in _SCREEN_DPI:
            return None
        per = 10.0 if unit == 3 else 25.4
        return [round(per / yres, 6), round(per / xres, 6)]
    except Exception:  # noqa: BLE001
        return None


def _ratio(v) -> float:
    """TIFF RATIONAL (PIL IFDRational, or tifffile (num, den) tuple) -> float."""
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return float(v[0]) / float(v[1]) if v[1] else 0.0
    return float(v)


def _exif_dt(s: str) -> str:
    """'YYYY:MM:DD HH:MM:SS' -> 'YYYYMMDDHHMMSS' (DICOM DT); '' when junk."""
    digits = re.sub(r"\D", "", s)[:14]
    return digits if len(digits) >= 8 and not digits.startswith("0000") else ""


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------
_DATA_IMG = re.compile(rb"""(?:xlink:)?href\s*=\s*["']data:image/(png|jpe?g|gif|bmp|webp|tiff?);base64,([^"']+)["']""",
                       re.I)
_HREF = re.compile(rb"""((?:xlink:)?href)\s*=\s*(["'])(.*?)\2""", re.I | re.S)


def _load_svg(data, path, ext, max_pixels: int = 80_000_000) -> Loaded:
    """Largest embedded raster image of an SVG (a figure that wraps a photo),
    else the SVG rendered at SVG_RENDER_PX wide on white. Rendering is
    sandboxed: every reference that is not an in-document fragment or an
    embedded data URI is removed first, and resources resolve in an empty
    directory, so nothing outside the file is read or fetched.

    Memory is bounded: a file over SVG_MAX_BYTES is refused before it is
    read, a gzip stream (.svgz) is decompressed only up to SVG_MAX_BYTES + 1
    bytes (a gzip bomb never expands whole), and an embedded image over
    ``max_pixels`` is never decoded (nor rendered by resvg, which would
    decode it)."""
    facts: dict = {"format": "SVG", "source_format": "SVG"}
    too_large = f"too_large: SVG over {SVG_MAX_BYTES >> 20} MB"
    if data is None:
        if Path(path).stat().st_size > SVG_MAX_BYTES:
            return Loaded(None, facts, too_large)
        raw = Path(path).read_bytes()
    else:
        raw = data
    if raw[:2] == b"\x1f\x8b":
        import gzip
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as g:
            raw = g.read(SVG_MAX_BYTES + 1)
    if len(raw) > SVG_MAX_BYTES:
        return Loaded(None, facts, too_large)
    best = None
    over = None
    for m in _DATA_IMG.finditer(raw):
        try:
            blob = base64.b64decode(re.sub(rb"\s+", b"", m.group(2)), validate=False)
            im = Image.open(io.BytesIO(blob))            # header only, nothing decoded yet
            if im.size[0] * im.size[1] > max_pixels:
                over = im.size
                continue
            if best is None or im.size[0] * im.size[1] > best.size[0] * best.size[1]:
                best = im
        except Exception:  # noqa: BLE001
            continue
    if over is not None and (best is None or min(best.size) < 32):
        return Loaded(None, facts, f"too_large: embedded image {over[0]}x{over[1]} over {max_pixels} pixels")
    if best is not None and min(best.size) >= 32:
        facts.update({"rows": best.height, "cols": best.width, "embedded_format": best.format})
        best.load()
        return _finish_image(to_rgb8(best), facts, f"SVG embedded {best.format} image")
    import resvg_py

    def keep_local(m):
        target = m.group(3).strip()
        if target.startswith(b"#") or target.lower().startswith(b"data:"):
            return m.group(0)
        return b""
    clean = _HREF.sub(keep_local, raw)
    clean = re.sub(rb"@import[^;]*;", b"", clean)
    with tempfile.TemporaryDirectory() as empty:
        png = resvg_py.svg_to_bytes(svg_string=clean.decode("utf-8", "replace"), width=SVG_RENDER_PX,
                                    background="#ffffff", resources_dir=empty)
    im = Image.open(io.BytesIO(bytes(png)))
    im.load()
    facts.update({"rows": im.height, "cols": im.width})
    return _finish_image(to_rgb8(im), facts, f"SVG rendered at {SVG_RENDER_PX} px (resvg)")


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------
def _load_dicom(data, path, max_pixels) -> Loaded:
    import pydicom
    src = io.BytesIO(data) if data is not None else str(path)
    ds = pydicom.dcmread(src, force=True, defer_size="4 MB")
    facts = {"format": "DICOM", "dicom": dicom_header(ds)}
    d = facts["dicom"]
    rows, cols = int(d.get("Rows") or 0), int(d.get("Columns") or 0)
    frames = int(d.get("NumberOfFrames") or 1)
    tsyntax = ""
    try:
        tsyntax = str(ds.file_meta.TransferSyntaxUID.name)
    except Exception:  # noqa: BLE001
        pass
    facts["source_format"] = "DICOM" + (f" ({tsyntax})" if tsyntax else "")
    facts.update({"rows": rows, "cols": cols, "frames": frames,
                  "photometric": d.get("PhotometricInterpretation"),
                  "bits": d.get("BitsStored") or d.get("BitsAllocated")})
    sp = d.get("PixelSpacing")
    # Only a well-formed [row, col] pair counts; a single value or a
    # 3-element list from a malformed file is dropped (it stays in "dicom").
    if isinstance(sp, (list, tuple)) and len(sp) == 2:
        try:
            facts["spacing_mm"] = [float(sp[0]), float(sp[1])]
        except (TypeError, ValueError):
            pass
    if d.get("Manufacturer"):
        facts["make"] = d["Manufacturer"]
    if d.get("ManufacturerModelName"):
        facts["model"] = d["ManufacturerModelName"]
    sw = d.get("SoftwareVersions")
    if sw:
        facts["software"] = "\\".join(sw) if isinstance(sw, list) else str(sw)
    if d.get("SamplesPerPixel"):
        facts["samples"] = int(d["SamplesPerPixel"])
    if d.get("BitsAllocated"):
        facts["bits_allocated"] = int(d["BitsAllocated"])
    if d.get("LossyImageCompression") in ("00", "01"):
        facts["lossy"] = d["LossyImageCompression"]
    if d.get("LossyImageCompressionMethod"):
        m = d["LossyImageCompressionMethod"]
        facts["lossy_method"] = "\\".join(m) if isinstance(m, list) else str(m)
    if "FloatPixelData" in ds or "DoubleFloatPixelData" in ds:
        facts["float"] = True
    if d.get("AcquisitionDateTime") or d.get("StudyDate"):
        facts["acq_datetime"] = str(d.get("AcquisitionDateTime") or d.get("StudyDate"))[:14]
    if "PixelData" not in ds and "FloatPixelData" not in ds:
        return Loaded(None, facts, "no pixel data")
    if rows * cols == 0:
        return Loaded(None, facts, "no dimensions")

    mid = frames // 2 if frames > 1 else None
    arr = None
    try:
        from pydicom.pixels import pixel_array as _pixel_array  # pydicom >= 3
        src2 = io.BytesIO(data) if data is not None else str(path)
        arr = _pixel_array(src2, index=mid)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        return Loaded(None, facts, f"pixel decode: {type(e).__name__}: {e}"[:200])
    if arr is None:
        # pydicom 2.x: ds.pixel_array decodes every frame to take one
        spp = int(d.get("SamplesPerPixel") or 1)
        bpp = max(1, (int(d.get("BitsAllocated") or 8) + 7) // 8)
        if rows * cols * frames * spp * bpp > DECODE_MAX_BYTES:
            return Loaded(None, facts, f"too_large: {frames} frames of {rows}x{cols} over "
                                       f"{DECODE_MAX_BYTES >> 20} MB decoded (needs pydicom 3 for one frame)")
        full = ds.pixel_array
        arr = full[mid] if (mid is not None and full.ndim >= 3 and full.shape[0] == frames) else full
    if str(d.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    arr = np.asarray(arr)
    conv = "pydicom" + (f" frame {mid}/{frames}" if mid is not None else "")
    if arr.shape[0] * arr.shape[1] > max_pixels:
        step = math.ceil(math.sqrt(arr.shape[0] * arr.shape[1] / STRIDED_TARGET_PIXELS))
        arr = arr[::step, ::step]
        conv += f" strided 1/{step}"
    return _from_array(arr, facts, conv)


def dicom_header(ds) -> dict:
    """Keep-list DICOM attributes as plain JSON values (no identifiers)."""
    out: dict = {}
    for kw in DICOM_KEEP:
        v = ds.get(kw)
        if v is None:
            continue
        v = getattr(v, "value", v)
        if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
            try:
                v = [float(x) if kw == "PixelSpacing" else str(x) for x in v]
            except (TypeError, ValueError):
                v = [str(x) for x in v]
        elif isinstance(v, (int, float)):
            pass
        else:
            v = str(v)
        if v not in ("", []):
            out[kw] = v
    for kw in DICOM_KEEP_SEQUENCES:
        seq = ds.get(kw)
        if seq is None:
            continue
        try:
            item = seq.value[0] if hasattr(seq, "value") else seq[0]
            out[kw] = " ".join(str(item.get(t, "")) for t in
                               ("CodingSchemeDesignator", "CodeValue", "CodeMeaning")).strip()
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------
def _load_volume(path: Path, ext: str, max_pixels: int, max_bytes: int = 2 << 30,
                 companion: Path | None = None) -> Loaded:
    """Middle slice of a NIfTI / NRRD / MHA volume, or of a header + data
    pair (MHD + RAW through SimpleITK, Analyze HDR + IMG through nibabel,
    detached NRRD). The pair's files must sit in one directory under their
    own names (the walker and the remote sampler arrange that).

    ``max_bytes`` caps the UNCOMPRESSED voxel size (from the header, before
    any pixel is read): a gzip NRRD or compressed MHA far smaller on disk can
    expand to many GB, and on a small VM that ends in the OOM killer rather
    than a catchable MemoryError. Only the middle slice is materialised where
    the reader allows it (nibabel proxy slicing, SimpleITK extract region).
    """
    fmt = {".hdr": "ANALYZE", ".mhd": "MHD", ".nhdr": "NRRD (detached)"}.get(ext, ext.lstrip(".").upper())
    facts: dict = {"format": ext.lstrip(".").upper(), "source_format": fmt}
    if ext in (".mhd", ".hdr", ".nhdr") and companion is None:
        from .constants import volume_companion
        sib = volume_companion(path.name, [p.name for p in path.parent.iterdir()])
        if sib is None:
            return Loaded(None, facts, "volume data file (companion of the header) not available")
    if ext in (".nii", ".nii.gz", ".hdr"):
        arr = _nifti_slice(path, facts, max_pixels, max_bytes)
        reader = "nibabel"
    else:
        try:
            arr = _itk_slice(path, facts, max_pixels, max_bytes)
            reader = "SimpleITK"
        except Exception as e:  # noqa: BLE001 - ITK cannot read every NRRD
            if ext not in (".nrrd", ".nhdr"):
                raise
            logger.debug("SimpleITK failed on %s (%s); trying pynrrd", path, e)
            arr = _pynrrd_slice(path, facts, max_pixels, max_bytes)
            reader = "pynrrd"
    if arr is None:
        return Loaded(None, facts, facts.pop("_error", "volume not loaded"))
    arr = np.asarray(arr)
    while arr.ndim > 2 and not (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
        arr = arr[..., 0]
    facts["rows"], facts["cols"] = int(arr.shape[0]), int(arr.shape[1])
    facts["bits"] = int(arr.dtype.itemsize * 8)
    if np.issubdtype(arr.dtype, np.floating):
        facts["float"] = True
    facts["photometric"] = "RGB" if arr.ndim == 3 else "MONOCHROME2"
    conv = f"{reader} middle slice"
    if arr.shape[0] * arr.shape[1] > max_pixels:
        step = math.ceil(math.sqrt(arr.shape[0] * arr.shape[1] / STRIDED_TARGET_PIXELS))
        arr = arr[::step, ::step]
        conv += f" strided 1/{step}"
    return _from_array(arr, facts, conv)


def _refuse(facts: dict, msg: str) -> None:
    facts["_error"] = msg
    return None


def _slice_pixels(shape: tuple[int, ...], axis: int | None) -> int:
    """Pixels of the 2D image that will be materialised."""
    spatial = list(shape[:3]) if axis is not None else list(shape[:2])
    if axis is not None:
        spatial.pop(axis)
    return int(np.prod(spatial[:2], dtype=np.int64))


def _nifti_slice(path: Path, facts: dict, max_pixels: int, max_bytes: int):
    import nibabel as nib
    img = nib.load(str(path))
    shape = tuple(int(s) for s in img.shape)
    facts["shape"] = list(shape)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    axis, idx = _pick_slice(shape)
    if axis is None:
        # Scaled NIfTI data comes back as float64: budget 8 bytes per value.
        if int(np.prod(shape, dtype=np.int64)) * 8 > max_bytes:
            step = math.ceil(math.sqrt(_slice_pixels(shape, None) / STRIDED_TARGET_PIXELS))
            arr = np.asarray(img.dataobj[::step, ::step])
        else:
            arr = np.asarray(img.dataobj)
        facts["frames"] = 1
        if len(zooms) >= 2:
            facts["spacing_mm"] = [round(zooms[0], 6), round(zooms[1], 6)]
        return arr
    sl: list = [slice(None)] * len(shape)
    sl[axis] = idx
    for extra in range(3, len(shape)):
        sl[extra] = 0
    if _slice_pixels(shape, axis) > max_pixels:
        step = math.ceil(math.sqrt(_slice_pixels(shape, axis) / STRIDED_TARGET_PIXELS))
        sl = [s if isinstance(s, int) else slice(None, None, step) for s in sl]
    arr = np.asarray(img.dataobj[tuple(sl)])  # proxy slicing: only this slice is read
    facts["frames"] = shape[axis]
    keep = [i for i in range(3) if i != axis]
    if len(zooms) >= 3:
        facts["spacing_mm"] = [round(zooms[keep[0]], 6), round(zooms[keep[1]], 6)]
    return arr


def _itk_bytes_per_component(pixel_id_string: str) -> int:
    """'16-bit unsigned integer' -> 2; unknown -> 8 (conservative)."""
    m = re.search(r"(\d+)-bit", pixel_id_string or "")
    return max(1, int(m.group(1)) // 8) if m else 8


def _itk_slice(path: Path, facts: dict, max_pixels: int, max_bytes: int):
    """MHA / MHD / NRRD through SimpleITK: header first, then only the middle slice."""
    import SimpleITK as sitk  # noqa: N813
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size = [int(s) for s in reader.GetSize()]          # ITK order: x, y, z, (t)
    spacing = [float(s) for s in reader.GetSpacing()]
    ncomp = int(reader.GetNumberOfComponents())
    bpc = _itk_bytes_per_component(sitk.GetPixelIDValueAsString(reader.GetPixelID()))
    np_shape = tuple(reversed(size)) + ((ncomp,) if ncomp > 1 else ())
    facts["shape"] = list(np_shape)
    total = int(np.prod(size, dtype=np.int64)) * ncomp * bpc
    if len(size) < 2:
        raise ValueError(f"unsupported shape {tuple(size)}")
    if len(size) == 2:
        if total > max_bytes:
            return _refuse(facts, "volume too large to load")
        arr = sitk.GetArrayFromImage(reader.Execute())
        facts["frames"] = 1
        facts["spacing_mm"] = [round(spacing[1], 6), round(spacing[0], 6)]
        return arr
    # Middle slice along the shortest of the first three spatial dims.
    d = int(np.argmin(size[:3]))
    keep = [i for i in range(3) if i != d]                # ITK dims kept (x-like first)
    # Compressed data may be decompressed whole by the reader even when only
    # a region is extracted, so the full uncompressed size is capped too.
    if total > max_bytes:
        return _refuse(facts, "volume too large to load")
    index = [0] * len(size)
    ext_size = list(size)
    index[d] = size[d] // 2
    ext_size[d] = 0
    for extra in range(3, len(size)):
        ext_size[extra] = 0
    reader.SetExtractIndex(index)
    reader.SetExtractSize(ext_size)
    arr = sitk.GetArrayFromImage(reader.Execute())   # 2D (rows, cols[, comp])
    facts["frames"] = size[d]
    # numpy rows follow the higher kept ITK dim, columns the lower one.
    facts["spacing_mm"] = [round(spacing[keep[1]], 6), round(spacing[keep[0]], 6)]
    return arr


def _pynrrd_slice(path: Path, facts: dict, max_pixels: int, max_bytes: int):
    """Fallback NRRD reader (whole volume, so the uncompressed cap matters)."""
    import nrrd
    hdr = nrrd.read_header(str(path))
    sizes = [int(s) for s in hdr.get("sizes") or []]    # fastest axis first
    try:
        itemsize = np.dtype(nrrd.reader._determine_datatype(hdr)).itemsize
    except Exception:  # noqa: BLE001 - private helper; be conservative
        itemsize = 8
    shape = tuple(reversed(sizes))                       # C order, as loaded below
    facts["shape"] = list(shape)
    axis, idx = _pick_slice(shape)
    if int(np.prod(sizes, dtype=np.int64)) * itemsize > max_bytes:
        return _refuse(facts, "volume too large to load")
    vol, hdr = nrrd.read(str(path), index_order="C")
    arr = vol if axis is None else np.take(vol, idx, axis=axis)
    del vol
    facts["frames"] = 1 if axis is None else shape[axis]
    sp = hdr.get("spacings")
    if sp is None and hdr.get("space directions") is not None:
        try:
            sp = [float(np.linalg.norm(v)) if v is not None and np.all(np.isfinite(v)) else float("nan")
                  for v in hdr["space directions"]]
        except Exception:  # noqa: BLE001
            sp = None
    if axis is not None and sp is not None and len(sp) >= 3:
        sp_c = list(reversed([float(s) for s in sp]))   # header is fastest-first
        keep = [i for i in range(3) if i != axis]
        if np.isfinite(sp_c[keep[0]]) and np.isfinite(sp_c[keep[1]]):
            facts["spacing_mm"] = [round(sp_c[keep[0]], 6), round(sp_c[keep[1]], 6)]
    return arr


def _pick_slice(shape: tuple[int, ...]) -> tuple[int | None, int]:
    """Axis and index of the middle slice along the shortest spatial axis.

    Returns (None, 0) when the array is already one 2D image (HxW or HxWx3/4).
    Raises ValueError for arrays with fewer than 2 dimensions.
    """
    if len(shape) < 2:
        raise ValueError(f"unsupported shape {shape}")
    if len(shape) == 2:
        return None, 0
    if len(shape) == 3 and shape[-1] in (3, 4) and min(shape[:2]) > 8:
        return None, 0
    spatial = shape[:3]
    axis = int(np.argmin(spatial))
    return axis, spatial[axis] // 2


# ---------------------------------------------------------------------------
# numeric arrays (NumPy, HDF5 / MAT v7.3 / Imaris, MAT v5)
# ---------------------------------------------------------------------------
def plane_plan(shape: tuple[int, ...]) -> tuple[tuple, tuple[int, int]] | None:
    """Index to take one 2D image (with channels when the last axis has 3 or
    4) out of an array of ``shape``, and that image's (rows, cols); None when
    the array is not image shaped.

    Size-1 axes are dropped (index 0). With channels last (3 or 4) the rest
    must be 2D, or 3D (middle slice along its shortest axis). Otherwise the
    last three axes hold the image: the middle slice along the shortest of
    them; extra leading axes take index 0. The image must have both sides of
    at least ARRAY_MIN_SIDE and an aspect ratio of at most ARRAY_MAX_ASPECT
    (signals, tables and feature matrices are not images)."""
    if len(shape) < 2:
        return None
    idx: list = [slice(None)] * len(shape)
    live = [i for i, n in enumerate(shape) if n > 1]
    for i, n in enumerate(shape):
        if n == 1:
            idx[i] = 0
    if len(live) < 2:
        return None
    if len(live) >= 3 and shape[live[-1]] in (3, 4):
        live = live[:-1]                  # channels last: kept whole
    while len(live) > 3:
        idx[live[0]] = 0
        live = live[1:]
    if len(live) == 3:
        ax = min(live, key=lambda i: (shape[i], i))
        idx[ax] = shape[ax] // 2
        live = [i for i in live if i != ax]
    r, c = shape[live[0]], shape[live[1]]
    if min(r, c) < ARRAY_MIN_SIDE or max(r, c) > ARRAY_MAX_ASPECT * min(r, c):
        return None
    return tuple(idx), (int(r), int(c))


def _strided(idx: tuple, plane: tuple[int, int], max_pixels: int) -> tuple[tuple, int]:
    """Add a stride to the two image axes of ``idx`` when the plane is over
    the pixel budget."""
    r, c = plane
    if r * c <= max_pixels:
        return idx, 1
    step = math.ceil(math.sqrt(r * c / min(max_pixels, STRIDED_TARGET_PIXELS)))
    out, n = [], 0
    for s in idx:
        if isinstance(s, slice) and n < 2:
            out.append(slice(None, None, step))
            n += 1
        else:
            out.append(s)
    return tuple(out), step


def _choose_dataset(cands: list[tuple[str, tuple, np.dtype]]):
    """Best image-shaped candidate: name hint (img, image, oct, bscan, vol,
    fundus, slo, frame) first, then the largest plane. Returns
    (name, index, plane) or None."""
    best = None
    for name, shape, dtype in cands:
        try:
            dt = np.dtype(dtype)
        except TypeError:
            continue
        if not (np.issubdtype(dt, np.number) or dt == np.bool_) or np.issubdtype(dt, np.complexfloating):
            continue
        if ARRAY_PARAM_NAME.search(name):
            continue                      # a network layer's parameters
        plan = plane_plan(tuple(int(s) for s in shape))
        if plan is None:
            continue
        idx, plane = plan
        base = name.rsplit("/", 1)[-1]
        score = (bool(ARRAY_NAME_HINT.search(base)), plane[0] * plane[1])
        if best is None or score > best[0]:
            best = (score, name, idx, plane)
    return None if best is None else best[1:]


def _load_array(path: Path, ext: str, max_pixels: int) -> Loaded:
    facts: dict = {"format": ext.lstrip(".").upper(), "source_format": ext.lstrip(".").upper()}
    with open(path, "rb") as fp:
        head = fp.read(520)
    is_h5 = head[:8] == b"\x89HDF\r\n\x1a\n" or head[512:520] == b"\x89HDF\r\n\x1a\n"
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        got = _choose_dataset([("array", arr.shape, arr.dtype)])
        if got is None:
            return Loaded(None, facts, f"not image shaped: {tuple(arr.shape)} {arr.dtype}")
        name, idx, plane = got
        idx, step = _strided(idx, plane, max_pixels)
        return _array_result(np.asarray(arr[idx]), facts, f"npy {tuple(arr.shape)}", step)
    if ext == ".npz":
        return _load_npz(path, facts, max_pixels)
    if is_h5:
        facts["source_format"] = "MAT v7.3 (HDF5)" if ext == ".mat" else f"HDF5 ({ext})"
        return _load_h5(path, facts, max_pixels)
    if ext == ".mat":
        facts["source_format"] = "MAT v5"
        return _load_mat5(path, facts, max_pixels)
    return Loaded(None, facts, "not an HDF5 or NumPy file")


def _array_result(arr: np.ndarray, facts: dict, what: str, step: int) -> Loaded:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    facts["rows"], facts["cols"] = int(arr.shape[0]), int(arr.shape[1])
    facts["bits"] = int(arr.dtype.itemsize * 8)
    if np.issubdtype(arr.dtype, np.floating):
        facts["float"] = True
    conv = what + (f" strided 1/{step}" if step > 1 else "")
    return _from_array(arr, facts, conv)


def _npy_header(fp):
    """(shape, fortran_order, dtype) of a .npy stream, positioned at its data."""
    fmt = np.lib.format
    version = fmt.read_magic(fp)
    if version == (1, 0):
        return fmt.read_array_header_1_0(fp)
    if version == (2, 0):
        return fmt.read_array_header_2_0(fp)
    return fmt._read_array_header(fp, version)


def _load_npz(path: Path, facts: dict, max_pixels: int) -> Loaded:
    cands = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as fp:
                try:
                    shape, fortran, dtype = _npy_header(fp)
                except Exception:  # noqa: BLE001
                    continue
            cands.append((info.filename[:-4], shape, dtype))
        got = _choose_dataset(cands)
        if got is None:
            return Loaded(None, facts, f"no image-shaped array among {len(cands)}")
        name, idx, plane = got
        info = zf.getinfo(name + ".npy")
        dtype = np.dtype(next(d for n, s, d in cands if n == name))
        shape = next(s for n, s, d in cands if n == name)
        if int(np.prod(shape, dtype=np.int64)) * dtype.itemsize > DECODE_MAX_BYTES:
            if info.compress_type == zipfile.ZIP_STORED:
                # np.savez stores members uncompressed: map the member in place.
                with open(path, "rb") as fp:
                    fp.seek(info.header_offset)
                    lh = fp.read(30)
                    n_len, x_len = struct.unpack("<HH", lh[26:30])
                    start = info.header_offset + 30 + n_len + x_len
                    fp.seek(start)
                    _npy_header(fp)
                    data_off = fp.tell()
                arr = np.memmap(path, dtype=dtype, mode="r", offset=data_off, shape=tuple(shape))
            else:
                return Loaded(None, facts, f"too_large: compressed npz member {name} over {DECODE_MAX_BYTES >> 20} MB")
        else:
            with zf.open(info) as fp:
                arr = np.lib.format.read_array(fp, allow_pickle=False)
    idx, step = _strided(idx, plane, max_pixels)
    return _array_result(np.asarray(arr[idx]), facts, f"npz {name} {tuple(shape)}", step)


def _load_h5(path: Path, facts: dict, max_pixels: int) -> Loaded:
    import h5py
    cands: list = []
    with h5py.File(path, "r") as f:
        if any(a in f.attrs for a in H5_MODEL_ATTRS) or "model_weights" in f:
            return Loaded(None, facts, "no image-shaped dataset: model weights file (Keras HDF5)")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and len(cands) < 5000:
                cands.append((name, obj.shape, obj.dtype))
        f.visititems(visit)
        got = _choose_dataset(cands)
        if got is None:
            return Loaded(None, facts, f"no image-shaped dataset among {len(cands)}")
        name, idx, plane = got
        ds = f[name]
        idx, step = _strided(idx, plane, max_pixels)
        arr = ds[idx]
        shape = tuple(ds.shape)
    # MATLAB v7.3 stores arrays transposed (column-major): the image is the
    # transpose, which only matters for orientation, not for the class.
    return _array_result(arr, facts, f"hdf5 {name} {shape}", step)


def _load_mat5(path: Path, facts: dict, max_pixels: int) -> Loaded:
    import scipy.io
    _MAT_TYPES = {"double": "f8", "single": "f4", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
                  "int32": "i4", "uint32": "u4", "int64": "i8", "uint64": "u8", "logical": "u1"}
    info = scipy.io.whosmat(str(path))
    cands = [(n, s, _MAT_TYPES[c]) for n, s, c in info if c in _MAT_TYPES]
    got = _choose_dataset(cands)
    if got is None:
        return Loaded(None, facts, f"no image-shaped variable among {len(info)}")
    name, idx, plane = got
    shape, code = next((s, c) for n, s, c in cands if n == name)
    if int(np.prod(shape, dtype=np.int64)) * np.dtype(code).itemsize > DECODE_MAX_BYTES:
        return Loaded(None, facts, f"too_large: MAT v5 variable {name} {tuple(shape)} over {DECODE_MAX_BYTES >> 20} MB "
                                   "(MAT v5 has no partial read)")
    arr = scipy.io.loadmat(str(path), variable_names=[name])[name]
    idx, step = _strided(idx, plane, max_pixels)
    return _array_result(np.asarray(arr)[idx], facts, f"mat {name} {tuple(shape)}", step)


# ---------------------------------------------------------------------------
# microscopy
# ---------------------------------------------------------------------------
def _load_microscopy(path: Path, ext: str, max_pixels: int) -> Loaded:
    facts: dict = {"format": ext.lstrip(".").upper(), "source_format": ext.lstrip(".").upper()}
    if ext == ".czi":
        from pylibCZIrw import czi as pyczi
        with pyczi.open_czi(str(path)) as doc:
            box = doc.total_bounding_box
            rect = doc.total_bounding_rectangle
            plane = {a: (lo + (hi - lo) // 2 if a == "Z" else lo) for a, (lo, hi) in box.items()
                     if a in ("Z", "C", "T")}
            w, h = int(rect.w), int(rect.h)
            zoom = min(1.0, math.sqrt(min(max_pixels, STRIDED_TARGET_PIXELS) / max(1, w * h)))
            arr = doc.read(plane=plane, zoom=zoom)
        facts.update({"full_rows": h, "full_cols": w})
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[..., ::-1]                      # BGR -> RGB
        return _array_result(arr, facts, f"pylibCZIrw plane {plane} zoom {zoom:.3g}", 1)
    if ext == ".lif":
        from readlif.reader import LifFile
        img = LifFile(str(path)).get_image(0)
        z = int(img.dims.z) // 2 if getattr(img.dims, "z", 1) else 0
        frame = img.get_frame(z=z, t=0, c=0)
        return _array_result(np.asarray(frame), facts, f"readlif image 0 z={z}", 1)
    if ext == ".nd2":
        import nd2
        with nd2.ND2File(str(path)) as f:
            sizes = dict(f.sizes)
            darr = f.to_dask()
            idx = []
            for a in sizes:
                if a in ("Y", "X", "S"):
                    idx.append(slice(None))
                else:
                    idx.append(sizes[a] // 2 if a == "Z" else 0)
            y, x = sizes.get("Y", 1), sizes.get("X", 1)
            step = 1
            if y * x > max_pixels:
                step = math.ceil(math.sqrt(y * x / STRIDED_TARGET_PIXELS))
                idx = [slice(None, None, step) if (a in ("Y", "X")) else i for a, i in zip(sizes, idx)]
            arr = np.asarray(darr[tuple(idx)].compute())
        return _array_result(arr, facts, f"nd2 {sizes}", step)
    if ext == ".mrc":
        import mrcfile
        with mrcfile.mmap(str(path), mode="r", permissive=True) as m:
            data = m.data
            plan = plane_plan(tuple(data.shape))
            if plan is None:
                return Loaded(None, facts, f"not image shaped: {data.shape}")
            idx, step = _strided(plan[0], plan[1], max_pixels)
            arr = np.array(data[idx])
        return _array_result(arr, facts, f"mrcfile {tuple(data.shape)}", step)
    if ext == ".oib":
        if path.stat().st_size > DECODE_MAX_BYTES:
            return Loaded(None, facts, "too_large: OIB over the decode budget (no partial read)")
        import oiffile
        arr = np.asarray(oiffile.imread(str(path)))
        plan = plane_plan(arr.shape)
        if plan is None:
            return Loaded(None, facts, f"not image shaped: {arr.shape}")
        return _array_result(arr[plan[0]], facts, f"oiffile {arr.shape}", 1)
    return Loaded(None, facts, "no reader for this microscopy format")


# ---------------------------------------------------------------------------
# vendor OCT
# ---------------------------------------------------------------------------
_VENDOR_MAKE = {".e2e": "Heidelberg Engineering", ".vol": "Heidelberg Engineering", ".sdb": "Heidelberg Engineering",
                ".fds": "Topcon", ".fda": "Topcon"}


def _load_vendor(path: Path, ext: str, max_pixels: int) -> Loaded:
    """Middle B-scan of a proprietary OCT file, plus its fundus / SLO image
    as a separate view. Readers: oct-converter (E2E, FDS, FDA, Bioptigen
    OCT), the survey's own Heidelberg VOL and Thorlabs OCT readers."""
    facts: dict = {"format": ext.lstrip(".").upper(), "source_format": ext.lstrip(".").upper()}
    if ext in _VENDOR_MAKE:
        facts["make"] = _VENDOR_MAKE[ext]
    size = path.stat().st_size
    if size > VENDOR_MAX_BYTES:
        return Loaded(None, facts, f"too_large: vendor file over {VENDOR_MAX_BYTES / 2**30:g} GiB (the readers load it whole)")
    with open(path, "rb") as fp:
        head = fp.read(16)
    bscan = fundus = None
    conv = ""
    if ext == ".vol" or head.startswith(b"HSF-OCT"):
        bscan, fundus, n = _read_heidelberg_vol(path)
        facts.update({"source_format": "Heidelberg VOL", "make": "Heidelberg Engineering", "frames": n})
        conv = f"VOL reader middle B-scan {n // 2}/{n}"
    elif ext == ".oct" and head[:2] == b"PK":
        bscan, fundus, n, how = _read_thorlabs_oct(path)
        facts.update({"source_format": "Thorlabs OCT", "make": "Thorlabs", "frames": n})
        conv = f"Thorlabs OCT reader {how}"
    elif ext in (".fds", ".fda") and b"FOCT" not in head:
        # .fds is also the Fire Dynamics Simulator input format (text)
        return Loaded(None, facts, "not a Topcon FDS/FDA file (no FOCT signature): wrong_format")
    elif ext in (".e2e", ".fds", ".fda", ".oct"):
        from oct_converter import readers
        if ext == ".e2e":
            r = readers.E2E(str(path))
            vols = [v for v in r.read_oct_volume() if v is not None and len(v.volume)]
            try:
                funds = [f for f in r.read_fundus_image() if f is not None]
            except Exception:  # noqa: BLE001
                funds = []
            facts["source_format"] = "Heidelberg E2E"
        elif ext in (".fds", ".fda"):
            r = readers.FDS(str(path)) if ext == ".fds" else readers.FDA(str(path))
            vols = [r.read_oct_volume()]
            try:
                funds = [r.read_fundus_image()]
            except Exception:  # noqa: BLE001
                funds = []
            facts["source_format"] = "Topcon " + ext.lstrip(".").upper()
        else:
            try:
                vols = readers.BOCT(str(path)).read_oct_volume()
                facts.update({"source_format": "Bioptigen OCT", "make": "Bioptigen"})
            except Exception:  # noqa: BLE001
                vols = readers.POCT(str(path)).read_oct_volume()
                facts.update({"source_format": "Optovue OCT", "make": "Optovue"})
            funds = []
        vols = [v for v in vols if v is not None and len(v.volume)]
        if not vols:
            return Loaded(None, facts, "no OCT volume in the file")
        vol = max(vols, key=lambda v: len(v.volume))
        n = len(vol.volume)
        bscan = vol.volume[n // 2]
        facts["frames"] = n
        if vol.laterality in ("L", "R"):
            facts["laterality"] = vol.laterality
        funds = [f for f in funds if f is not None and getattr(f, "image", None) is not None]
        fundus = funds[0].image if funds else None
        conv = f"oct-converter middle B-scan {n // 2}/{n}"
    else:
        return Loaded(None, facts, "proprietary format without an open reader (catalogued)")
    if bscan is None:
        return Loaded(None, facts, "no B-scan decoded")
    bscan = np.asarray(bscan)
    facts["rows"], facts["cols"] = int(bscan.shape[0]), int(bscan.shape[1])
    out = _from_array(bscan, facts, conv)
    if out.image is not None and fundus is not None:
        fimg = array_to_rgb8(np.asarray(fundus))
        if fimg is not None:
            ff: dict = {"conversion": f"{conv.split(' middle')[0]} fundus / SLO image",
                        "rows": fimg.height, "cols": fimg.width}
            _pixel_flags(fimg, ff)
            out.views.append(("fundus", fimg, ff))
    return out


def _read_heidelberg_vol(path: Path):
    """Heidelberg Spectralis .vol (HSF-OCT): 2048-byte header, the SLO
    image, then B-scans as float32 with their own headers. Returns (middle
    B-scan, SLO, number of B-scans). Values are shown on the vendor's
    fourth-root scale; 3.4e38 marks invalid pixels."""
    with open(path, "rb") as fp:
        hdr = fp.read(2048)
        if not hdr.startswith(b"HSF-OCT"):
            raise ValueError("not a Heidelberg VOL file")
        size_x, n_bscans, size_z = struct.unpack("<3i", hdr[12:24])
        size_x_slo, size_y_slo = struct.unpack("<2i", hdr[48:56])
        bscan_hdr = struct.unpack("<i", hdr[100:104])[0]
        if not (0 < size_x <= 8192 and 0 < size_z <= 8192 and 0 < n_bscans <= 4096):
            raise ValueError("implausible VOL dimensions")
        slo = np.frombuffer(fp.read(size_x_slo * size_y_slo), np.uint8)
        slo = slo.reshape(size_y_slo, size_x_slo) if slo.size == size_x_slo * size_y_slo else None
        mid = n_bscans // 2
        fp.seek(2048 + size_x_slo * size_y_slo + mid * (bscan_hdr + size_x * size_z * 4) + bscan_hdr)
        b = np.frombuffer(fp.read(size_x * size_z * 4), "<f4").reshape(size_z, size_x).copy()
    b[b > 1e30] = 0
    b = np.power(np.clip(b, 0, None), 0.25)
    return b, slo, n_bscans


THORLABS_MAX_ASCANS = 1024       # A-scans read from a raw spectral file to build one B-scan


def _read_thorlabs_oct(path: Path):
    """Thorlabs .oct: a zip with Header.xml and data\\*.data members (member
    names use backslashes). Returns (B-scan, video image, B-scans in the
    file, how the B-scan was made).

    * An ``Intensity`` data file (Type Real, float32, SizeY B-scans of SizeX
      A-scans of SizeZ values, z fastest) gives its middle B-scan.
    * Otherwise the raw spectra (``Spectral<n>``, Type Raw, uint16, SizeX
      A-scans of SizeZ samples) of the middle spectral file are turned into
      a B-scan: up to THORLABS_MAX_ASCANS A-scans from its middle, the mean
      spectrum subtracted, a Hann window, an FFT along z and the log
      magnitude of the first half (no chirp or dispersion correction: a
      rough image, enough to tell the modality).
    * ``VideoImage`` (Type Colored, SizeX rows of SizeZ pixels, 4 bytes per
      pixel) is the camera image shown next to the scan: the fundus-like
      view. Read as interleaved BGRA when every fourth byte (alpha) is
      opaque (checked on a real file: 640 rows of 480 pixels, alpha 253 to
      255), else as its first gray plane."""
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as zf:
        names = {n.replace("\\", "/").lower(): n for n in zf.namelist()}
        header = names.get("header.xml")
        if header is None:
            raise ValueError("no Header.xml: not a Thorlabs OCT file")
        root = ET.fromstring(zf.read(header))
        entries = []
        for df in root.iter("DataFile"):
            rel = (df.text or "").strip()
            member = names.get(rel.replace("\\", "/").lower())
            if member is None:
                continue
            dims = {k: int(float(df.get(k) or 0)) for k in ("SizeZ", "SizeX", "SizeY", "BytesPerPixel")}
            entries.append(((df.get("Type") or "").lower(), rel.replace("\\", "/").rsplit("/", 1)[-1].lower(),
                            member, dims))
        bscan = video = None
        n, how = 1, ""
        for typ, base, member, d in entries:
            if base.startswith("videoimage") and d["SizeZ"] and d["SizeX"]:
                # SizeZ is the row length, SizeX the row count (a 480 x 640
                # camera frame is stored as 640 rows of 480 BGRA pixels)
                w, h = d["SizeZ"], d["SizeX"]
                raw = zf.read(member)
                bpp = d["BytesPerPixel"] or (4 if len(raw) >= w * h * 4 else 3)
                buf = np.frombuffer(raw[: w * h * bpp], np.uint8)
                if len(buf) == w * h * 4 and bpp == 4 and int(buf[3::4].min()) >= 250:
                    video = buf.reshape(h, w, 4)[..., [2, 1, 0]]          # interleaved BGRA, opaque alpha
                elif len(buf) >= w * h:
                    # planar (or 8-bit) camera data, as real files store it:
                    # the first w x h bytes are one gray plane
                    video = buf[: w * h].reshape(h, w)
        intensity = [e for e in entries if e[1].startswith("intensity") and e[3]["SizeZ"] and e[3]["SizeX"]]
        spectral = sorted((e for e in entries if e[1].startswith("spectral") and e[0] == "raw"
                           and e[3]["SizeZ"] and e[3]["SizeX"]),
                          key=lambda e: int(re.sub(r"\D", "", e[1]) or 0))
        if intensity:
            _, _, member, d = intensity[0]
            sz, sx, sy = d["SizeZ"], d["SizeX"], max(1, d["SizeY"])
            n = sy
            plane = sz * sx * 4
            with zf.open(member) as fp:
                fp.seek(plane * (sy // 2))
                raw = fp.read(plane)
            if len(raw) == plane:
                bscan = np.frombuffer(raw, "<f4").reshape(sx, sz).T
                how = f"intensity B-scan {sy // 2}/{sy}"
        elif spectral:
            _, _, member, d = spectral[len(spectral) // 2]
            sz, sx = d["SizeZ"], d["SizeX"]
            bpp = d["BytesPerPixel"] or 2
            k = min(THORLABS_MAX_ASCANS, sx)
            n = len(spectral)
            with zf.open(member) as fp:
                fp.seek(((sx - k) // 2) * sz * bpp)
                raw = fp.read(k * sz * bpp)
            k = len(raw) // (sz * bpp)
            if k:
                spec = np.frombuffer(raw[: k * sz * bpp], "<u2" if bpp == 2 else "<f4").reshape(k, sz)
                spec = spec.astype(np.float32)
                spec -= spec.mean(axis=0, keepdims=True)
                spec *= np.hanning(sz).astype(np.float32)
                mag = np.abs(np.fft.rfft(spec, axis=1))[:, 1: sz // 2]
                bscan = (20 * np.log10(mag + 1e-6)).T
                how = f"raw spectra FFT ({k} A-scans of spectral file {len(spectral) // 2}/{len(spectral)})"
    return bscan, video, n, how


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
VIDEO_POSITIONS = (0.5, 0.25, 0.75)      # first is the main image


def _load_video(path: Path, ext: str) -> Loaded:
    """Frames at 25, 50 and 75% of the duration (bundled ffmpeg through
    imageio-ffmpeg); the runner averages their class probabilities."""
    import imageio_ffmpeg
    facts: dict = {"format": ext.lstrip(".").upper(), "source_format": f"video {ext.lstrip('.').upper()}"}
    gen = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(gen)
    finally:
        gen.close()
    duration = float(meta.get("duration") or 0.0)
    w, h = (int(x) for x in meta.get("size") or (0, 0))
    facts.update({"rows": h, "cols": w, "duration_s": round(duration, 2), "fps": meta.get("fps"),
                  "codec": str(meta.get("codec") or "")[:40]})
    frames = []
    for pos in VIDEO_POSITIONS if duration > 0 else (0.0,):
        t = duration * pos
        g = imageio_ffmpeg.read_frames(str(path), input_params=["-ss", f"{t:.3f}"],
                                       output_params=["-frames:v", "1"])
        try:
            m = next(g)
            fw, fh = (int(x) for x in m.get("size") or (w, h))
            raw = next(g, None)
        finally:
            g.close()
        if raw is not None and len(raw) >= fw * fh * 3:
            frames.append(Image.fromarray(np.frombuffer(raw, np.uint8)[: fw * fh * 3].reshape(fh, fw, 3), "RGB"))
    if not frames:
        return Loaded(None, facts, "no frame decoded")
    facts["frames_classified"] = len(frames)
    out = _finish_image(frames[0], facts, f"ffmpeg frames at {', '.join(f'{p:.0%}' for p in VIDEO_POSITIONS[:len(frames)])}"
                        if duration > 0 else "ffmpeg first frame")
    out.frames = frames[1:]
    return out


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------
def _window(a: np.ndarray, finite: np.ndarray, is_float: bool) -> tuple[float, float]:
    """(lo, hi) intensity window: percentiles 0.5 / 99.5 (on at most 1M
    samples) for floats and for integers with more than WINDOW_MIN_LEVELS
    distinct values; min-max for few-level data (label maps, masks).
    Subsamples before the finite mask is applied: no full-size copy."""
    step = max(1, a.size // 1_000_000)
    flat, fmask = a.reshape(-1), finite.reshape(-1)
    vals = flat[::step][fmask[::step]]
    if vals.size == 0:                       # sparse finite values the stride missed: few, take them all
        vals = flat[fmask]
    if not is_float and len(np.unique(vals)) <= WINDOW_MIN_LEVELS:
        return float(vals.min()), float(vals.max())
    lo, hi = (float(x) for x in np.percentile(vals, [0.5, 99.5]))
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    return lo, hi


def array_to_rgb8(arr: np.ndarray) -> Image.Image | None:
    """Any 2D / HxWx{1,3,4} array -> 8-bit RGB PIL image (windowed, alpha
    composited over white)."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
    # Multi-band stacks (GeoTIFF bands, channel-first volumes): first band of
    # the shortest axis.
    while arr.ndim > 3 or (arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4)):
        axis = int(np.argmin(arr.shape))
        arr = np.take(arr, 0, axis=axis)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    alpha = None
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3].astype(np.float32)
        amax = float(alpha.max()) if alpha.size else 0.0
        alpha = alpha / amax if amax > 0 else None
        arr = arr[..., :3]
    if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] == 3):
        return None
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        # One float32 working copy, scaled in place; the masks are bool
        # (a plane at the decode budget stays near 2x its size, not 4x).
        is_float = np.issubdtype(arr.dtype, np.floating)
        a = np.array(arr, dtype=np.float32, order="C", copy=True)
        del arr
        finite = np.isfinite(a)
        if is_float:
            # Float rasters often carry huge nodata values (-3.4e38): scale
            # between robust percentiles and treat everything else as clipped.
            with np.errstate(invalid="ignore"):
                finite &= a < 1e30
                finite &= a > -1e30
        if not finite.any():
            return None
        lo, hi = _window(a, finite, is_float)
        np.logical_not(finite, out=finite)
        a[finite] = lo
        del finite
        if hi > lo:
            np.subtract(a, lo, out=a)
            np.multiply(a, 255.0 / (hi - lo), out=a)
            np.clip(a, 0, 255, out=a)
        else:
            a.fill(0)
        arr = a.astype(np.uint8)
        del a
    if alpha is not None and float(alpha.min()) < 1.0:
        a3 = alpha[..., None]
        arr = (arr.astype(np.float32) * a3 + 255.0 * (1.0 - a3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L" if arr.ndim == 2 else "RGB")
    return img.convert("RGB")


def to_rgb8(im: Image.Image) -> Image.Image:
    """PIL image of any mode -> 8-bit RGB. 16/32-bit and float are windowed
    (array_to_rgb8); transparency (RGBA, LA, PA, P or L with a transparent
    color) is composited over white, so a transparent figure background does
    not turn black."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        out = array_to_rgb8(np.asarray(im))
        if out is not None:
            return out
    if im.mode in ("RGBA", "LA", "PA", "La", "RGBa") or (im.mode in ("P", "L", "RGB")
                                                         and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if im.mode == "RGB":
        return im.copy()
    return im.convert("RGB")


MASK_SAMPLE = 128          # side of the nearest-neighbour sample the mask check counts values in
MASK_MAX_GRAY_LEVELS = 4
MASK_MAX_COLORS = 8
# Dominance test (anti-aliased masks and label maps): the MASK_MAX_GRAY_LEVELS
# most common gray levels, or the MASK_MAX_COLORS most common colors, cover
# at least MASK_DOMINANT_COVERAGE of the sample, and the values after the
# most common one (the background) cover at least MASK_DOMINANT_REST of the
# pixels outside it. The second part keeps a small textured object on a flat
# black canvas (a dark frame, a padded crop) from counting as a mask. Tuned on
# the tier-3 training images (no real image of any class flagged) and the
# HRF-Seg+ masks (all flagged); see docs/survey.md.
MASK_DOMINANT_COVERAGE = 0.95
MASK_DOMINANT_REST = 0.6


def _dominated(values: np.ndarray, k: int) -> bool:
    """The k most common values cover MASK_DOMINANT_COVERAGE of ``values``
    and the next k-1 after the most common cover MASK_DOMINANT_REST of the
    values that are not the most common one."""
    counts = np.sort(np.unique(values, return_counts=True)[1])[::-1]
    total = int(counts.sum())
    top = int(counts[:k].sum())
    if top < MASK_DOMINANT_COVERAGE * total:
        return False
    outside = total - int(counts[0])
    return outside == 0 or (top - int(counts[0])) >= MASK_DOMINANT_REST * outside


def is_mask_like(img: Image.Image) -> bool:
    """Segmentation mask or label map, judged in a MASK_SAMPLE x MASK_SAMPLE
    nearest-neighbour sample (no interpolation, so a mask keeps its exact
    values): a grayscale image with at most MASK_MAX_GRAY_LEVELS gray
    levels, or a color image with at most MASK_MAX_COLORS distinct colors;
    or, for anti-aliased masks and label maps (a binary mask with 16 edge
    levels), one where the MASK_MAX_GRAY_LEVELS (gray) or MASK_MAX_COLORS
    (color) most common values dominate (``_dominated``). A real photograph
    or scan has hundreds of values in 16384 pixels spread well beyond its
    top few, even a dark, low-contrast IR or FAF frame; the check is tuned
    to miss rather than to flag (a JPEG-compressed mask is flagged only while
    its flat regions still dominate, not when ringing spreads over more than
    5% of the pixels). Flat-colour graphics (plots, word clouds, logos) pass
    it too: the runner keeps MASK only when the model's argmax is an eye
    class (runner._classify)."""
    small = np.asarray(img.convert("RGB").resize((MASK_SAMPLE, MASK_SAMPLE), Image.NEAREST)).astype(np.int32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    # Gray (within +-2 per channel, the chroma residue of a saved mask) is
    # judged by its green levels only, with the stricter gray limit, so a
    # dark grayscale frame of 5 to 8 levels is never a "color label map".
    if np.abs(r - g).max() <= 2 and np.abs(g - b).max() <= 2:
        values, limit = g.ravel(), MASK_MAX_GRAY_LEVELS
    else:
        values, limit = ((r << 16) | (g << 8) | b).ravel(), MASK_MAX_COLORS
    return len(np.unique(values)) <= limit or _dominated(values, limit)


def native_mask_like(arr: np.ndarray) -> bool | None:
    """Mask test on the native values of a single-channel integer array
    deeper than 8 bits (None for anything else: the 8-bit test applies).
    Min-max or window scaling to 8 bits only merges values, so one hot pixel
    can squeeze a real 12- or 16-bit image into a few gray levels; the
    native values keep it apart. Same rule and sample as is_mask_like:
    at most MASK_MAX_GRAY_LEVELS values, or dominated by that many."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or a.dtype.itemsize < 2:
        return None
    h, w = a.shape
    rows = (np.arange(MASK_SAMPLE) * h) // MASK_SAMPLE
    cols = (np.arange(MASK_SAMPLE) * w) // MASK_SAMPLE
    v = np.asarray(a[np.ix_(rows, cols)]).ravel()
    return bool(len(np.unique(v)) <= MASK_MAX_GRAY_LEVELS or _dominated(v, MASK_MAX_GRAY_LEVELS))


def _pixel_flags(img: Image.Image, facts: dict):
    """rgb_is_gray (R==G==B), blue_zero (Optos-style pseudocolor hint),
    mask_like (is_mask_like) and blank (a constant frame: not classified)."""
    try:
        small = np.asarray(img.resize((64, 64), Image.NEAREST)).astype(np.int16)
        r, g, b = small[..., 0], small[..., 1], small[..., 2]
        facts["rgb_is_gray"] = bool(np.abs(r - g).max() <= 2 and np.abs(g - b).max() <= 2)
        facts["blue_zero"] = bool(b.max() <= 3 and max(r.max(), g.max()) > 20)
        facts["mask_like"] = is_mask_like(img)
        ext = img.getextrema()
        if ext and not isinstance(ext[0], (tuple, list)):
            ext = (ext,)
        facts["blank"] = all(lo == hi for lo, hi in ext)
    except Exception:  # noqa: BLE001
        pass


def content_crop(img: Image.Image) -> Image.Image:
    """Crop to the bounding box of non-near-black pixels (training rule).

    Identical to the tier-3 training crop: threshold max(12, 0.08 * max gray);
    skip when under 2% of pixels are foreground or the box is under 16 px.
    """
    # uint8 gray compared with a float threshold gives the same mask as the
    # float32 training code, without a 4-byte-per-pixel copy; the box comes
    # from row/column projections instead of np.where index arrays (which
    # cost 16 bytes per foreground pixel at the 80 MP cap).
    g = np.asarray(img.convert("L"))
    thr = max(12.0, float(g.max()) * 0.08)
    m = g > thr
    del g
    if np.count_nonzero(m) < 0.02 * m.size:
        return img
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return img
    return img.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


_MEAN = np.array(MEAN, np.float32).reshape(3, 1, 1)
_STD = np.array(STD, np.float32).reshape(3, 1, 1)


def preprocess(img: Image.Image, size: int = IMG_SIZE, stored: int = TRAIN_STORED_SIZE) -> np.ndarray:
    """Rebuild the training input exactly.

    Training images were content-cropped and then squashed (aspect ratio NOT
    kept) to ``stored`` x ``stored`` PNGs; the eval transform
    Resize(size) + CenterCrop(size) then turned each square into size x size
    with nothing cut off. So here:

        content_crop -> resize((stored, stored)) -> resize((size, size))
        -> ToTensor -> Normalize

    The second resize equals Resize(size) + CenterCrop(size) on a square.
    Keeping the aspect ratio and center-cropping instead would drop the sides
    of wide images (a third of a 768x496 OCT B-scan) that the model saw in
    training. The squash also bounds memory for extreme aspect ratios.
    """
    img = content_crop(img)
    img = img.resize((stored, stored), Image.BILINEAR)
    if stored != size:
        img = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
    return (a - _MEAN) / _STD
