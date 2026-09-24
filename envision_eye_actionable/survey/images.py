"""Decode one frame per image file and collect DICOM-like header facts.

Every loader returns ``Loaded(image, facts, error)``:

* ``image`` is an 8-bit RGB ``PIL.Image`` ready for preprocessing, or None
  when the file is inventory-only (too large, undecodable, compressed DICOM
  without a codec, ...).
* ``facts`` is a small dict of header facts that map to DICOM attributes
  (Rows, Columns, PhotometricInterpretation, BitsStored, NumberOfFrames,
  PixelSpacing, Manufacturer, ...). Real DICOM files carry their own tags
  under ``facts["dicom"]``. Identifying tags are never read into facts.

Which frame is classified:
    raster / multi-page TIFF: first frame (frame count recorded)
    DICOM multi-frame:        middle frame
    NIfTI / NRRD / MHA:       middle slice along the shortest spatial axis
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from .constants import IMG_SIZE, MEAN, STD, TRAIN_STORED_SIZE

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # the caller enforces its own pixel cap

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


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def load_frame(kind: str, ext: str, data: bytes | None, path: Path | None,
               max_pixels: int = 80_000_000) -> Loaded:
    """Decode one representative frame of an image file (see module doc)."""
    try:
        if kind == "dicom":
            return _load_dicom(data, path, max_pixels)
        if kind == "volume":
            if path is None:
                return Loaded(None, {"format": ext}, "volume needs a file path")
            return _load_volume(path, ext, max_pixels)
        return _load_raster(data, path, ext, max_pixels)
    except MemoryError:
        return Loaded(None, {"format": ext}, "MemoryError")
    except Exception as e:  # noqa: BLE001 - one bad file never stops a record
        return Loaded(None, {"format": ext}, f"{type(e).__name__}: {e}"[:200])


# ---------------------------------------------------------------------------
# raster (PIL first, tifffile fallback)
# ---------------------------------------------------------------------------
def _load_raster(data, path, ext, max_pixels) -> Loaded:
    src = io.BytesIO(data) if data is not None else path
    try:
        im = Image.open(src)
    except Exception as e:  # noqa: BLE001
        if ext in (".tif", ".tiff"):
            return _load_tiff_tifffile(data, path, max_pixels, first_error=str(e))
        raise
    with im:
        facts = _raster_facts(im, ext)
        w, h = im.size
        if w * h > max_pixels:
            return Loaded(None, facts, "too_large")
        if im.format == "JPEG" and min(w, h) > 4 * IMG_SIZE:
            # Decode at a reduced DCT scale: far less RAM and CPU, and the
            # image is resized to 224 anyway.
            im.draft(im.mode, (2 * IMG_SIZE, 2 * IMG_SIZE))
        try:
            im.seek(0)
            im.load()
        except Exception as e:  # noqa: BLE001
            if ext in (".tif", ".tiff"):
                return _load_tiff_tifffile(data, path, max_pixels, first_error=str(e), facts=facts)
            raise
        rgb = to_rgb8(im)
    _pixel_flags(rgb, facts)
    return Loaded(rgb, facts)


def _load_tiff_tifffile(data, path, max_pixels, first_error="", facts=None,
                        max_bytes: int = 1 << 30) -> Loaded:
    import tifffile
    src = io.BytesIO(data) if data is not None else path
    with tifffile.TiffFile(src) as tf:
        page = tf.pages[0]
        # page.shape is (S, H, W) for planar-separate pages, so rows/cols come
        # from the image dimensions, never from shape[0]/shape[1].
        rows, cols = int(page.imagelength), int(page.imagewidth)
        depth = int(getattr(page, "imagedepth", 1) or 1)
        samples = int(getattr(page, "samplesperpixel", 1) or 1)
        bits = int(page.bitspersample)
        compression = int(page.compression)
        facts = dict(facts or {})
        facts.update({
            "format": "TIFF", "rows": rows, "cols": cols,
            "frames": len(tf.pages), "bits": bits, "samples": samples,
            "photometric": _TIFF_PHOTOMETRIC.get(int(page.photometric), str(page.photometric)),
            "lossy": "01" if compression in (6, 7) else "00",
        })
        if compression in (6, 7):
            facts["lossy_method"] = "ISO_10918_1"
        elif compression in (33003, 33005, 34712):
            facts["lossy_method"] = "ISO_15444_1"
            facts.pop("lossy", None)   # JPEG 2000 may be lossless: unknown
        if int(getattr(page, "sampleformat", 1) or 1) == 3:
            facts["float"] = True
        if rows * cols > max_pixels:
            return Loaded(None, facts, "too_large")
        # Cap the decoded size before asarray(): array_to_rgb8 makes float32
        # copies on top, and a 7 GB VM is OOM-killed rather than raising.
        if rows * cols * depth * samples * max(1, bits) // 8 > max_bytes:
            return Loaded(None, facts, "too_large")
        arr = page.asarray()
    img = array_to_rgb8(arr)
    if img is None:
        return Loaded(None, facts, f"unsupported TIFF array (PIL: {first_error})"[:200])
    _pixel_flags(img, facts)
    return Loaded(img, facts)


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
        xres, yres = float(xres), float(yres)
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


def _exif_dt(s: str) -> str:
    """'YYYY:MM:DD HH:MM:SS' -> 'YYYYMMDDHHMMSS' (DICOM DT); '' when junk."""
    digits = re.sub(r"\D", "", s)[:14]
    return digits if len(digits) >= 8 and not digits.startswith("0000") else ""


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
    if rows * cols == 0 or rows * cols > max_pixels:
        return Loaded(None, facts, "too_large" if rows * cols else "no dimensions")

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
        full = ds.pixel_array
        arr = full[mid] if (mid is not None and full.ndim >= 3 and full.shape[0] == frames) else full
    if str(d.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    img = array_to_rgb8(np.asarray(arr))
    if img is None:
        return Loaded(None, facts, f"unsupported DICOM array shape {getattr(arr, 'shape', None)}")
    _pixel_flags(img, facts)
    return Loaded(img, facts)


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
def _load_volume(path: Path, ext: str, max_pixels: int, max_bytes: int = 2 << 30) -> Loaded:
    """Middle slice of a NIfTI / NRRD / MHA volume.

    ``max_bytes`` caps the UNCOMPRESSED voxel size (from the header, before
    any pixel is read): a gzip NRRD or compressed MHA far smaller on disk can
    expand to many GB, and on a small VM that ends in the OOM killer rather
    than a catchable MemoryError. Only the middle slice is materialised where
    the reader allows it (nibabel proxy slicing, SimpleITK extract region).
    """
    facts: dict = {"format": ext.lstrip(".").upper()}
    if ext in (".nii", ".nii.gz"):
        arr = _nifti_slice(path, facts, max_pixels, max_bytes)
    else:
        try:
            arr = _itk_slice(path, facts, max_pixels, max_bytes)
        except Exception as e:  # noqa: BLE001 - ITK cannot read every NRRD
            if ext != ".nrrd":
                raise
            logger.debug("SimpleITK failed on %s (%s); trying pynrrd", path, e)
            arr = _pynrrd_slice(path, facts, max_pixels, max_bytes)
    if arr is None:
        return Loaded(None, facts, facts.pop("_error", "volume not loaded"))
    while arr.ndim > 2 and not (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
        arr = arr[..., 0]
    facts["rows"], facts["cols"] = int(arr.shape[0]), int(arr.shape[1])
    facts["bits"] = int(arr.dtype.itemsize * 8)
    if np.issubdtype(arr.dtype, np.floating):
        facts["float"] = True
    facts["photometric"] = "RGB" if arr.ndim == 3 else "MONOCHROME2"
    if arr.shape[0] * arr.shape[1] > max_pixels:
        return Loaded(None, facts, "too_large")
    img = array_to_rgb8(arr)
    if img is None:
        return Loaded(None, facts, "unsupported slice")
    _pixel_flags(img, facts)
    return Loaded(img, facts)


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
    if _slice_pixels(shape, axis) > max_pixels:
        return _refuse(facts, "too_large")
    if axis is None:
        # Scaled NIfTI data comes back as float64: budget 8 bytes per value.
        if int(np.prod(shape, dtype=np.int64)) * 8 > max_bytes:
            return _refuse(facts, "volume too large to load")
        arr = np.asarray(img.dataobj)
        facts["frames"] = 1
        if len(zooms) >= 2:
            facts["spacing_mm"] = [round(zooms[0], 6), round(zooms[1], 6)]
        return arr
    sl: list = [slice(None)] * len(shape)
    sl[axis] = idx
    for extra in range(3, len(shape)):
        sl[extra] = 0
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
    """MHA / NRRD through SimpleITK: header first, then only the middle slice."""
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
        if size[0] * size[1] > max_pixels:
            return _refuse(facts, "too_large")
        if total > max_bytes:
            return _refuse(facts, "volume too large to load")
        arr = sitk.GetArrayFromImage(reader.Execute())
        facts["frames"] = 1
        facts["spacing_mm"] = [round(spacing[1], 6), round(spacing[0], 6)]
        return arr
    # Middle slice along the shortest of the first three spatial dims.
    d = int(np.argmin(size[:3]))
    keep = [i for i in range(3) if i != d]                # ITK dims kept (x-like first)
    if size[keep[0]] * size[keep[1]] > max_pixels:
        return _refuse(facts, "too_large")
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
    if _slice_pixels(shape, axis) > max_pixels:
        return _refuse(facts, "too_large")
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
# pixels
# ---------------------------------------------------------------------------
def array_to_rgb8(arr: np.ndarray) -> Image.Image | None:
    """Any 2D / HxWx{1,3,4} array -> 8-bit RGB PIL image (min-max scaled)."""
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
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] == 3):
        return None
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        finite = np.isfinite(a)
        if np.issubdtype(arr.dtype, np.floating):
            # Float rasters often carry huge nodata values (-3.4e38): scale
            # between robust percentiles and treat everything else as clipped.
            finite &= np.abs(a) < 1e30
        if not finite.any():
            return None
        vals = a[finite]
        if np.issubdtype(arr.dtype, np.floating):
            lo, hi = (float(x) for x in np.percentile(vals, [0.5, 99.5]))
        else:
            lo, hi = float(vals.min()), float(vals.max())
        a = np.where(finite, a, lo)
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        arr = a.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L" if arr.ndim == 2 else "RGB")
    return img.convert("RGB")


def to_rgb8(im: Image.Image) -> Image.Image:
    """PIL image of any mode -> 8-bit RGB. 16/32-bit and float are min-max scaled."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        out = array_to_rgb8(np.asarray(im))
        if out is not None:
            return out
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
    5% of the pixels). Masks get the MASK label instead of a classifier
    modality (runner._classify)."""
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


def _pixel_flags(img: Image.Image, facts: dict):
    """rgb_is_gray (R==G==B), blue_zero (Optos-style pseudocolor hint) and
    mask_like (is_mask_like)."""
    try:
        small = np.asarray(img.resize((64, 64), Image.NEAREST)).astype(np.int16)
        r, g, b = small[..., 0], small[..., 1], small[..., 2]
        facts["rgb_is_gray"] = bool(np.abs(r - g).max() <= 2 and np.abs(g - b).max() <= 2)
        facts["blue_zero"] = bool(b.max() <= 3 and max(r.max(), g.max()) > 20)
        facts["mask_like"] = is_mask_like(img)
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
