"""Shared constants for the Zenodo modality survey.

Everything that decides "what kind of file is this" lives here so the
downloader, the archive walker and the Excel builder agree on the same sets.
"""

from __future__ import annotations

from pathlib import PurePosixPath

# Classifier output order. Must match the checkpoint's final layer.
CLASSES: list[str] = ["CFP", "IR", "PSC", "FAF", "OCT", "OCTA", "NEG"]
EYE_CLASSES: list[str] = CLASSES[:6]
UNCERTAIN = "UNCERTAIN"
# Images the pixel check flags as masks or label maps (binary, at most 4 gray
# levels, or at most 8 colors, or dominated by that few values) get this
# label instead of a classifier class: like UNCERTAIN it is not a modality,
# and presence, eye status and dominant class are decided over the other
# images.
MASK = "MASK"
# A record is mask_dominated when masks are at least MASK_DOMINATED_FRAC of
# its classified images (frac_MASK) and fewer than MASK_DOMINATED_MIN_NON_MASK
# non-mask images are classified: its eye classes are then review-only.
MASK_DOMINATED_FRAC = 0.5
MASK_DOMINATED_MIN_NON_MASK = 50
# ... unless the non-mask images hold at least MASK_EXEMPT_MIN_IMAGES
# thresholded images of one eye class other than OCTA with a mean top-1
# confidence of at least MASK_EXEMPT_MIN_CONF (photos next to their masks in
# a segmentation dataset). OCTA never exempts: missed masks come out as OCTA.
MASK_EXEMPT_MIN_IMAGES = 10
MASK_EXEMPT_MIN_CONF = 0.75
CLASS_LABELS: dict[str, str] = {
    "CFP": "Color fundus photograph",
    "IR": "Infrared reflectance (near-infrared SLO)",
    "PSC": "Ultra-widefield pseudocolor",
    "FAF": "Fundus autofluorescence",
    "OCT": "OCT B-scan",
    "OCTA": "OCT angiography en face",
    "NEG": "Non-eye image",
    UNCERTAIN: "Below confidence threshold",
    MASK: "Mask, label map or flat-colour graphic whose model class would be an eye modality (no modality label)",
}


def dominant_of(counts, order: list[str] | None = None) -> str | None:
    """Most frequent class in ``counts`` (a mapping, counts may be weighted
    floats). Ties go to the class that comes first in ``order`` (default
    CLASSES + UNCERTAIN), so every place that picks a dominant class picks
    the same one. None when every count is zero."""
    order = order or CLASSES + [UNCERTAIN]
    best = max(order, key=lambda c: (counts.get(c, 0) or 0, -order.index(c)))
    return best if (counts.get(best, 0) or 0) > 0 else None

# Preprocessing used at training time: content crop, squash to a
# TRAIN_STORED_SIZE square PNG (tier-3 data prep, aspect ratio not kept), then
# torchvision eval_tf Resize(224) + CenterCrop(224) + ImageNet stats.
TRAIN_STORED_SIZE = 256
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# File kinds
# ---------------------------------------------------------------------------
RASTER_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm", ".webp", ".jp2", ".j2k", ".jpx", ".psd",
    ".heic", ".heif", ".avif",
    # TIFF-based containers read through tifffile (OME-TIFF, Zeiss LSM,
    # whole-slide SVS/NDPI, BigTIFF variants)
    ".ome.tif", ".ome.tiff", ".lsm", ".svs", ".ndpi", ".btf", ".tf8", ".scn",
}
TIFF_EXTS = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".lsm", ".svs", ".ndpi", ".btf", ".tf8", ".scn"}
# Vector figures: embedded raster images are taken out, else rendered.
VECTOR_EXTS = {".svg", ".svgz"}
DICOM_EXTS = {".dcm", ".dicom", ".ima"}
# Volumes we can slice directly (a single file holds header and voxels).
VOLUME_EXTS = {".nii", ".nii.gz", ".nrrd", ".mha"}
# Volume headers whose voxels live in a companion data file with the same
# stem (x.mhd + x.raw / x.zraw, Analyze x.hdr + x.img, detached x.nhdr +
# x.raw): both files are fetched together and read as one volume.
VOLUME_HEADER_EXTS = {".mhd", ".nhdr", ".hdr"}
VOLUME_DATA_EXTS = {".raw", ".zraw", ".img"}
VOLUME_INVENTORY_ONLY_EXTS = VOLUME_HEADER_EXTS       # old name
# Proprietary OCT / SLO containers: decoded where an open reader exists
# (oct-converter: E2E, FDS, FDA, Bioptigen OCT; Heidelberg VOL and Thorlabs
# OCT by the survey's own readers); .sdb is catalogued only.
VENDOR_OCT_EXTS = {".e2e", ".sdb", ".vol", ".fds", ".fda", ".oct"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v", ".flv", ".webm", ".3gp", ".ogv"}
# Numeric array containers read when a dataset is image shaped (MAT v5 and
# v7.3, HDF5 incl. Imaris .ims, NumPy).
ARRAY_EXTS = {".npy", ".npz", ".h5", ".hdf5", ".hdf", ".he5", ".mat", ".ims"}
# Microscopy containers (middle Z plane, first channel).
MICROSCOPY_EXTS = {".czi", ".lif", ".nd2", ".oib", ".mrc"}

ZIP_EXTS = {".zip"}
TAR_EXTS = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tzst"}
SEVEN_ZIP_EXTS = {".7z"}
RAR_EXTS = {".rar"}
# Single-file compression (x.tif.gz, x.dcm.bz2). .nii.gz is a volume, not this.
SINGLE_COMPRESSED_EXTS = {".gz", ".bz2", ".xz", ".zst"}
ARCHIVE_EXTS = ZIP_EXTS | TAR_EXTS | SEVEN_ZIP_EXTS | RAR_EXTS

# Files never worth downloading for an image survey (their count is known from
# the Zenodo file list, which is all the workbook needs).
SKIP_DOWNLOAD_EXTS = {
    ".pdf", ".txt", ".md", ".rst", ".csv", ".tsv", ".xlsx", ".xls", ".json",
    ".xml", ".yaml", ".yml", ".doc", ".docx", ".ppt", ".pptx", ".html", ".htm",
    ".py", ".ipynb", ".r", ".m", ".c", ".cpp", ".h", ".java", ".js", ".sh",
    ".pt", ".pth", ".ckpt", ".onnx", ".pb",
    ".pkl", ".pickle", ".parquet", ".feather", ".sqlite", ".db",
    ".fastq", ".fq", ".fastq.gz", ".fq.gz", ".bam", ".sam", ".cram", ".vcf",
    ".vcf.gz", ".bed", ".fasta", ".fa", ".gtf", ".gff", ".bw", ".bigwig",
    ".mp3", ".wav", ".flac",
}

# Longest-first so ".tar.gz" wins over ".gz" and ".nii.gz" over ".gz".
_ALL_KNOWN = sorted(
    RASTER_EXTS | VECTOR_EXTS | DICOM_EXTS | VOLUME_EXTS | VOLUME_HEADER_EXTS | VOLUME_DATA_EXTS
    | VENDOR_OCT_EXTS | VIDEO_EXTS | ARRAY_EXTS | MICROSCOPY_EXTS | ARCHIVE_EXTS
    | SINGLE_COMPRESSED_EXTS | SKIP_DOWNLOAD_EXTS,
    key=len, reverse=True,
)


def detect_ext(name: str) -> str:
    """Longest known extension of a file or archive member name (lower case)."""
    low = PurePosixPath(name.replace("\\", "/")).name.lower()
    for ext in _ALL_KNOWN:
        if low.endswith(ext):
            return ext
    if "." in low:
        return "." + low.rsplit(".", 1)[-1]
    return ""


import re as _re

# Split archive parts beyond the ones 7z and unrar name themselves
# (x.part2.rar, x.r00, x.z01, x.7z.002): numbered parts of any file
# (x.tar.001, data.002), `split` suffixes (x.zip.partaa, x.partab) and
# x.part02 without .rar.
_SPLIT_NUM = _re.compile(r"\.(\d{3})$")
_SPLIT_ALPHA = _re.compile(r"\.part([a-z]{2})$")
_SPLIT_PARTN = _re.compile(r"\.part(\d{1,3})$")


def is_fragment(name: str) -> bool:
    """True for secondary parts of split archives (.z01, .r00, .part2.rar,
    .7z.002, .tar.002, .partab, .part02).

    These are opened through their first part, never on their own.
    """
    low = name.lower()
    if _re.search(r"\.z\d{2,3}$", low) or _re.search(r"\.r\d{2,3}$", low):
        return True
    m = _re.search(r"\.part(\d+)\.rar$", low)
    if m and int(m.group(1)) > 1:
        return True
    m = _SPLIT_NUM.search(low)
    if m and int(m.group(1)) > 1:
        return True
    m = _SPLIT_ALPHA.search(low)
    if m and m.group(1) != "aa":
        return True
    m = _SPLIT_PARTN.search(low)
    if m and int(m.group(1)) > 1:
        return True
    return False


def split_first(name: str) -> bool:
    """First part of a split set the survey joins or hands to 7z
    (x.7z.001, x.zip.001, x.tar.001, data.001, x.zip.partaa, x.part01)."""
    low = name.lower()
    m = _SPLIT_NUM.search(low)
    if m and int(m.group(1)) == 1:
        return True
    m = _SPLIT_ALPHA.search(low)
    if m and m.group(1) == "aa":
        return True
    m = _SPLIT_PARTN.search(low)
    return bool(m and int(m.group(1)) == 1)


def split_stem(name: str) -> str | None:
    """Name of the whole file a split part belongs to ('' suffix removed):
    x.tar.001 -> x.tar, x.zip.partab -> x.zip, x.part02 -> x. None when the
    name is not a generic split part (rar and spanned zip sets are handled
    by their tools)."""
    low = name.lower()
    for rx in (_SPLIT_NUM, _SPLIT_ALPHA, _SPLIT_PARTN):
        m = rx.search(low)
        if m:
            return name[: m.start()]
    return None


def file_kind(name: str) -> str:
    """Coarse kind: raster, vector, dicom, volume, volume_pair (header of a
    header + data pair), volume_data (its data file), vendor_oct, video,
    array, microscopy, archive, compressed, fragment, noext, other."""
    if is_fragment(name):
        return "fragment"
    ext = detect_ext(name)
    if ext in RASTER_EXTS:
        return "raster"
    if ext in VECTOR_EXTS:
        return "vector"
    if ext in DICOM_EXTS:
        return "dicom"
    if ext in VOLUME_EXTS:
        return "volume"
    if ext in VOLUME_HEADER_EXTS:
        return "volume_pair"
    if ext in VOLUME_DATA_EXTS:
        return "volume_data"
    if ext in VENDOR_OCT_EXTS:
        return "vendor_oct"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ARRAY_EXTS:
        return "array"
    if ext in MICROSCOPY_EXTS:
        return "microscopy"
    if ext in ARCHIVE_EXTS or split_first(name):
        return "archive"
    if ext in SINGLE_COMPRESSED_EXTS:
        return "compressed"
    if ext == "":
        return "noext"
    return "other"


# Kinds the walker registers as image entries and images.load_frame turns
# into a classifiable RGB image (a file of such a kind that holds no image,
# an HDF5 of model weights for example, is reported as a decode error).
IMAGE_KINDS = {"raster", "vector", "dicom", "volume", "volume_pair", "vendor_oct", "video", "array", "microscopy"}
# Kinds the loaders need as a file on disk (never as bytes in memory).
PATH_KINDS = {"volume", "volume_pair", "vendor_oct", "video", "array", "microscopy"}
# Short names for the per-record formats summary.
PIXEL_KINDS = IMAGE_KINDS


def nested_worth(name: str, kind: str) -> bool:
    """Is an archive or single-file compressed file worth reading?

    Archives always are. A compressed file (x.tif.gz, x.csv.gz) only when the
    name inside is an image, an archive or extensionless (often DICOM).
    """
    if kind == "archive":
        return True
    if kind != "compressed":
        return False
    inner = name[: -len(detect_ext(name))]
    return file_kind(inner) in IMAGE_KINDS | {"archive", "noext"}


def wanted_for_download(name: str) -> bool:
    """Should the streaming path fetch this Zenodo file?

    Every pixel-bearing kind (IMAGE_KINDS: images, vector figures, DICOM,
    volumes and volume headers, vendor OCT, video, numeric arrays,
    microscopy), archives (and their split parts), single-file compressed
    images or archives (x.tif.gz, x.tar.bz2 parts) and extensionless files
    (often DICOM) are fetched. Volume data files (x.raw, x.img) are fetched
    only next to their header (runner.plan_files). Documents, code, tables,
    genomics and compressed non-images (x.csv.gz) are not: nothing
    downstream would read them.
    """
    kind = file_kind(name)
    if kind == "compressed" and not nested_worth(name, kind):
        return False
    if kind in IMAGE_KINDS | {"archive", "fragment", "compressed", "noext"}:
        return detect_ext(name) not in SKIP_DOWNLOAD_EXTS
    return False


def volume_companion(name: str, names) -> str | None:
    """The data file of a volume header (x.mhd -> x.raw / x.zraw, x.hdr ->
    x.img, x.nhdr -> x.raw) among ``names`` (same directory, case
    insensitive), or None."""
    ext = detect_ext(name)
    if ext not in VOLUME_HEADER_EXTS:
        return None
    stem = name[: -len(ext)]
    lookup = {n.lower(): n for n in names}
    order = (".img",) if ext == ".hdr" else (".raw", ".zraw", ".img", ".gz", ".raw.gz")
    for data_ext in order:
        hit = lookup.get((stem + data_ext).lower())
        if hit is not None:
            return hit
    return None


def volume_header_of(name: str, names) -> str | None:
    """The header a volume data file belongs to (inverse of
    volume_companion), or None."""
    ext = detect_ext(name)
    if ext not in VOLUME_DATA_EXTS:
        return None
    stem = name[: -len(ext)]
    lookup = {n.lower(): n for n in names}
    for hdr_ext in (".hdr",) if ext == ".img" else (".mhd", ".nhdr"):
        hit = lookup.get((stem + hdr_ext).lower())
        if hit is not None:
            return hit
    return None


# ---------------------------------------------------------------------------
# Magic-byte sniffing of extensionless (or misnamed) files
# ---------------------------------------------------------------------------
SNIFF_BYTES = 132


def sniff_kind(head: bytes) -> tuple[str, str] | None:
    """(kind, ext) from a file's first SNIFF_BYTES bytes, or None.

    DICOM with the 128-byte preamble and 'DICM', and preamble-less DICOM (a
    group 0x0008 or 0x0002 element at offset 0 with a plausible explicit VR
    or implicit length); PNG, JPEG, TIFF (II*, MM*, BigTIFF) and BMP."""
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom", ".dcm"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "raster", ".png"
    if head[:3] == b"\xff\xd8\xff":
        return "raster", ".jpg"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "raster", ".tif"
    if head[:2] == b"BM" and len(head) >= 18:
        import struct
        size, = struct.unpack("<I", head[2:6])
        dib, = struct.unpack("<I", head[14:18])
        if dib in (12, 40, 52, 56, 64, 108, 124) and size >= 26:
            return "raster", ".bmp"
    if len(head) >= 8 and head[:2] in (b"\x08\x00", b"\x02\x00"):
        import struct
        elem, = struct.unpack("<H", head[2:4])
        vr = head[4:6]
        if elem <= 0x0100 and (vr.isalpha() and vr.isupper() or struct.unpack("<I", head[4:8])[0] < 1024):
            return "dicom", ".dcm"
    return None
