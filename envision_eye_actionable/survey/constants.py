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
CLASS_LABELS: dict[str, str] = {
    "CFP": "Color fundus photograph",
    "IR": "Infrared reflectance (near-infrared SLO)",
    "PSC": "Ultra-widefield pseudocolor",
    "FAF": "Fundus autofluorescence",
    "OCT": "OCT B-scan",
    "OCTA": "OCT angiography en face",
    "NEG": "Non-eye image",
    UNCERTAIN: "Below confidence threshold",
    MASK: "Mask or label map (binary or few colors; no modality label)",
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
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm", ".webp", ".jp2", ".j2k",
}
DICOM_EXTS = {".dcm", ".dicom"}
# Volumes we can slice directly (a single file holds header and voxels).
VOLUME_EXTS = {".nii", ".nii.gz", ".nrrd", ".mha"}
# Volumes that need a companion data file; inventory only.
VOLUME_INVENTORY_ONLY_EXTS = {".mhd", ".nhdr", ".hdr"}
# Proprietary OCT / SLO containers: counted and used as manufacturer hints,
# never decoded here.
VENDOR_OCT_EXTS = {".e2e", ".sdb", ".vol", ".fds", ".fda"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}

ZIP_EXTS = {".zip"}
TAR_EXTS = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}
SEVEN_ZIP_EXTS = {".7z"}
RAR_EXTS = {".rar"}
# Single-file compression (x.tif.gz, x.dcm.bz2). .nii.gz is a volume, not this.
SINGLE_COMPRESSED_EXTS = {".gz", ".bz2", ".xz"}
ARCHIVE_EXTS = ZIP_EXTS | TAR_EXTS | SEVEN_ZIP_EXTS | RAR_EXTS

# Files never worth downloading for an image survey (their count is known from
# the Zenodo file list, which is all the workbook needs).
SKIP_DOWNLOAD_EXTS = {
    ".pdf", ".txt", ".md", ".rst", ".csv", ".tsv", ".xlsx", ".xls", ".json",
    ".xml", ".yaml", ".yml", ".doc", ".docx", ".ppt", ".pptx", ".html", ".htm",
    ".py", ".ipynb", ".r", ".m", ".c", ".cpp", ".h", ".java", ".js", ".sh",
    ".pt", ".pth", ".ckpt", ".onnx", ".pb", ".h5", ".hdf5", ".mat", ".npy",
    ".npz", ".pkl", ".pickle", ".parquet", ".feather", ".sqlite", ".db",
    ".fastq", ".fq", ".fastq.gz", ".fq.gz", ".bam", ".sam", ".cram", ".vcf",
    ".vcf.gz", ".bed", ".fasta", ".fa", ".gtf", ".gff", ".bw", ".bigwig",
    ".mp3", ".wav", ".flac",
} | VIDEO_EXTS | VENDOR_OCT_EXTS

# Longest-first so ".tar.gz" wins over ".gz" and ".nii.gz" over ".gz".
_ALL_KNOWN = sorted(
    RASTER_EXTS | DICOM_EXTS | VOLUME_EXTS | VOLUME_INVENTORY_ONLY_EXTS
    | VENDOR_OCT_EXTS | VIDEO_EXTS | ARCHIVE_EXTS | SINGLE_COMPRESSED_EXTS
    | SKIP_DOWNLOAD_EXTS,
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


def is_fragment(name: str) -> bool:
    """True for secondary parts of split archives (.z01, .r00, .part2.rar, .7z.002).

    These are opened through their first part, never on their own.
    """
    import re
    low = name.lower()
    if re.search(r"\.z\d{2,3}$", low) or re.search(r"\.r\d{2,3}$", low):
        return True
    m = re.search(r"\.part(\d+)\.rar$", low)
    if m and int(m.group(1)) > 1:
        return True
    m = re.search(r"\.(?:7z|zip)\.(\d{3})$", low)
    if m and int(m.group(1)) > 1:
        return True
    return False


def file_kind(name: str) -> str:
    """Coarse kind: raster, dicom, volume, volume_inventory, vendor_oct, video,
    archive, compressed, fragment, noext, other."""
    if is_fragment(name):
        return "fragment"
    ext = detect_ext(name)
    if ext in RASTER_EXTS:
        return "raster"
    if ext in DICOM_EXTS:
        return "dicom"
    if ext in VOLUME_EXTS:
        return "volume"
    if ext in VOLUME_INVENTORY_ONLY_EXTS:
        return "volume_inventory"
    if ext in VENDOR_OCT_EXTS:
        return "vendor_oct"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ARCHIVE_EXTS or name.lower().endswith((".7z.001", ".zip.001")):
        return "archive"
    if ext in SINGLE_COMPRESSED_EXTS:
        return "compressed"
    if ext == "":
        return "noext"
    return "other"


IMAGE_KINDS = {"raster", "dicom", "volume"}


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

    Images, DICOM, volumes, archives (and their split parts), single-file
    compressed images or archives (x.tif.gz, x.tar.bz2 parts) and
    extensionless files (often DICOM) are fetched. Documents, code, tables,
    arrays, genomics, video, vendor OCT containers and compressed non-images
    (x.csv.gz, x.h5.gz) are not: nothing downstream would read them.
    """
    kind = file_kind(name)
    if kind == "compressed" and not nested_worth(name, kind):
        return False
    if kind in IMAGE_KINDS | {"archive", "fragment", "compressed", "noext"}:
        return detect_ext(name) not in SKIP_DOWNLOAD_EXTS
    return False
