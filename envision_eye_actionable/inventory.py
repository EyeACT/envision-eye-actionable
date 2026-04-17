"""Walk an unpacked tree, classify every file by format and role."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Format → (reader_module, modality_tag, is_readable)
#   modality_tag is used for ADDF directoryList emission
#   is_readable → True if we have an open-source decoder for this format
FORMAT_TABLE: dict[str, tuple[str, str, bool]] = {
    # Standard imaging
    ".jpg":   ("pillow",     "retinal_photography", True),
    ".jpeg":  ("pillow",     "retinal_photography", True),
    ".png":   ("pillow",     "retinal_photography", True),
    ".bmp":   ("pillow",     "retinal_photography", True),
    ".gif":   ("pillow",     "retinal_photography", True),
    ".tif":   ("tifffile",   "retinal_photography", True),
    ".tiff":  ("tifffile",   "retinal_photography", True),

    # Medical imaging (open)
    ".dcm":   ("pydicom",    "retinal_imaging",     True),
    ".dicom": ("pydicom",    "retinal_imaging",     True),
    ".nii":   ("nibabel",    "volumetric_imaging",  True),
    ".nii.gz":("nibabel",    "volumetric_imaging",  True),
    ".nrrd":  ("pynrrd",     "volumetric_imaging",  True),
    ".nhdr":  ("pynrrd",     "volumetric_imaging",  True),
    ".mha":   ("simpleitk",  "volumetric_imaging",  True),
    ".mhd":   ("simpleitk",  "volumetric_imaging",  True),

    # Vendor OCT (open via oct-converter)
    ".fds":   ("oct_converter", "retinal_oct", True),
    ".fda":   ("oct_converter", "retinal_oct", True),
    ".e2e":   ("oct_converter", "retinal_oct", True),
    ".img":   ("oct_converter", "retinal_oct", True),
    ".oct":   ("oct_converter", "retinal_oct", True),

    # Scientific arrays
    ".h5":    ("h5py",       "derived_data",        True),
    ".hdf5":  ("h5py",       "derived_data",        True),
    ".mat":   ("scipy_mat73","derived_data",        True),
    ".npy":   ("numpy",      "derived_data",        True),
    ".npz":   ("numpy",      "derived_data",        True),

    # Vendor formats we can NOT decode (pass-through)
    ".vol":   ("none",       "retinal_oct",         False),
    ".nce":   ("none",       "retinal_oct",         False),
    ".nck":   ("none",       "retinal_oct",         False),
    ".ndb":   ("none",       "retinal_oct",         False),
    ".zrx":   ("none",       "retinal_oct",         False),
    ".ims":   ("none",       "retinal_imaging",     False),
    ".tco":   ("none",       "retinal_oct",         False),

    # Docs / labels
    ".pdf":   ("pypdf",      "documentation",       True),
    ".docx":  ("python_docx","documentation",       True),
    ".txt":   ("text",       "documentation",       True),
    ".md":    ("text",       "documentation",       True),
    ".rst":   ("text",       "documentation",       True),
    ".csv":   ("pandas",     "tabular_data",        True),
    ".tsv":   ("pandas",     "tabular_data",        True),
    ".xlsx":  ("pandas",     "tabular_data",        True),
    ".xls":   ("pandas",     "tabular_data",        True),
    ".json":  ("json",       "metadata",            True),
    ".xml":   ("xml",        "metadata",            True),
    ".yaml":  ("yaml",       "metadata",            True),
    ".yml":   ("yaml",       "metadata",            True),

    # Leftover multi-part archive fragments (shouldn't be placed as data)
    ".part":  ("none",       "archive_fragment",    False),
    **{
        f".z{i:02d}": ("none", "archive_fragment", False)
        for i in range(1, 100)  # .z01..z99
    },
    **{
        f".r{i:02d}": ("none", "archive_fragment", False)
        for i in range(0, 100)  # .r00..r99 (split RAR)
    },
}

README_NAMES = {"readme.md", "readme.txt", "readme", "readme.rst"}
LABEL_HINT_NAMES = {
    "labels.csv", "label.csv", "metadata.csv", "ground_truth.csv",
    "gt.csv", "annotations.csv", "classes.csv",
}


@dataclass
class FileRecord:
    path: Path  # relative to the unpacked root
    size_bytes: int
    ext: str
    reader: str        # open-source reader module, "none", or "unknown"
    modality: str      # ADDF modality tag
    readable: bool     # do we have a decoder?


@dataclass
class Inventory:
    root: Path
    files: list[FileRecord] = field(default_factory=list)
    dir_depth_max: int = 0
    total_bytes: int = 0
    ext_counts: Counter = field(default_factory=Counter)
    modality_counts: Counter = field(default_factory=Counter)
    readme_paths: list[Path] = field(default_factory=list)
    label_hint_paths: list[Path] = field(default_factory=list)
    unreadable_extensions: set[str] = field(default_factory=set)

    def summary(self) -> dict:
        return {
            "total_files": len(self.files),
            "total_bytes": self.total_bytes,
            "dir_depth_max": self.dir_depth_max,
            "ext_counts": dict(self.ext_counts.most_common(20)),
            "modality_counts": dict(self.modality_counts),
            "readme_paths": [str(p) for p in self.readme_paths],
            "label_hint_paths": [str(p) for p in self.label_hint_paths],
            "unreadable_extensions": sorted(self.unreadable_extensions),
        }


def _detect_ext(p: Path) -> str:
    """Return the longest matching extension from FORMAT_TABLE."""
    name = p.name.lower()
    for ext in sorted(FORMAT_TABLE.keys(), key=len, reverse=True):
        if name.endswith(ext):
            return ext
    # Fallback: last dotted segment
    if "." in name:
        return "." + name.rsplit(".", 1)[-1]
    return ""


def inventory_tree(root: Path) -> Inventory:
    """Walk root and build an Inventory."""
    inv = Inventory(root=root)

    for p in root.rglob("*"):
        if p.is_dir():
            rel = p.relative_to(root)
            inv.dir_depth_max = max(inv.dir_depth_max, len(rel.parts))
            continue
        if not p.is_file():
            continue

        try:
            size = p.stat().st_size
        except OSError:
            continue
        inv.total_bytes += size

        ext = _detect_ext(p)
        inv.ext_counts[ext] += 1

        fmt_info = FORMAT_TABLE.get(ext)
        if fmt_info:
            reader, modality, readable = fmt_info
        else:
            reader, modality, readable = "unknown", "other", False

        rel_path = p.relative_to(root)
        inv.files.append(FileRecord(
            path=rel_path, size_bytes=size, ext=ext,
            reader=reader, modality=modality, readable=readable,
        ))

        inv.modality_counts[modality] += 1
        if not readable and ext:
            inv.unreadable_extensions.add(ext)

        lower = p.name.lower()
        if lower in README_NAMES:
            inv.readme_paths.append(rel_path)
        if lower in LABEL_HINT_NAMES:
            inv.label_hint_paths.append(rel_path)

    return inv


def sample_text(root: Path, paths: list[Path], max_chars: int = 4000) -> str:
    """Concatenate readable text from README-ish files for agent context."""
    chunks = []
    remaining = max_chars
    for rel in paths:
        full = root / rel
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        snippet = text[:remaining]
        chunks.append(f"--- {rel} ---\n{snippet}")
        remaining -= len(snippet)
        if remaining <= 0:
            break
    return "\n\n".join(chunks)
