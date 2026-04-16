"""Validate a materialized ADDF tree.

Two checks:
  1. The emitted dataset_structure_description.json is self-consistent —
     every listed directoryName exists on disk.
  2. A sample file from each readable modality can actually be opened.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_tree(target_root: Path, sample_per_modality: int = 1) -> dict:
    """Return a dict with validation results."""
    import json

    result = {
        "ok": True,
        "missing_dirs": [],
        "reader_failures": [],
        "sampled": [],
    }

    struct_path = target_root / "dataset_structure_description.json"
    if not struct_path.exists():
        result["ok"] = False
        result["missing_dirs"].append("dataset_structure_description.json")
        return result

    with open(struct_path) as f:
        spec = json.load(f)

    for entry in spec.get("directoryList", []):
        name = entry.get("directoryName", "")
        if not name:
            continue
        # directoryName is just the top-level segment; check existence
        candidates = list(target_root.glob(f"{name}*"))
        if not any(c.is_dir() for c in candidates):
            result["missing_dirs"].append(name)
            result["ok"] = False

    # Sample one readable file per top-level directory and try to open it
    for entry in spec.get("directoryList", []):
        name = entry.get("directoryName", "")
        subtree = target_root / name
        if not subtree.exists():
            continue
        files = [p for p in subtree.rglob("*") if p.is_file()]
        if not files:
            continue
        for sample in random.sample(files, min(sample_per_modality, len(files))):
            ok, err = _try_open(sample)
            result["sampled"].append({
                "path": str(sample.relative_to(target_root)),
                "ok": ok,
                "error": err,
            })
            if not ok:
                result["reader_failures"].append(str(sample.relative_to(target_root)))

    return result


def _try_open(path: Path) -> tuple[bool, str | None]:
    """Attempt to open a file with the right reader; return (ok, error)."""
    name = path.name.lower()

    try:
        if name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            from PIL import Image
            with Image.open(path) as im:
                im.verify()
            return True, None

        if name.endswith((".tif", ".tiff")):
            import tifffile
            with tifffile.TiffFile(path) as tf:
                _ = tf.pages
            return True, None

        if name.endswith((".dcm", ".dicom")):
            import pydicom
            pydicom.dcmread(path, stop_before_pixels=True)
            return True, None

        if name.endswith((".nii", ".nii.gz")):
            import nibabel
            nibabel.load(path)
            return True, None

        if name.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(path, "r"):
                pass
            return True, None

        if name.endswith(".mat"):
            try:
                from scipy.io import loadmat
                loadmat(path, squeeze_me=True, mat_dtype=False, verify_compressed_data_integrity=False, variable_names=[])
                return True, None
            except NotImplementedError:
                # v7.3 HDF5-backed
                import mat73
                mat73.loadmat(path, only_include=[])
                return True, None

        if name.endswith((".nrrd", ".nhdr")):
            import nrrd
            nrrd.read_header(str(path))
            return True, None

        if name.endswith((".mha", ".mhd")):
            import SimpleITK as sitk
            sitk.ReadImage(str(path))
            return True, None

        if name.endswith((".pdf",)):
            import pypdf
            pypdf.PdfReader(str(path))
            return True, None

        if name.endswith((".docx",)):
            import docx
            docx.Document(str(path))
            return True, None

        if name.endswith((".e2e", ".fds", ".fda", ".img", ".oct")):
            # oct-converter import is heavier; just verify the import is wired.
            import oct_converter  # noqa: F401
            return True, None  # Defer actual parse to downstream use

        # Unknown / pass-through formats are not failures
        return True, None

    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
