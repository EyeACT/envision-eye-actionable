"""Archive unpacking. Handles nested and multi-part archives."""

from __future__ import annotations

import logging
import shutil
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

SINGLE_ARCHIVE_EXTS = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}
SEVEN_ZIP_EXTS = {".7z"}
RAR_EXTS = {".rar"}
# Multi-part zip uses .zip with sibling .z01..z99 parts (WinZip / 7-Zip style)
MULTIPART_ZIP_HINT = ".z01"


def _has_ext(p: Path, exts: set[str]) -> str | None:
    name = p.name.lower()
    for ext in sorted(exts, key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return None


def _is_multipart_zip(zip_path: Path) -> bool:
    return (zip_path.parent / (zip_path.stem + ".z01")).exists()


def _unpack_zip(archive: Path, dest: Path):
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def _unpack_multipart_zip(archive: Path, dest: Path):
    """Merge .zip + .z01..zNN into a single stream and extract.

    Uses `zip -FF` if available (handles truncated spanning zips); otherwise
    falls back to manual concatenation then zipfile.extract.
    """
    if shutil.which("zip"):
        import subprocess
        merged = archive.parent / f"_merged_{archive.stem}.zip"
        try:
            subprocess.run(
                ["zip", "-FF", str(archive), "--out", str(merged)],
                check=True, capture_output=True,
            )
            with zipfile.ZipFile(merged) as zf:
                zf.extractall(dest)
        finally:
            if merged.exists():
                merged.unlink()
        return

    # Fallback: concatenate parts in numeric order, then unzip
    parts = sorted(archive.parent.glob(f"{archive.stem}.z*"))
    parts.append(archive)  # main .zip goes last by convention
    merged = archive.parent / f"_merged_{archive.stem}.zip"
    with open(merged, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
    try:
        with zipfile.ZipFile(merged) as zf:
            zf.extractall(dest)
    finally:
        if merged.exists():
            merged.unlink()


def _unpack_tar(archive: Path, dest: Path):
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")  # filter="data" is safe in 3.12+


def _unpack_7z(archive: Path, dest: Path):
    import py7zr
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(dest)


def _unpack_rar(archive: Path, dest: Path):
    import rarfile
    with rarfile.RarFile(archive) as rf:
        rf.extractall(dest)


def unpack_archive(archive: Path, dest: Path) -> bool:
    """Unpack a single archive to dest. Returns True on success."""
    dest.mkdir(parents=True, exist_ok=True)

    if _has_ext(archive, SINGLE_ARCHIVE_EXTS):
        name = archive.name.lower()
        try:
            if name.endswith(".zip"):
                if _is_multipart_zip(archive):
                    _unpack_multipart_zip(archive, dest)
                else:
                    _unpack_zip(archive, dest)
            else:
                _unpack_tar(archive, dest)
            return True
        except Exception as e:
            logger.warning(f"Failed to unpack {archive}: {e}")
            return False

    if _has_ext(archive, SEVEN_ZIP_EXTS):
        try:
            _unpack_7z(archive, dest)
            return True
        except Exception as e:
            logger.warning(f"Failed to 7z-unpack {archive}: {e}")
            return False

    if _has_ext(archive, RAR_EXTS):
        try:
            _unpack_rar(archive, dest)
            return True
        except Exception as e:
            logger.warning(f"Failed to rar-unpack {archive}: {e}")
            return False

    return False


def unpack_tree(root: Path, dest: Path, max_depth: int = 3) -> Path:
    """Recursively unpack archives starting from root.

    Archives found inside extracted content are unpacked in place (up to max_depth).
    Returns the dest path — the fully unpacked tree.
    """
    dest.mkdir(parents=True, exist_ok=True)

    # Copy-by-symlink any non-archive files into dest to preserve the source layout
    # For archives at the top level, extract them directly
    for entry in root.iterdir():
        if entry.is_dir():
            # Recursively mirror directory
            _mirror_with_unpack(entry, dest / entry.name, depth=1, max_depth=max_depth)
        elif entry.is_file():
            _handle_file(entry, dest, depth=1, max_depth=max_depth)

    return dest


def _mirror_with_unpack(src: Path, dst: Path, depth: int, max_depth: int):
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_dir():
            _mirror_with_unpack(entry, dst / entry.name, depth + 1, max_depth)
        elif entry.is_file():
            _handle_file(entry, dst, depth, max_depth)


def _handle_file(src: Path, dst_dir: Path, depth: int, max_depth: int):
    """Either link the file or unpack it in place (if it's an archive)."""
    is_archive = (
        _has_ext(src, SINGLE_ARCHIVE_EXTS)
        or _has_ext(src, SEVEN_ZIP_EXTS)
        or _has_ext(src, RAR_EXTS)
    )

    if is_archive and depth <= max_depth:
        # Extract into a sibling directory named after the archive stem
        stem = src.name
        for ext in sorted(SINGLE_ARCHIVE_EXTS | SEVEN_ZIP_EXTS | RAR_EXTS, key=len, reverse=True):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        target = dst_dir / stem
        if unpack_archive(src, target):
            return
        # Fall through to link the archive itself if unpack failed

    # Link (hardlink when possible, else copy)
    dst = dst_dir / src.name
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        # Cross-device or filesystem without hardlink support
        shutil.copy2(src, dst)
