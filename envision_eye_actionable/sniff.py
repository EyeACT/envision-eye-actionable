"""Rule-based layout detection.

Given an Inventory, propose an ADDF conversion recipe: which source globs
go into which ADDF directory, plus modality tags and the task_type.

The recipe is a list of placements:
    {"glob": "Raw/**/*.jpg", "addf_dir": "retinal_photography",
     "modality": "retinal_photography", "role": "image"}

A sniffer match is confident (>=0.8) or tentative (<0.8). Tentative matches
should fall through to the agent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .inventory import Inventory, FileRecord


@dataclass
class Placement:
    glob: str                # source glob relative to the unpacked root
    addf_dir: str            # target directory inside the ADDF tree
    modality: str            # retinal_photography, retinal_oct, etc.
    role: str                # "image", "mask", "label", "documentation", "raw"
    directory_type: str = "modality"  # modality / dataType / device


@dataclass
class Recipe:
    task_type: str                 # classification / segmentation / detection / raw
    placements: list[Placement] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Rules. Each returns (Recipe, confidence 0..1) or None.
# ──────────────────────────────────────────────────────────────────────


def _image_files(inv: Inventory) -> list[FileRecord]:
    return [f for f in inv.files if f.modality in (
        "retinal_photography", "retinal_imaging", "retinal_oct", "volumetric_imaging"
    )]


def _rule_image_folder(inv: Inventory) -> Recipe | None:
    """Classification dataset: <root>/<class_name>/*.jpg style."""
    imgs = [f for f in inv.files if f.reader in ("pillow", "tifffile")]
    if len(imgs) < 10:
        return None

    # Check if images are grouped under top-level class directories
    class_dirs = defaultdict(int)
    for f in imgs:
        if len(f.path.parts) >= 2:
            class_dirs[f.path.parts[0]] += 1

    if len(class_dirs) < 2:
        return None
    # Every top-level dir should contain images
    if min(class_dirs.values()) < 3:
        return None

    # Confidence based on how cleanly the split works
    top_level_dirs = {f.path.parts[0] for f in imgs if len(f.path.parts) >= 2}
    if len(top_level_dirs) >= 2 and len(top_level_dirs) <= 50:
        conf = 0.9
    else:
        conf = 0.6

    placements = [
        Placement(
            glob=f"{d}/**/*",
            addf_dir=f"retinal_photography/{_safe(d)}",
            modality="retinal_photography",
            role="image",
        )
        for d in sorted(class_dirs)
    ]

    return Recipe(
        task_type="classification",
        placements=placements,
        confidence=conf,
        notes=[f"Detected {len(class_dirs)} top-level class directories: "
               f"{', '.join(sorted(class_dirs)[:5])}"],
    )


def _rule_paired_image_mask(inv: Inventory) -> Recipe | None:
    """Segmentation: images/ + masks/ (or labels/) with matching filenames."""
    ALIASES = {
        "images": ("images", "image", "img", "imgs", "raw", "original"),
        "masks":  ("masks", "mask", "labels", "label", "gt", "groundtruth",
                   "ground_truth", "annotations"),
    }

    top_dirs = defaultdict(list)
    for f in inv.files:
        if len(f.path.parts) >= 2 and f.reader in ("pillow", "tifffile"):
            top_dirs[f.path.parts[0].lower()].append(f)

    img_dir = next((d for d in top_dirs if d in ALIASES["images"]), None)
    mask_dir = next((d for d in top_dirs if d in ALIASES["masks"]), None)

    if not img_dir or not mask_dir:
        return None
    if not top_dirs[img_dir] or not top_dirs[mask_dir]:
        return None

    img_stems = {f.path.stem for f in top_dirs[img_dir]}
    mask_stems = {f.path.stem for f in top_dirs[mask_dir]}
    overlap = img_stems & mask_stems

    if len(overlap) < min(5, len(img_stems) // 2):
        return None

    conf = 0.9 if len(overlap) >= len(img_stems) * 0.8 else 0.7

    return Recipe(
        task_type="segmentation",
        confidence=conf,
        placements=[
            Placement(
                glob=f"{img_dir}/**/*",
                addf_dir="retinal_photography/image",
                modality="retinal_photography", role="image",
            ),
            Placement(
                glob=f"{mask_dir}/**/*",
                addf_dir="retinal_photography/mask",
                modality="retinal_photography", role="mask",
            ),
        ],
        notes=[f"Paired {img_dir}/ and {mask_dir}/ with {len(overlap)} matching filenames"],
    )


def _rule_modality_bucket(inv: Inventory) -> Recipe:
    """Catch-all: bucket files by detected modality.

    Always returns a recipe — even when nothing is recognizable — so
    sniff() never returns None. Confidence drops to 0.2 when we couldn't
    find any readable modality; the agent pass should pick that up.
    """
    by_modality = defaultdict(list)
    for f in inv.files:
        if f.modality in ("other", "documentation", "metadata", "archive_fragment"):
            continue
        by_modality[f.modality].append(f)

    placements = []
    for mod, files in sorted(by_modality.items()):
        # Find the deepest common ancestor directory for files in this modality
        glob = _common_dir_glob(files)
        placements.append(Placement(
            glob=glob, addf_dir=mod, modality=mod, role="raw",
        ))

    # Documentation placement
    doc_files = [f for f in inv.files if f.modality == "documentation"]
    if doc_files:
        placements.append(Placement(
            glob=_common_dir_glob(doc_files),
            addf_dir="documentation",
            modality="documentation", role="documentation",
            directory_type="dataType",
        ))

    # Archive fragments — leftover multi-part pieces that weren't successfully
    # reassembled. Park them under _unreassembled/ so they aren't silently
    # treated as data files.
    frag_files = [f for f in inv.files if f.modality == "archive_fragment"]
    if frag_files:
        placements.append(Placement(
            glob=_common_dir_glob(frag_files),
            addf_dir="_unreassembled",
            modality="archive_fragment", role="raw",
            directory_type="dataType",
        ))

    if not by_modality and not doc_files and not frag_files:
        # Nothing placeable at all — emit a bare recipe so callers don't crash.
        return Recipe(
            task_type="raw",
            placements=[],
            confidence=0.1,
            notes=["No placeable files detected; needs agent or manual review"],
        )

    confidence = 0.5 if by_modality else 0.2
    notes = []
    if by_modality:
        notes.append(f"Bucketed by detected modality ({', '.join(by_modality)})")
    if frag_files:
        notes.append(f"{len(frag_files)} archive fragment(s) parked under _unreassembled/")
    return Recipe(
        task_type="raw",
        placements=placements,
        confidence=confidence,
        notes=notes,
    )


def _common_dir_glob(files: list[FileRecord]) -> str:
    """Find the most common top-level directory for these files; return a glob."""
    if not files:
        return "**/*"
    top_dirs = defaultdict(int)
    for f in files:
        if len(f.path.parts) >= 2:
            top_dirs[f.path.parts[0]] += 1
        else:
            top_dirs["."] += 1
    # Pick the dominant top dir if it accounts for most files
    total = sum(top_dirs.values())
    dom = max(top_dirs.items(), key=lambda kv: kv[1])
    if dom[1] >= total * 0.8 and dom[0] != ".":
        return f"{dom[0]}/**/*"
    return "**/*"


def _safe(name: str) -> str:
    """Make a directory name ADDF-safe (lowercase, replace spaces/slashes)."""
    out = name.lower().strip().replace(" ", "_").replace("-", "_")
    return "".join(c for c in out if c.isalnum() or c == "_") or "unnamed"


# ──────────────────────────────────────────────────────────────────────


RULES = [
    _rule_paired_image_mask,   # most specific first
    _rule_image_folder,
    _rule_modality_bucket,     # catch-all last
]


def sniff(inv: Inventory) -> Recipe:
    """Run all rules; return the highest-confidence match.

    _rule_modality_bucket always returns a low-confidence recipe, so `sniff`
    never returns None — callers decide whether to accept based on confidence.
    """
    best: Recipe | None = None
    for rule in RULES:
        try:
            result = rule(inv)
        except Exception as e:  # noqa: BLE001
            result = None
            _log_warn(f"Rule {rule.__name__} raised {e!r}")
        if result is None:
            continue
        if best is None or result.confidence > best.confidence:
            best = result
    assert best is not None, "modality bucket should always return a recipe"
    return best


def _log_warn(msg: str):
    import logging
    logging.getLogger(__name__).warning(msg)
