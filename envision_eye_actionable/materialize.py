"""Turn a Recipe into an on-disk directory tree (AI-READI schema)."""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from .inventory import Inventory, FileRecord, FORMAT_TABLE
from .recipe import Recipe, Placement

logger = logging.getLogger(__name__)

SCHEMA_URL = "https://schema.aireadi.org/v0.1.0/dataset_structure_description.json"


def materialize(
    inv: Inventory,
    recipe: Recipe,
    source_root: Path,
    target_root: Path,
    copy: bool = False,
) -> dict:
    """Apply a recipe — place every file into the ADDF tree.

    Args:
        inv: Inventory of the unpacked source tree.
        recipe: Proposed placements.
        source_root: Unpacked source (where inv.files are relative to).
        target_root: Output root, e.g. data/actionable/zenodo/14926762/.
        copy: If False (default), hardlink files (saves disk). If True, copy.

    Returns:
        A report dict: which files were placed where, which were orphaned,
        and the emitted dataset_structure_description.json path.
    """
    target_root.mkdir(parents=True, exist_ok=True)

    # Map every file to at most one placement (first match wins, in recipe order)
    placed: dict[Path, tuple[Placement, Path]] = {}
    orphan: list[FileRecord] = []

    for f in inv.files:
        match = _find_placement(f.path, recipe.placements)
        if match is None:
            orphan.append(f)
            continue
        dest = target_root / match.addf_dir / f.path
        placed[f.path] = (match, dest)

    # Execute placements
    link_stats = {"hardlinked": 0, "copied": 0, "failed": 0}
    for rel_path, (placement, dest) in placed.items():
        src = source_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        try:
            if copy:
                shutil.copy2(src, dest)
                link_stats["copied"] += 1
            else:
                try:
                    dest.hardlink_to(src)
                    link_stats["hardlinked"] += 1
                except OSError:
                    shutil.copy2(src, dest)
                    link_stats["copied"] += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to place {rel_path} → {dest}: {e}")
            link_stats["failed"] += 1

    # Orphans — stash them under _unplaced/ so nothing is silently lost
    for f in orphan:
        dest = target_root / "_unplaced" / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        src = source_root / f.path
        try:
            if copy:
                shutil.copy2(src, dest)
            else:
                try:
                    dest.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dest)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to stash orphan {f.path}: {e}")

    # Emit dataset_structure_description.json for the emitted tree
    struct_doc = _build_structure_descriptor(recipe, placed, inv, target_root)
    struct_path = target_root / "dataset_structure_description.json"
    with open(struct_path, "w", encoding="utf-8") as fp:
        json.dump(struct_doc, fp, indent=2)

    # Emit conform_report.json for debuggability
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "task_type": recipe.task_type,
        "recipe_confidence": recipe.confidence,
        "recipe_notes": recipe.notes,
        "placements": [
            {
                "source_path": str(rel),
                "placement_dir": pl.addf_dir,
                "modality": pl.modality,
                "role": pl.role,
            }
            for rel, (pl, _) in placed.items()
        ][:2000],
        "orphan_count": len(orphan),
        "orphan_extensions": sorted({f.ext for f in orphan if f.ext}),
        "link_stats": link_stats,
        "inventory_summary": inv.summary(),
    }
    report_path = target_root / "conform_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report


def _find_placement(rel_path: Path, placements: list[Placement]) -> Placement | None:
    s = str(rel_path).replace("\\", "/")
    for pl in placements:
        if fnmatch.fnmatch(s, pl.glob):
            return pl
        # Also accept dir-prefix match: "Raw/**/*" should match "Raw/sub/x.jpg"
        if pl.glob.endswith("/**/*"):
            prefix = pl.glob[: -len("/**/*")]
            if s.startswith(prefix + "/"):
                return pl
    return None


def _build_structure_descriptor(
    recipe: Recipe,
    placed: dict,
    inv: Inventory,
    target_root: Path,
) -> dict:
    """Emit an ADDF dataset_structure_description.json for the materialized tree.

    Placements that share a top-level directory collapse into one entry
    with nested directoryList[] children for each subdirectory.
    """
    # Group placements by top-level directory
    top_level: dict[str, dict] = {}
    for _rel_path, (pl, _) in placed.items():
        parts = pl.addf_dir.split("/")
        top = parts[0]
        sub = "/".join(parts[1:]) if len(parts) > 1 else None

        if top not in top_level:
            related_term = _related_term_for(pl.modality)
            top_level[top] = {
                "directoryName": top,
                "directoryType": pl.directory_type,
                "directoryDescription": f"{pl.modality} data",
                "relatedTerm": [related_term] if related_term else [],
                "directoryList": [],
                "_seen_subs": set(),
            }

        if sub and sub not in top_level[top]["_seen_subs"]:
            top_level[top]["directoryList"].append({
                "directoryName": sub,
                "directoryType": "dataType",
                "directoryDescription": f"{pl.modality} — {pl.role} ({sub})",
            })
            top_level[top]["_seen_subs"].add(sub)

    # Strip internal bookkeeping and drop empty directoryList
    directory_list = []
    for entry in top_level.values():
        entry.pop("_seen_subs", None)
        if not entry["directoryList"]:
            entry.pop("directoryList")
        directory_list.append(entry)

    return {
        "schema": SCHEMA_URL,
        "directoryList": directory_list,
        "metadataFileList": [
            {
                "metadataFileName": "dataset_structure_description.json",
                "metadataFileDescription": "ADDF structure descriptor for this dataset",
            },
            {
                "metadataFileName": "conform_report.json",
                "metadataFileDescription": "ENVISION conformer placement report",
            },
        ],
    }


def _related_term_for(modality: str) -> dict | None:
    """Map our internal modality tag to a relatedTerm entry with an ontology code."""
    MAP = {
        "retinal_oct": ("Optical Coherence Tomography", "D041623",
                        "Medical Subject Headings (MeSH)",
                        "https://meshb.nlm.nih.gov/record/ui?ui=D041623"),
        "retinal_photography": ("Fundus Photography", "D005654",
                        "Medical Subject Headings (MeSH)",
                        "https://meshb.nlm.nih.gov/record/ui?ui=D005654"),
        "retinal_imaging": ("Retinal Imaging", "C168215",
                        "NCI Thesaurus (NCIT)",
                        "https://ncit.nci.nih.gov/ncitbrowser/pages/concept_details.jsf?code=C168215"),
        "volumetric_imaging": ("Volumetric Imaging", "C188577",
                        "NCI Thesaurus (NCIT)",
                        "https://ncit.nci.nih.gov/ncitbrowser/pages/concept_details.jsf?code=C188577"),
    }
    hit = MAP.get(modality)
    if not hit:
        return None
    term, code, scheme, url = hit
    return {
        "relatedTermValue": term,
        "relatedTermIdentifier": [{
            "relatedTermClassificationCode": code,
            "relatedTermScheme": scheme,
            "relatedTermValueURI": url,
        }],
    }


def _ext(path_str: str) -> str:
    name = path_str.lower()
    for ext in sorted(FORMAT_TABLE.keys(), key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return ""
