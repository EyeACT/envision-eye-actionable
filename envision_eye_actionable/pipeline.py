"""Top-level conformer pipeline.

Orchestrates: unpack → inventory → agent → materialize → validate.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import agent as agent_mod
from . import materialize as mat_mod
from . import unpack as unpack_mod
from . import validate as validate_mod
from .inventory import inventory_tree

logger = logging.getLogger(__name__)


@dataclass
class ConformResult:
    source_id: str
    status: str                # "ok" or "failed"
    recipe_confidence: float
    report_path: Path | None
    validation: dict | None


def conform_record(
    source: str,
    source_id: str,
    downloaded_dir: Path,
    target_root: Path,
    agent_config: agent_mod.AgentConfig,
    source_metadata: dict | None = None,
    copy: bool = False,
    tmp_unpack_dir: Path | None = None,
) -> ConformResult:
    """Conform one downloaded record.

    Args:
        source: e.g. "zenodo".
        source_id: e.g. "14926762".
        downloaded_dir: data/downloads/{source}/{source_id}/.
        target_root: base for conformed output; record lands at target_root/{source}/{source_id}/.
        agent_config: AgentConfig with the path to the local GGUF model.
        source_metadata: optional classification result dict (title/desc/keywords).
        copy: if True, copy files instead of hardlinking.
        tmp_unpack_dir: override for the unpack scratch dir.

    Returns:
        ConformResult.
    """
    out_dir = target_root / source / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Unpack archives into a scratch tree
    tmp_parent = tmp_unpack_dir or target_root.parent / ".conform_tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{source_id}_", dir=tmp_parent) as td:
        unpacked_root = Path(td) / "unpacked"
        unpack_mod.unpack_tree(downloaded_dir, unpacked_root)

        # 2) Inventory
        inv = inventory_tree(unpacked_root)

        # 3) Agent proposes a recipe
        recipe = agent_mod.propose_recipe(inv, source_metadata, agent_config)
        if recipe is None:
            return ConformResult(
                source_id=source_id,
                status="failed",
                recipe_confidence=0.0,
                report_path=None,
                validation={"ok": False, "error": "Agent returned no recipe"},
            )

        # 4) Materialize
        report = mat_mod.materialize(
            inv, recipe,
            source_root=unpacked_root,
            target_root=out_dir,
            copy=copy,
        )

        # 5) Validate
        validation = validate_mod.validate_tree(out_dir)

    status = "ok" if validation.get("ok") else "failed"

    return ConformResult(
        source_id=source_id,
        status=status,
        recipe_confidence=recipe.confidence,
        report_path=out_dir / "conform_report.json",
        validation=validation,
    )


def conform_source(
    source: str,
    downloads_root: Path,
    target_root: Path,
    agent_config: agent_mod.AgentConfig,
    eye_imaging_results: list[dict] | None = None,
    copy: bool = False,
) -> list[ConformResult]:
    """Conform every downloaded record under downloads_root/{source}/."""
    source_dir = downloads_root / source
    if not source_dir.exists():
        logger.warning(f"No download dir for {source} at {source_dir}")
        return []

    meta_by_id = {str(r.get("source_id")): r for r in (eye_imaging_results or [])}

    results = []
    record_dirs = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    record_dirs.sort()

    for i, rec_dir in enumerate(record_dirs, 1):
        sid = rec_dir.name
        print(f"\n  [{i}/{len(record_dirs)}] Conforming {source}/{sid}", flush=True)
        try:
            result = conform_record(
                source=source,
                source_id=sid,
                downloaded_dir=rec_dir,
                target_root=target_root,
                agent_config=agent_config,
                source_metadata=meta_by_id.get(sid),
                copy=copy,
            )
            print(f"    status={result.status}  confidence={result.recipe_confidence:.2f}"
                  f"  validation_ok={result.validation.get('ok') if result.validation else 'n/a'}",
                  flush=True)
            results.append(result)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Conform failed for {sid}")
            print(f"    FAILED: {e}", flush=True)
            results.append(ConformResult(
                source_id=sid, status="failed", recipe_confidence=0.0,
                report_path=None, validation={"ok": False, "error": str(e)},
            ))

    # Summary
    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"\n  Conform summary [{source}]: {ok} ok, {failed} failed", flush=True)

    # Write top-level run log
    log_path = target_root / source / "_conform_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": source,
            "records": [
                {
                    "source_id": r.source_id,
                    "status": r.status,
                    "recipe_confidence": r.recipe_confidence,
                    "validation_ok": r.validation.get("ok") if r.validation else None,
                }
                for r in results
            ],
            "summary": {"ok": ok, "failed": failed},
        }, f, indent=2)

    return results
