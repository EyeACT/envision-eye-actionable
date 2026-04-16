"""
envision-eye-actionable CLI.

Defaults target envision-discovery's directory layout:
  ./data/downloads/{source}/{source_id}/   — downloaded files
  ./results/{source}_eye_imaging.json      — classification metadata (optional)
  ./data/actionable/{source}/{source_id}/  — conformed ADDF output

Examples:

    # Conform every downloaded zenodo record
    envision-conform --source zenodo

    # One specific record
    envision-conform --source zenodo --source-id 4521044

    # Every source under ./data/downloads/
    envision-conform --all-sources

    # Enable the Gemma 4 agent fallback
    envision-conform --source zenodo --agent-model ~/models/gemma-4-e4b-it-q4.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import conform_record, conform_source
from .agent import AgentConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="envision-conform",
        description="ADDF conformer for eye imaging datasets (envision-eye-actionable)",
    )
    parser.add_argument(
        "--source", default="zenodo",
        help="Source name to conform (default: zenodo)",
    )
    parser.add_argument(
        "--source-id",
        help="Conform only this record; otherwise conform every record in --source",
    )
    parser.add_argument(
        "--all-sources", action="store_true",
        help="Conform every source directory under --downloads-dir",
    )
    parser.add_argument(
        "--downloads-dir", default="./data/downloads",
        help="Root of downloaded data (default: ./data/downloads)",
    )
    parser.add_argument(
        "--output-dir", default="./data/actionable",
        help="Root for conformed ADDF trees (default: ./data/actionable)",
    )
    parser.add_argument(
        "--results-dir", default="./results",
        help="envision-discovery results/ dir for classification metadata "
             "(default: ./results; ignored if absent)",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of hardlinking (portable across filesystems, uses 2x disk)",
    )
    parser.add_argument(
        "--agent-model",
        help="Path to a Gemma 4 GGUF for agent fallback on low-confidence layouts",
    )

    args = parser.parse_args(argv)

    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) if args.results_dir else None

    agent_config: AgentConfig | None = None
    if args.agent_model:
        agent_config = AgentConfig(model_path=Path(args.agent_model).expanduser())

    # Mode: --all-sources
    if args.all_sources:
        if not downloads_dir.exists():
            print(f"error: downloads dir not found: {downloads_dir}", file=sys.stderr)
            return 1
        sources = [d.name for d in sorted(downloads_dir.iterdir()) if d.is_dir()]
        if not sources:
            print(f"error: no source subdirs found in {downloads_dir}", file=sys.stderr)
            return 1
        for src in sources:
            _run_source(src, downloads_dir, output_dir, results_dir, agent_config, args.copy)
        return 0

    # Mode: single record
    if args.source_id:
        record_dir = downloads_dir / args.source / args.source_id
        if not record_dir.exists():
            print(f"error: record not found: {record_dir}", file=sys.stderr)
            return 1
        result = conform_record(
            source=args.source,
            source_id=args.source_id,
            downloaded_dir=record_dir,
            target_root=output_dir,
            source_metadata=_load_record_metadata(args.source, args.source_id, results_dir),
            agent_config=agent_config,
            copy=args.copy,
        )
        print(
            f"{args.source}/{args.source_id}: status={result.status}"
            f"  confidence={result.recipe_confidence:.2f}"
            f"  validation_ok={result.validation.get('ok') if result.validation else 'n/a'}"
        )
        return 0 if result.status != "failed" else 2

    # Mode: whole source
    return _run_source(args.source, downloads_dir, output_dir, results_dir, agent_config, args.copy)


def _run_source(
    source: str,
    downloads_dir: Path,
    output_dir: Path,
    results_dir: Path | None,
    agent_config: AgentConfig | None,
    copy: bool,
) -> int:
    source_dir = downloads_dir / source
    if not source_dir.exists():
        print(f"error: source dir not found: {source_dir}", file=sys.stderr)
        return 1

    eye_imaging_results: list[dict] = []
    if results_dir and results_dir.exists():
        eye_file = results_dir / f"{source}_eye_imaging.json"
        if eye_file.exists():
            try:
                with open(eye_file) as f:
                    eye_imaging_results = json.load(f)
                print(f"  Loaded {len(eye_imaging_results):,} {source} classification records "
                      f"← {eye_file}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  warn: failed to load {eye_file}: {e}", file=sys.stderr)

    conform_source(
        source=source,
        downloads_root=downloads_dir,
        target_root=output_dir,
        eye_imaging_results=eye_imaging_results,
        agent_config=agent_config,
        copy=copy,
    )
    return 0


def _load_record_metadata(source: str, source_id: str, results_dir: Path | None) -> dict | None:
    if not results_dir or not results_dir.exists():
        return None
    eye_file = results_dir / f"{source}_eye_imaging.json"
    if not eye_file.exists():
        return None
    try:
        with open(eye_file) as f:
            data = json.load(f)
        for r in data:
            if str(r.get("source_id")) == source_id:
                return r
    except Exception:
        return None
    return None


if __name__ == "__main__":
    sys.exit(main())
