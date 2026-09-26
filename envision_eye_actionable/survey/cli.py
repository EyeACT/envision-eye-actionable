"""envision-survey: classify every image-bearing Zenodo record of a scrape.

Subcommands:

    envision-survey run          fetch, classify, describe in one process (resumable)
    envision-survey pipeline     fetch + process + monitor as three processes (the full pull)
    envision-survey fetch        producer: fetch records into the spool (disk high-water mark)
    envision-survey process      consumer: classify spooled records, write results, delete the spool dir
    envision-survey monitor      disk, spool, queue, rates and ETA into <state>/status.json
    envision-survey metadata     prefetch record JSON and DataCite JSON into the metadata cache
    envision-survey excel        build the workbook from survey_results.jsonl (one or several runs)
    envision-survey partition    id lists by expected mode (local, remote_zip, download, none)
    envision-survey keep-backfill-ids  ids finished before --keep-dir existed that hold an eye image
    envision-survey enrich-access  access facts (embargo, request settings, license) and CMDS of the
                                   restricted and embargoed records, into a supplement dir for excel
    envision-survey export-onnx  export the timm checkpoint to ONNX (needs torch + timm)

The full pull (see docs/survey.md):

    envision-survey pipeline --model /path/model.onnx --scrape results/zenodo_full.json \
        --metadata-cache-dir data/metadata/zenodo_full --out-dir results/survey_pull \
        --spool-dir data/survey_spool --state-dir results/survey_pull_state

Defaults follow the envision-discovery layout, so from the envision-discovery checkout root:

    envision-survey run --limit 3
    envision-survey run --ids 10043461 7505822
    envision-survey excel --out ~/zenodo_modality_survey.xlsx

Two processes against Zenodo at once (each with its own out and scratch dir,
sharing the 429 cooldown, the combined request budget --shared-rpm and the
disk floor through --shared-state-dir; --rpm gives each its share). Records without
local metadata (ids_unknown) may need remote zip sampling, hundreds of
requests each, so they go to the process with the larger budget:

    envision-survey partition --out-dir results/parts
    envision-survey run --ids-file results/parts/ids_remote_zip.txt --ids-file results/parts/ids_unknown.txt \\
        --rpm 90 --shared-state-dir results/zenodo_state \\
        --out-dir results/survey_remote --scratch-dir data/scratch_remote
    envision-survey run --ids-file results/parts/ids_download.txt --ids-file results/parts/ids_local.txt \\
        --ids-file results/parts/ids_none.txt --rpm 20 --shared-state-dir results/zenodo_state \\
        --out-dir results/survey_download --scratch-dir data/scratch_download
    envision-survey excel --results-dir results/survey_remote --results-dir results/survey_download

The model has no built-in default: pass --model, or set the
ENVISION_SURVEY_MODEL environment variable to the .onnx file (kept outside
the repository, with its .json sidecar next to it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# The classifier (for example the flagship regnety_004 distilled from
# synthetic-only training, exported with export-onnx; its .json sidecar holds
# the checkpoint sha256 and the parity numbers) lives outside the repo.
MODEL_ENV = "ENVISION_SURVEY_MODEL"


def _default_model() -> Path | None:
    v = os.environ.get(MODEL_ENV, "").strip()
    return Path(v) if v else None


def _add_run(sub):
    _add_common(sub.add_parser("run", help="classify records (resumable; one JSONL line per record)"))
    for role, text in (("pipeline", "start fetch, process and monitor as three processes (restarts a crashed "
                                    "fetch or process with backoff)"),
                       ("fetch", "producer: fetch records into the spool dir"),
                       ("process", "consumer: classify the spooled records and delete their spool dirs"),
                       ("monitor", "write disk, spool, queue, rate and ETA status every --interval s"),
                       ("metadata", "prefetch record and DataCite JSON of the scrape into the metadata cache")):
        _add_common(sub.add_parser(role, help=text))


def _add_common(p):
    p.add_argument("--model", type=Path, default=_default_model(),
                   help=f"ONNX model from export-onnx (required unless ${MODEL_ENV} is set)")
    p.add_argument("--scrape", type=Path, default=Path("./results/zenodo_all_results.json"),
                   help="unfiltered scrape JSON (default: ./results/zenodo_all_results.json)")
    p.add_argument("--downloads-dir", type=Path, default=None,
                   help="existing downloads, used in place and never modified (default: ./data/downloads/zenodo; "
                        "an explicitly given dir must exist)")
    p.add_argument("--metadata-dir", type=Path, default=Path("./data/metadata/zenodo"),
                   help="discovery's raw Zenodo record JSONs (default: ./data/metadata/zenodo)")
    p.add_argument("--out-dir", type=Path, default=Path("./results/survey"),
                   help="results JSONL, CMDS JSON, predictions, caches (default: ./results/survey)")
    p.add_argument("--scratch-dir", type=Path, default=Path("./data/survey_scratch"),
                   help="per-record download/extract scratch under <dir>/records, deleted after each record "
                        "(default: ./data/survey_scratch; keep it on the big disk). Must be new, empty or "
                        "created by an earlier survey run (marker file .envision_survey_scratch)")
    p.add_argument("--max-images", type=int, default=2000,
                   help="classification cap per record for local and downloaded files (default 2000)")
    p.add_argument("--remote-cap", type=int, default=300,
                   help="images sampled per record from zips read remotely, member by member (default 300)")
    p.add_argument("--max-download-gb", type=float, default=15.0,
                   help="download non-zip files (rar/7z/tar/gz, top-level images) only when the record needs "
                        "less than this; larger records get status skipped_size (default 15)")
    p.add_argument("--series-sample-k", type=int, default=3,
                   help="top-level image files downloaded per homogeneous series of large files (same kind, "
                        "extension and name with digits masked, each at least --series-min-mb); the rest are "
                        "counted only (default 3; 0 downloads every file)")
    p.add_argument("--series-min-mb", type=float, default=64.0,
                   help="file size from which --series-sample-k applies, in MB (default 64)")
    p.add_argument("--token-file", type=Path, default=None,
                   help="file holding a Zenodo access token (one line; the ZENODO_TOKEN environment variable "
                        "wins; default ~/.config/envision-survey/zenodo_token when it exists). Sent as a Bearer "
                        "header to zenodo.org only, never logged, printed or written")
    p.add_argument("--metadata-cache-dir", type=Path, default=None,
                   help="where record metadata fetched from Zenodo is cached (legacy/ and datacite/ "
                        "subdirs; default: <out-dir>/cache)")
    p.add_argument("--no-remote-zip", action="store_true",
                   help="download missing zips whole (subject to --max-download-gb) instead of sampling remotely")
    p.add_argument("--no-archive-probe", action="store_true",
                   help="download non-zip archives whole without first reading their listing by range "
                        "requests (by default an archive whose whole listing holds no image, DICOM, volume or "
                        "nested archive member is not downloaded)")
    p.add_argument("--archive-probe-mb", type=float, default=64.0,
                   help="bytes read at most per archive probe, in MB; compressed tars are listed from their "
                        "first this many bytes (default 64)")
    p.add_argument("--archive-probe-max-requests", type=int, default=50,
                   help="range requests at most per archive probe (default 50)")
    p.add_argument("--remote-nested-max", type=int, default=20,
                   help="nested archives fetched per record when a remote zip holds zips (default 20)")
    p.add_argument("--remote-nested-gb", type=float, default=2.0,
                   help="bytes of nested archives fetched per record from remote zips, in GB; a budget of its "
                        "own, separate from --max-download-gb (default 2; each nested archive at most 0.5 GB)")
    p.add_argument("--remote-record-budget-gb", type=float, default=2.0,
                   help="bytes fetched per record from remote zips (direct members and nested archives "
                        "together), in GB; sampled members past it are not fetched (default 2)")
    p.add_argument("--remote-max-member-mb", type=float, default=200.0,
                   help="remote zip members larger than this are never sampled; another member is drawn "
                        "instead where possible (default 200)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="top-1 probability below this counts as UNCERTAIN (default 0.6)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op threads (default 2)")
    p.add_argument("--disk-floor-gb", type=float, default=80.0,
                   help="refuse a record whose download would leave less free space (default 80)")
    p.add_argument("--download-workers", type=int, default=3, help="parallel file downloads per record")
    p.add_argument("--download-all", action="store_true",
                   help="also fetch non-image file types (docs, tables, code, arrays)")
    p.add_argument("--min-eye-fraction", type=float, default=0.05,
                   help="a record has eye images (status ok) only when eye classes make up at least this "
                        "fraction of its non-mask classified images, thresholded labels (default 0.05)")
    p.add_argument("--min-class-fraction", type=float, default=0.01,
                   help="in a record with eye images, an eye class is listed as present at this fraction "
                        "of the non-mask classified images (default 0.01)")
    p.add_argument("--max-pixels", type=int, default=80_000_000,
                   help="larger images are inventoried, not decoded (default 80 MP)")
    p.add_argument("--max-depth", type=int, default=3, help="nested archive depth (default 3)")
    p.add_argument("--no-link-check", action="store_true", help="skip HTTP status checks of weblinks")
    p.add_argument("--link-interval", type=float, default=1.0, help="seconds between link checks")
    p.add_argument("--zenodo-interval", type=float, default=0.5,
                   help="minimum seconds between two Zenodo requests (default 0.5)")
    p.add_argument("--rpm", "--zenodo-rpm", dest="zenodo_rpm", type=int, default=120,
                   help="Zenodo requests per sliding minute for this process, all threads together (default "
                        "120; Zenodo allows 133 per minute per IP and User-Agent, token or not). With "
                        "--shared-state-dir (or --state-dir) the processes "
                        "together are also held to --shared-rpm, so --rpm only sets a process's share, "
                        "e.g. --rpm 90 for remote_zip and --rpm 20 for download")
    p.add_argument("--shared-state-dir", type=Path, default=None,
                   help="dir shared by survey processes running at the same time (one IP, one disk): a 429 in "
                        "one of them pauses all of them (zenodo_not_before), their requests together stay "
                        "under --shared-rpm (zenodo_requests, under a file lock), and each subtracts the "
                        "downloads the others have in flight on the same disk before checking "
                        "--disk-floor-gb (disk_reserve.*). Without it all three are per process")
    p.add_argument("--shared-rpm", type=int, default=120,
                   help="Zenodo requests per sliding minute of all the processes sharing --shared-state-dir "
                        "(or --state-dir) together (default 120); ignored without either")
    p.add_argument("--offline", action="store_true", help="no Zenodo API calls (cache and metadata dir only)")
    p.add_argument("--keep-scratch", action="store_true", help="do not delete the scratch dir (debugging)")
    p.add_argument("--no-predictions", action="store_true", help="skip per-image predictions files")
    p.add_argument("--ids", nargs="+", default=[], help="only these record ids")
    p.add_argument("--ids-file", type=Path, action="append", default=[],
                   help="only the record ids in this file (one per line, e.g. from 'partition'); repeatable, "
                        "combined with --ids")
    p.add_argument("--limit", type=int, default=None, help="process at most N pending records")
    p.add_argument("--retry-status", default="",
                   help="comma-separated statuses to reprocess (e.g. error,skipped_disk,skipped_size,crashed,"
                        "remote_listing_failed,ok_partial_download,images_unreadable)")
    p.add_argument("--triage-cap", type=int, default=50,
                   help="two-pass sampling: images classified first from each source (local and downloaded "
                        "files; remote zip members and top-level image files fetched one by one). Only a "
                        "record whose triage sample holds eye classes at --min-eye-fraction or more gets the "
                        "deep sample (up to --max-images local, --remote-cap remote), reusing the triage "
                        "images (default 50; 0: one pass)")
    g = p.add_argument_group("pipeline (fetch, process, monitor, pipeline)")
    g.add_argument("--spool-dir", type=Path, default=None,
                   help="where the producer writes records for the consumer (new, empty or created by the "
                        "survey; marker .envision_survey_spool). Never inside, around or equal to "
                        "--downloads-dir or --out-dir")
    g.add_argument("--state-dir", type=Path, default=None,
                   help="status files, role locks, logs, events, request budget and 429 cooldown of the "
                        "pipeline processes")
    g.add_argument("--spool-max-gb", type=float, default=300.0,
                   help="the producer starts no record while the spool holds more than this (default 300)")
    g.add_argument("--fetch-workers", type=int, default=4,
                   help="records the producer fetches at once, each with --download-workers streams "
                        "(default 4)")
    g.add_argument("--poll-s", type=float, default=10.0, help="polling interval of the roles (default 10)")
    g.add_argument("--order", choices=("cost", "scrape"), default="cost",
                   help="producer order: cost (records with little to download first, big archives last; "
                        "default) or scrape order")
    g.add_argument("--max-attempts", type=int, default=2,
                   help="consumer starts of one record before it gets status crashed (default 2)")
    g.add_argument("--consumer-grace-s", type=float, default=600.0,
                   help="fetch: stop when the process role has been gone this long while the producer has to "
                        "wait for it (full spool, or deep passes still possible) (default 600)")
    g.add_argument("--max-role-restarts", type=int, default=20,
                   help="pipeline: restarts of fetch or process (each) after a crash before the pipeline "
                        "gives that role up (default 20)")
    g.add_argument("--restart-backoff-s", type=float, default=10.0,
                   help="pipeline: delay before the first restart of a role, doubled per restart, at most "
                        "300 s (default 10)")
    g.add_argument("--interval", type=float, default=60.0, help="monitor: seconds between snapshots (default 60)")
    g.add_argument("--once", action="store_true", help="monitor: one snapshot, then exit")
    g.add_argument("--exit-when-idle", action="store_true",
                   help="monitor: exit when neither fetch nor process is running")
    g.add_argument("--metadata-threads", type=int, default=2, help="metadata: worker threads (default 2)")
    g.add_argument("--keep-dir", type=Path, default=None,
                   help="process: move (rename, never copy) the fetched files of every record with at least one "
                        "image classified as an eye class into <dir>/<record id>/ instead of deleting them "
                        "(new, empty or created by the survey, marker .envision_survey_keep; on the spool's "
                        "filesystem). Rows get kept, kept_reason, kept_path, kept_files, kept_bytes")
    g.add_argument("--keep-max-gb", type=float, default=200.0,
                   help="process: stop keeping new records once the keep dir holds this much (their rows get "
                        "kept false with the reason; kept records are never deleted). Keeping also stops while "
                        "the free space with an empty spool would fall under --disk-floor-gb + --spool-max-gb "
                        "(default 200; 0: no size cap)")
    g.add_argument("--refetch-keep-ids", type=Path, default=None,
                   help="fetch: fetch and classify again the record ids in this file (one per line, e.g. from "
                        "keep-backfill-ids) whose last final row was written before --keep-dir existed, so "
                        "their files reach the keep dir; they go first. The new row replaces the old one (last "
                        "line wins); rows written with retention are never redone, so the option can stay on "
                        "across restarts")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_excel(sub):
    p = sub.add_parser("excel", help="build the Excel workbook from the results JSONL")
    p.add_argument("--results", type=Path, default=Path("./results/survey/survey_results.jsonl"),
                   help="results JSONL (ignored when --results-dir is given)")
    p.add_argument("--results-dir", type=Path, action="append", default=[],
                   help="a run's --out-dir; repeat to merge the runs of several processes (rows merged by "
                        "record id, the last dir given wins; CMDS files read from each row's own dir). A dir "
                        "with access_results.jsonl (enrich-access --out-dir) is a supplement: it sets only the "
                        "access and CMDS fields of the rows and adds the Access_Requests sheet")
    p.add_argument("--out", type=Path, default=Path("./results/survey/zenodo_modality_survey.xlsx"))
    p.add_argument("--model", type=Path, default=_default_model(),
                   help=f"ONNX model whose .json sidecar goes into the README (default: ${MODEL_ENV})")
    p.add_argument("--cmds-json", choices=("auto", "embed", "paths"), default="auto",
                   help="CMDS_JSON sheet: the documents as text (embed), their file paths and sizes (paths), "
                        "or auto: embed up to 2000 records (default auto)")
    p.add_argument("--training-sources", type=Path, default=None,
                   help="CSV of the classifier's training and evaluation sources (one row per model_version, "
                        "source and class; needs a model_version column), copied to a Model_Training_Datasets "
                        "sheet and summarised in the README (default: sheet left out)")


def _add_partition(sub):
    p = sub.add_parser("partition", help="split the scrape into id lists by expected mode (no Zenodo requests)")
    p.add_argument("--scrape", type=Path, default=Path("./results/zenodo_all_results.json"))
    p.add_argument("--downloads-dir", type=Path, default=Path("./data/downloads/zenodo"))
    p.add_argument("--metadata-dir", type=Path, action="append", default=[],
                   help="dirs of legacy Zenodo record JSONs (<id>.json), searched in order; repeatable "
                        "(default: ./data/metadata/zenodo and ./results/survey/cache/legacy)")
    p.add_argument("--out-dir", type=Path, default=Path("./results/survey_partitions"),
                   help="where ids_local.txt, ids_remote_zip.txt, ids_download.txt, ids_none.txt, "
                        "ids_unknown.txt (no metadata file) and partition_summary.json go")
    p.add_argument("--no-remote-zip", action="store_true", help="as in run: zips count as downloads")
    p.add_argument("--download-all", action="store_true", help="as in run")


def _add_keep_backfill(sub):
    p = sub.add_parser("keep-backfill-ids",
                       help="list the records finished before --keep-dir existed that hold an eye image")
    p.add_argument("--out-dir", type=Path, required=True, help="the run's --out-dir (survey_results.jsonl, "
                                                                "predictions/)")
    p.add_argument("--keep-dir", type=Path, default=None, help="leave out ids already kept there (KEPT.json)")
    p.add_argument("--output", type=Path, required=True, help="where the ids go, one per line")


def _add_enrich_access(sub):
    p = sub.add_parser("enrich-access",
                       help="fetch the InvenioRDM record JSON of restricted and embargoed records: access, "
                            "embargo, request settings, license, CMDS; rows into <out-dir>/access_results.jsonl")
    p.add_argument("--scrape", type=Path, default=Path("./results/zenodo_all_results.json"))
    p.add_argument("--survey-out-dir", type=Path, default=None,
                   help="the survey's --out-dir: survey_results.jsonl picks further targets (status or "
                        "access_right restricted, embargoed, closed) and its cache/ gives legacy and DataCite "
                        "JSON; only read, never written")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="where access_results.jsonl, access_summary.json, cmds/<id>/ and cache/ go (its own "
                        "dir, e.g. results/<run>/access; never the survey's out dir)")
    p.add_argument("--read-cache-dir", type=Path, action="append", default=None,
                   help="metadata caches read before fetching (legacy/ and datacite/ subdirs); repeatable "
                        "(default: <survey-out-dir>/cache)")
    p.add_argument("--rpm", type=int, default=10,
                   help="Zenodo requests per minute of this job (default 10: a small share next to a running "
                        "survey)")
    p.add_argument("--shared-state-dir", type=Path, default=None,
                   help="the running survey's shared state dir, so both stay under --shared-rpm together and "
                        "share a 429 cooldown")
    p.add_argument("--shared-rpm", type=int, default=120)
    p.add_argument("--token-file", type=Path, default=None,
                   help="as in run (default ~/.config/envision-survey/zenodo_token when it exists)")
    p.add_argument("--ids", nargs="+", default=[], help="only these record ids (among the targets)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--refresh", action="store_true", help="fetch again the records already done")
    p.add_argument("--no-cmds", action="store_true", help="do not write the CMDS documents")
    p.add_argument("--offline", action="store_true", help="no Zenodo requests (cache only)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="envision-survey",
        description="Zenodo eye-modality survey: ONNX classifier + DICOM-aligned facts + AI-READI/CMDS metadata",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_run(sub)
    _add_excel(sub)
    _add_partition(sub)
    _add_keep_backfill(sub)
    _add_enrich_access(sub)
    from .export_onnx import add_arguments as _onnx_args
    _onnx_args(sub.add_parser("export-onnx", help="export the timm checkpoint to ONNX (needs torch + timm)"))
    args = parser.parse_args(argv)

    if args.cmd == "export-onnx":
        from .export_onnx import run as _onnx_run
        return _onnx_run(args)

    if args.cmd == "excel":
        from .excel import build_workbook, load_model_meta
        if args.model is None:
            print(f"[survey] warning: no --model (and no ${MODEL_ENV}); the README sheet gets model provenance "
                  "and parity only from the result rows", file=sys.stderr, flush=True)
        if args.results_dir:
            missing = [d for d in args.results_dir if not d.is_dir()]
            if missing:
                parser.error(f"--results-dir {missing[0]} does not exist")
            from .excel import SUPPLEMENT_NAME, split_results_dirs
            source, supplements, empty = split_results_dirs(args.results_dir)
            if empty:
                parser.error(f"--results-dir {empty[0]} holds neither survey_results.jsonl nor {SUPPLEMENT_NAME}")
            if not source:
                parser.error("--results-dir: no survey_results.jsonl in any dir given (supplements alone "
                             "make no workbook)")
        else:
            source, supplements = args.results, []
        if args.training_sources is not None:
            if not args.training_sources.is_file():
                parser.error(f"--training-sources {args.training_sources} does not exist")
            from .excel import load_training_sources
            try:
                load_training_sources(args.training_sources)
            except ValueError as e:
                parser.error(str(e))
        stats = build_workbook(source, args.out, load_model_meta(args.model), cmds_json=args.cmds_json,
                               training_sources=args.training_sources, supplements=supplements)
        print(json.dumps(stats, indent=2), flush=True)
        return 0

    if args.cmd == "keep-backfill-ids":
        from .keep import backfill_ids
        res = args.out_dir / "survey_results.jsonl"
        if not res.is_file():
            parser.error(f"no {res}")
        ids, summary = backfill_ids(res, args.out_dir / "predictions", args.keep_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(args.output.name + ".tmp")
        tmp.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
        tmp.replace(args.output)
        print(json.dumps({**summary, "output": str(args.output)}, indent=2), flush=True)
        return 0

    if args.cmd == "enrich-access":
        from .access import AccessEnricher
        from .zenodo import install_log_redaction
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        install_log_redaction()
        if not args.scrape.is_file():
            parser.error(f"--scrape {args.scrape} not found")
        if args.survey_out_dir is not None and not (args.survey_out_dir / "survey_results.jsonl").is_file():
            parser.error(f"--survey-out-dir {args.survey_out_dir} has no survey_results.jsonl")
        if args.token_file is not None and not args.token_file.expanduser().is_file():
            parser.error(f"--token-file {args.token_file} not found")
        if args.rpm < 1:
            parser.error("--rpm must be at least 1")
        try:
            enricher = AccessEnricher(args.scrape, args.survey_out_dir, args.out_dir,
                                      read_cache_dirs=args.read_cache_dir, rpm=args.rpm,
                                      shared_state_dir=args.shared_state_dir, shared_rpm=args.shared_rpm,
                                      token_file=args.token_file, offline=args.offline,
                                      write_cmds=not args.no_cmds, refresh=args.refresh)
        except ValueError as e:
            parser.error(str(e))
        stats = enricher.run(ids=args.ids or None, limit=args.limit)
        print(json.dumps(stats, indent=2), flush=True)
        return 0

    if args.cmd == "partition":
        from .runner import partition_records
        meta_dirs = args.metadata_dir or [Path("./data/metadata/zenodo"), Path("./results/survey/cache/legacy")]
        summary = partition_records(args.scrape, args.downloads_dir, meta_dirs, args.out_dir,
                                    remote_zip=not args.no_remote_zip, download_all=args.download_all)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .zenodo import install_log_redaction
    install_log_redaction()
    from .runner import SurveyConfig, run_survey
    need_model = args.cmd in ("run", "process", "pipeline")
    if need_model and args.model is None:
        parser.error(f"no model: pass --model /path/to/model.onnx or set {MODEL_ENV}")
    if need_model and not args.model.is_file():
        parser.error(f"--model {args.model} not found (export it with envision-survey export-onnx)")
    if args.downloads_dir is not None and not args.downloads_dir.is_dir():
        parser.error(f"--downloads-dir {args.downloads_dir} does not exist (wrong working directory?)")
    if args.cmd in ("fetch", "process", "pipeline") and (args.spool_dir is None or args.state_dir is None):
        parser.error(f"{args.cmd} needs --spool-dir and --state-dir")
    if args.cmd == "monitor" and args.state_dir is None:
        parser.error("monitor needs --state-dir")
    if args.token_file is not None and not args.token_file.expanduser().is_file():
        parser.error(f"--token-file {args.token_file} not found")
    downloads_dir = args.downloads_dir or Path("./data/downloads/zenodo")
    refetch_ids: list[str] = []
    if args.refetch_keep_ids is not None:
        if not args.refetch_keep_ids.is_file():
            parser.error(f"--refetch-keep-ids {args.refetch_keep_ids} not found")
        from .runner import read_ids_file
        refetch_ids = read_ids_file(args.refetch_keep_ids)
    if args.keep_dir is not None and args.cmd == "run":
        parser.error("--keep-dir works with the pipeline roles (pipeline, process), not run")
    ids = list(args.ids)
    if args.ids_file:
        from .runner import read_ids_file
        for f in args.ids_file:
            if not f.is_file():
                parser.error(f"--ids-file {f} not found")
            seen = set(ids)
            ids += [i for i in read_ids_file(f) if i not in seen]
        if not ids:
            # An empty id list would mean "every record": refuse instead.
            parser.error("--ids-file: no record ids in " + ", ".join(str(f) for f in args.ids_file))
    cfg = SurveyConfig(
        scrape_path=args.scrape, model_path=args.model or Path("model-not-needed.onnx"), out_dir=args.out_dir,
        downloads_dir=downloads_dir,
        metadata_dir=args.metadata_dir if args.metadata_dir and args.metadata_dir.exists() else None,
        scratch_dir=args.scratch_dir, max_images=args.max_images, remote_cap=args.remote_cap,
        max_download_gb=args.max_download_gb, remote_zip=not args.no_remote_zip,
        archive_probe=not args.no_archive_probe, archive_probe_mb=args.archive_probe_mb,
        archive_probe_max_requests=args.archive_probe_max_requests,
        series_sample_k=args.series_sample_k, series_min_mb=args.series_min_mb,
        token_file=args.token_file, metadata_cache_dir=args.metadata_cache_dir,
        remote_nested_max=args.remote_nested_max,
        remote_nested_gb=args.remote_nested_gb, remote_record_budget_gb=args.remote_record_budget_gb,
        remote_max_member_mb=args.remote_max_member_mb, min_eye_fraction=args.min_eye_fraction,
        zenodo_per_minute=args.zenodo_rpm, shared_state_dir=args.shared_state_dir,
        zenodo_shared_per_minute=args.shared_rpm, seed=args.seed,
        threshold=args.threshold, batch_size=args.batch_size, threads=args.threads,
        disk_floor_gb=args.disk_floor_gb, download_workers=args.download_workers,
        download_all=args.download_all, min_class_fraction=args.min_class_fraction,
        max_pixels=args.max_pixels, max_depth=args.max_depth, check_links=not args.no_link_check,
        link_interval=args.link_interval, zenodo_interval=args.zenodo_interval,
        keep_scratch=args.keep_scratch, write_predictions=not args.no_predictions,
        offline=args.offline, ids=ids, limit=args.limit,
        retry_statuses=[s.strip() for s in args.retry_status.split(",") if s.strip()],
        triage_cap=args.triage_cap, spool_dir=args.spool_dir, state_dir=args.state_dir,
        spool_max_gb=args.spool_max_gb, fetch_workers=args.fetch_workers, poll_s=args.poll_s,
        order=args.order, max_attempts=args.max_attempts, consumer_grace_s=args.consumer_grace_s,
        max_role_restarts=args.max_role_restarts, restart_backoff_s=args.restart_backoff_s,
        keep_dir=args.keep_dir, keep_max_gb=args.keep_max_gb, refetch_ids=refetch_ids,
    )
    if args.cmd == "run":
        stats = run_survey(cfg)
        print(json.dumps(stats, indent=2), flush=True)
        return 0
    from . import pipeline as pl
    if args.cmd == "metadata":
        pl.prefetch_metadata(cfg, threads=args.metadata_threads)
        return 0
    if args.cmd == "fetch":
        return pl.Producer(cfg).run()
    if args.cmd == "process":
        return pl.Processor(cfg).run()
    if args.cmd == "monitor":
        return pl.Monitor(cfg).run(args.interval, once=args.once, exit_when_idle=args.exit_when_idle)
    raw = list(argv) if argv is not None else sys.argv[1:]
    passthrough = raw[raw.index("pipeline") + 1:]
    return pl.run_pipeline(cfg, passthrough, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
