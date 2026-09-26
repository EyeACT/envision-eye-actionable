# Zenodo modality survey (`envision-survey`)

`envision-survey` classifies the images of every record in an
[envision-discovery](https://github.com/EyeACT/envision-discovery) Zenodo
scrape with the EyeACT 7-class eye-modality classifier. It then writes,
per record:

- DICOM-aligned header facts (Modality, SOP Class, laterality, dimensions,
  manufacturer, ...)
- AI-READI / CMDS metadata: `dataset_description.json` and
  `dataset_structure_description.json`
- a catalogue of external dataset links when the record has no usable images

An Excel workbook summarises the run.

The metadata (SetFit) classifier from envision-discovery is **not** used to
pick records. Every record in the scrape is processed, and its SetFit label
appears in the workbook for information only.

## Classes

| Class | Meaning | CMDS directory | DICOM Modality |
|---|---|---|---|
| CFP | color fundus photograph | `retinal_photography/cfp` | OP |
| IR | infrared reflectance (near-infrared SLO) | `retinal_photography/ir` | OP |
| PSC | ultra-widefield pseudocolor | `retinal_photography/uwf_pseudocolor` | OP |
| FAF | fundus autofluorescence | `retinal_photography/faf` | OP |
| OCT | OCT B-scan | `retinal_oct/structural_oct` | OPT |
| OCTA | OCT angiography en face | `retinal_octa/enface` | OPTENF |
| NEG | non-eye image | no directory | OT |
| UNCERTAIN | top-1 probability below the threshold | no directory | |
| MASK | segmentation mask or label map (pixel check, not the model) | no directory | |

UNCERTAIN and MASK are not modalities. MASK replaces the model's label, both
thresholded and argmax, for every image the pixel check flags as a mask
(see step 4 below); the model's own prediction stays in the per-image
predictions file.

## Install

```bash
pip install -e '.[survey]'        # onnxruntime, openpyxl, jsonschema, requests, imagecodecs
```

The survey never imports torch. The one-off ONNX export needs torch and timm.
Run it where the checkpoint lives, either through the CLI or as a standalone
file, since it has no package imports when run directly:

```bash
envision-survey export-onnx --checkpoint ckpt/student_regnety400_synthonly_distill_s2.pt \
    --arch regnety_004 --out models/regnety004_synthonly_distill_s2.onnx \
    --parity-dir data/tier3_images            # one sub-folder of real images per class
# or, on a box without this package installed:
python envision_eye_actionable/survey/export_onnx.py --checkpoint ... --out ... --parity-dir ...
```

The export also needs the `onnx` package and `onnxruntime` (required for
the parity check; the export stops at once without it). It writes
`<name>.onnx` plus a `<name>.json` sidecar that records the class order,
the checkpoint sha256, the ONNX sha256 and the preprocessing. The export's
logits are checked against PyTorch on random input and, with
`--parity-dir`, on `--parity-per-class` (default 3) seeded real images per
class through the training eval transform. The same picks are then fed
through the survey's own input path (`images.load_frame`, then
`images.preprocess`: content crop, squash to 256, resize to 224), so the
check also covers the tensors the survey will produce; standalone, the
export loads `images.py` and `constants.py` from next to itself. Both
checks refuse the export unless the largest logit difference is under 1e-4
and every argmax agrees; their numbers go into the sidecar as
`parity_real_images` and `parity_survey_preprocess` (recorded as skipped
only when the survey module cannot be imported), and the workbook README
quotes them when `excel` is given `--model`. The model
is written to a temp name and moved to `--out` only after the check
passes, so a failed export never leaves an unchecked `.onnx` behind (an
earlier file at `--out` and its sidecar stay as they were).

When the survey loads a model, it hashes the `.onnx` and refuses to run if
the hash differs from the sidecar's `onnx_sha256` (a stale sidecar or a
swapped model would otherwise put the wrong checkpoint into the workbook).
It warns when the sidecar is missing or has no parity result.

### Flagship model

The flagship model is `regnety004_synthonly_distill_s2.onnx`: timm
`regnety_004`, 7 classes (CFP, IR, PSC, FAF, OCT, OCTA, NEG), distilled
from synthetic-only training (seed 2), same preprocessing as before. Keep
it outside the repository with its `.json` sidecar next to it, and point
the CLI at it with `--model` or the `ENVISION_SURVEY_MODEL` environment
variable (there is no built-in default path). Export checks:
- checkpoint sha256 `09c50d57520ac24aab8c3045859e3bdade48388c749081e79b2c2f6484ce6902`,
  ONNX sha256 `2c6e04885c6444e10e66b7dd418c1cedeb1b8eb8cf4c8ef28ae3215aeab20eec`;
- random input: max logit difference 1.2e-6;
- 21 real tier-3 images (3 per class), training eval transform: max logit
  difference 2.4e-6, max probability difference 4.2e-7, argmax agreement
  21 of 21 (19 of 21 match the folder label);
- the same 21 images through the survey's decoding and preprocessing
  (`parity_survey_preprocess` in the sidecar): max logit difference
  5.2e-6, argmax agreement 21 of 21 (19 of 21 match the folder label).
  The re-export that recorded it produced a byte-identical `.onnx` (same
  sha256).

The earlier flagship (`regnety004_synthonly.onnx`, synthetic-only seed 1)
was checked the same way. Its export matched to 1.7e-6. On 105 labelled training images, the numpy
preprocessing matched torchvision exactly and the ONNX and PyTorch argmax
agreed on all of them. Because those training PNGs are already square, the
preprocessing was also checked on 160 non-square originals (aspect ratio
1.16 to 3.10, 30 downloaded records) against the training path rebuilt
with torchvision (content crop, squash to 256 x 256, then the
`distill_synthonly.py` eval transform): the tensors were identical (max
difference 0.000 in 8-bit units) and so were all 160 predictions.

## Run

Defaults follow the envision-discovery layout, so run from its root:

```bash
cd /path/to/envision-discovery
export ENVISION_SURVEY_MODEL=/path/outside/repo/regnety004_synthonly_distill_s2.onnx
envision-survey run --scratch-dir /path/to/scratch --limit 3    # smoke test
envision-survey run --ids 7505822 12775880                     # specific records
envision-survey excel --out results/survey/zenodo_modality_survey.xlsx
```

There is no built-in model default: `--model` is required unless
`ENVISION_SURVEY_MODEL` is set in the environment of the process that runs
the survey (a variable exported in another shell, or missing from a
`nohup` line, does not count). Pass the same model to `excel`, so the
README sheet gets the model provenance and parity numbers from the
sidecar; `excel` warns when it has none.

For the full run in the background, create the output directory before
the shell redirect needs it and put the model on the command line itself:

```bash
cd /path/to/envision-discovery
mkdir -p results/survey
nohup envision-survey run --model /path/outside/repo/regnety004_synthonly_distill_s2.onnx \
    --out-dir results/survey --scratch-dir /path/to/scratch > results/survey/run.log 2>&1 &
tail -f results/survey/run.log            # a missing model or path shows up here at once
```

### Two processes at once

Remote zip sampling is request-heavy (a listing plus one request per
sampled member, hundreds per record, few bytes each), while whole-file
downloads are bandwidth-heavy (one request per file). Running both kinds
of record in two processes overlaps them. `partition` splits the scrape by
the work each record needs, from the scrape, the files on disk and the
legacy metadata files alone (no Zenodo request):

```bash
envision-survey partition --out-dir results/survey_parts
```

It writes one id per line to `ids_remote_zip.txt` (at least one zip to
sample remotely, whatever else the record has), `ids_download.txt` (files
to download whole, no remote zip), `ids_local.txt` (everything needed is
on disk), `ids_none.txt` (no files, or none an image survey reads) and
`ids_unknown.txt` (no metadata file found in the `--metadata-dir` dirs,
default `./data/metadata/zenodo` then `./results/survey/cache/legacy`),
plus `partition_summary.json` with the counts. `--no-remote-zip` and
`--download-all` mean the same as in `run`.

Zenodo's guest limit (about 133 requests per minute) is per IP. Each
process needs its own `--out-dir` and `--scratch-dir` (the survey locks
both, with `flock` on Linux and `msvcrt` on Windows; a second process on the
same dir stops at once; where no file locking exists the survey warns and
does not sweep stale scratch dirs). Give both the same `--shared-state-dir`.
Three things are then shared through that dir:

- the 429 cooldown: a 429 in one process pauses the other as well (the
  cooldown end is written to `zenodo_not_before`), instead of the other
  one extending the per-IP penalty;
- the request budget: every request is logged in `zenodo_requests` under a
  file lock, and a request goes out only while all the processes together
  sent fewer than `--shared-rpm` (120) in the last 60 s. Two processes left
  at the default `--rpm 120` therefore cannot send 240 per minute; `--rpm`
  still caps each process on its own, so it sets a process's share (90 and
  20 below);
- the disk floor: before downloading, a process publishes the bytes it is
  about to write (`disk_reserve.<pid>.json`, removed when it finishes; a
  reservation whose process is gone is recognised by its free lock and
  deleted) and subtracts the other processes' reservations on the same
  file system from the free space, both for the download check and for
  remote member and nested archive fetches.

Without it all three are per process. The
records of `ids_unknown.txt` have no metadata file yet and may turn out to
need remote zip sampling (hundreds of requests each, about 15 minutes or
more per record at 20 per minute), so they go to the remote process:

```bash
P=results/survey_parts
nohup envision-survey run --model $MODEL --ids-file $P/ids_remote_zip.txt --ids-file $P/ids_unknown.txt \
    --rpm 90 --shared-state-dir results/zenodo_state \
    --out-dir results/survey_remote --scratch-dir /path/to/scratch_remote > remote.log 2>&1 &
nohup envision-survey run --model $MODEL --ids-file $P/ids_download.txt --ids-file $P/ids_local.txt \
    --ids-file $P/ids_none.txt --rpm 20 --shared-state-dir results/zenodo_state \
    --out-dir results/survey_download --scratch-dir /path/to/scratch_download > download.log 2>&1 &
envision-survey excel --model $MODEL --results-dir results/survey_remote \
    --results-dir results/survey_download --out results/zenodo_modality_survey.xlsx
```

`--ids-file` is repeatable and adds to `--ids`; ids files that together hold
no id stop the run (an empty list would otherwise mean every record).
`excel` with several `--results-dir` merges the rows by record id (the last
line within a file and the last dir given win, except that an unfinished
row, status `started` or `crashed`, never replaces a finished row from an
earlier dir), reads each row's CMDS files from its own dir, and names the
dir in the `results_dir` column by the shortest trailing part of its path
that the other dirs do not share (`survey_smoke/results` and
`survey_smoke2/results`, not `results` twice; never an absolute path).

Main options (`envision-survey run --help` lists all of them):

| Option | Default | Meaning |
|---|---|---|
| `--scrape` | `./results/zenodo_all_results.json` | unfiltered scrape (JSON list) |
| `--downloads-dir` | `./data/downloads/zenodo` | existing downloads, read in place, never modified; a dir given explicitly must exist |
| `--metadata-dir` | `./data/metadata/zenodo` | discovery's raw Zenodo record JSON |
| `--out-dir` | `./results/survey` | results, CMDS JSON, predictions, caches |
| `--scratch-dir` | `./data/survey_scratch` | per-record scratch under `<dir>/records/`, deleted after each record; must be new, empty or marked as a survey scratch dir |
| `--model` | `$ENVISION_SURVEY_MODEL` | ONNX model with its `.json` sidecar; required when the variable is not set |
| `--max-images` | 2000 | classification cap per record for local and downloaded files (seeded random sample) |
| `--remote-cap` | 300 | images sampled per record from zips read remotely, and top-level image files downloaded per record |
| `--max-download-gb` | 15 | per-record budget for non-zip downloads, filled smallest first; files that do not fit are listed, and `skipped_size` only when nothing is readable |
| `--no-remote-zip` | off | download missing zips whole instead of sampling them remotely |
| `--remote-nested-max` | 20 | nested archives fetched per record when a remote zip holds zips |
| `--remote-nested-gb` | 2 | bytes of nested archives fetched per record from remote zips; a budget of its own, separate from `--max-download-gb` |
| `--remote-record-budget-gb` | 2 | bytes fetched per record from remote zips, direct members and nested archives together; sampled members past it are not fetched |
| `--remote-max-member-mb` | 200 | remote zip members larger than this are never sampled; another member is drawn in their place where the zip has one |
| `--threshold` | 0.6 | top-1 probability below this is UNCERTAIN |
| `--disk-floor-gb` | 80 | skip a record (`skipped_disk`) if its download would leave less free space |
| `--download-workers` | 3 | parallel downloads and member fetches per record |
| `--rpm` (or `--zenodo-rpm`) | 120 | Zenodo requests per sliding minute for this process, all threads together (Zenodo allows 133 per 60 s window per IP and User-Agent); with `--shared-state-dir` it sets this process's share, e.g. 90 + 20 |
| `--shared-state-dir` | none | dir shared by processes running at the same time: a 429 in one pauses all of them, their requests together stay under `--shared-rpm`, and each subtracts the others' in-flight downloads on the same disk before checking the disk floor |
| `--shared-rpm` | 120 | Zenodo requests per sliding minute of all the processes sharing `--shared-state-dir` (or the pipeline's `--state-dir`) together; ignored without either |
| `--triage-cap` | 50 | two-pass sampling: images classified first from each source; only records whose triage sample holds eye classes get the deep sample (see Two-pass sampling); 0 turns it off |
| `--zenodo-interval` | 0.5 | minimum seconds between two Zenodo requests |
| `--download-all` | off | also fetch documents, tables, code and arrays |
| `--min-eye-fraction` | 0.05 | a record has eye images (status `ok`) only when the eye classes make up at least this fraction of its non-mask classified images (thresholded labels) |
| `--min-class-fraction` | 0.01 | in a record with eye images, fraction of the non-mask classified images for an eye class to be listed as present |
| `--ids`, `--ids-file` | all records | only these record ids; `--ids-file` (repeatable) reads them from a file, one per line, e.g. from `partition` |
| `--no-link-check` | off | skip HTTP HEAD checks of weblinks |
| `--retry-status` | none | reprocess records with these statuses, e.g. `error,skipped_disk,skipped_size,remote_listing_failed,ok_partial_download,images_unreadable` |
| `--no-archive-probe` | off | download non-zip archives whole without reading their listing first (see Archive probe) |
| `--archive-probe-mb` | 64 | bytes read at most per archive probe; a compressed tar is listed from its first this many MB |
| `--archive-probe-max-requests` | 50 | range requests at most per archive probe |
| `--series-sample-k` | 3 | top-level image files downloaded per homogeneous series of large files (same kind, extension and name with digits masked, each at least `--series-min-mb`); the rest are counted only; 0 downloads every file |
| `--series-min-mb` | 64 | file size from which `--series-sample-k` applies |
| `--token-file` | `~/.config/envision-survey/zenodo_token` when it exists | file holding a Zenodo access token (one line); `ZENODO_TOKEN` in the environment wins |
| `--metadata-cache-dir` | `<out-dir>/cache` | where record metadata fetched from Zenodo is cached (`legacy/`, `datacite/`); point several runs at one dir to fetch each record once |

**Zenodo token.** With `ZENODO_TOKEN` set, `--token-file`, or the default
token file `~/.config/envision-survey/zenodo_token` (read when it exists
and neither of the others is given), requests to zenodo.org carry
`Authorization: Bearer <token>`. An explicit `--token-file` that does not
exist stops the command. Measured on the VM, the
token does not raise the rate limit (the bucket is keyed by IP and User
Agent, 133 requests per 60 s window whatever the auth); it only opens
records the account was granted. The token goes to zenodo.org hosts only
(never to another host, never on a redirect elsewhere, never through the
weblink checker, which has its own session), and it is never logged,
printed or written to results, events, the workbook or CMDS files: the
run config in the events file holds the token file path, not the token.
As a second line of defence the token string is registered for redaction
when it is loaded: every results line, event line, predictions line, CMDS
file, status file and pipeline JSONL line, and every log record (a filter
on the log handlers, tracebacks included), has it replaced by
`[REDACTED]` should it ever turn up in a message or a file name. The
pipeline passes `--token-file` (a path) to its child processes, never the
token, so it does not appear in a process list either; only the `fetch`
role loads it (the `process` role makes no Zenodo API request). Keep the
token file outside the repo with mode 600, and never pass the token on a
command line.

## Full pull: `pipeline` (fetch, process, monitor)

`run` does everything in one process, one record at a time: a record that
downloads a 5 GB tar at 1 MB/s holds up the classifier for more than an
hour, and the classifier holds up the downloads. The full keyword pull
(about 30,500 records, several TB listed, about 120 Zenodo requests per
minute) is instead split over three processes that share a spool dir and
a state dir:

```
             Zenodo (one throttle, --rpm)
                      |
          +-----------v-----------+   deep / refetch requests   +---------------------+
          | fetch (producer)      |<----------------------------| process (consumer)  |
          | metadata, plan, probe |                             | walk, convert,      |
          | remote zip members,   |  spool/records/<id>/ READY  | classify (ONNX),    |
          | top-level images,     |---------------------------->| rows, CMDS, preds,  |
          | whole archives        |                             | weblinks; deletes   |
          +-----------------------+                             | the spool dir       |
                      |                                         +---------------------+
                      +----------> state dir <-------- monitor (status.json every --interval s)
```

- **fetch** plans every record of the scrape from its Zenodo record JSON,
  reads what can be read remotely (zip members through the container API
  or range reads, top-level image files one by one, archive listings by
  range reads: see Archive probe) and downloads whole only the archives
  that cannot be read remotely and whose probe allows it. Each record goes
  into `<spool>/records/<id>/` with a `manifest.json` (written and fsynced
  first) and then a `READY` marker. It runs `--fetch-workers` records at
  once (default 4), each with `--download-workers` streams (default 2 to
  3), all under the one request budget. Files of the discovery downloads
  dir are never copied into the spool: the manifest names them by path
  and the consumer reads them in place.
- **process** takes `READY` records oldest first, re-validates every
  spooled file (exists, same size as recorded; local originals exist),
  classifies them (format conversion, MASK, UNCERTAIN, two-pass sampling,
  below), appends the result row with flush and fsync, writes CMDS JSON,
  predictions and the weblink catalogue, and only then deletes the
  record's spool dir. It is the only process that writes
  `survey_results.jsonl` and it makes no Zenodo API request of its own (the
  metadata comes from the cache the producer filled; weblink checks of
  zenodo.org links, including each redirect hop and the GET that follows a
  refused HEAD, count against the same request budget through the state
  dir, and their rate-limit headers pause the producer too). On SIGTERM or
  SIGINT it finishes the current record and exits 0; the record's attempt
  count is restored at once, so a stop is never counted as a crash.
- **monitor** writes `<state>/status.json` every `--interval` seconds
  (default 60) and appends the same snapshot to `monitor.jsonl`: free disk,
  spool size and records by state, pending requests, results by status,
  rows finished in the last hour, ETA, and each role's liveness, state,
  counters, current records and the producer's wait reason.
- **pipeline** starts fetch (and waits until it holds its lock), then
  process and monitor, each with the same options and its own log
  `<state>/logs/<role>.log`. A fetch or process that exits with a
  non-zero code (a crash, an out-of-memory kill) is logged as
  `role_crashed` in `<state>/pipeline_events.jsonl` and in
  `<state>/pipeline_status.json` (which the monitor includes) and started
  again (`role_restarted`) after `--restart-backoff-s` (default 10,
  doubled per restart, at most 300 s), at most `--max-role-restarts`
  (default 20) times per role; then it is given up (`role_given_up`). A
  record that kills the consumer on every start is stopped by the attempts
  guard (status `crashed`), and the survey goes on with the next record.
  A process that ends cleanly (exit 0; it stops when fetch has been gone
  for a few polls) while fetch still runs is started again after
  `--restart-backoff-s`, and one that ended while fetch waits for its
  restart starts again with fetch. Such clean restarts are counted apart
  (`clean_restarts`) and neither use up `--max-role-restarts` nor grow the
  backoff, so a fetch that crashes now and then never gets process given
  up. The pipeline ends once fetch and process have both ended and no
  restart is pending, with exit code 1 when a role was given up or ended
  with a non-zero code.

### Two-pass sampling

Every sampled source (local and downloaded files; remote zip members and
top-level image files fetched one by one) is put in one seeded order. The
triage pass classifies the first `--triage-cap` (default 50) of each
source. Only when the eye classes make up at least `--min-eye-fraction` of
the triage sample's non-mask images (thresholded labels) does the deep
pass follow: the rest of the same orders, up to `--max-images` (2000) for
local and downloaded files and `--remote-cap` (300) for remote members.
The triage images are kept and part of the deep sample; nothing is
fetched twice.

- Remote members: the producer fetches only the triage members first. The
  seeded order does not depend on the cap (each zip's candidates are
  shuffled once, and a zip contributes the first members of its order;
  the per-zip quota of the larger cap is never below the triage one), so
  the triage sample is a prefix of the deep sample. When the consumer needs
  the deep pass it marks the record `AWAITING_DEEP`, appends an
  `awaiting_deep` line to the results, and writes a deep request; the
  producer fetches the missing members into `<record>/deep/` (reusing the
  saved zip listings, so no listing request is repeated), and the
  consumer runs the record again with both passes. The triage decision is
  taken again from the triage-pass images only, so it is the same.
- Local and downloaded files (archives downloaded whole are downloaded
  once, on the triage pass): the consumer takes the deep sample at once.
- The row says which pass the record got: `sampling_pass` (single, triage
  or deep), `n_triage_classified`, `triage_eye_fraction` and
  `deep_pass_reason`; `n_remote_reused` counts the deep sample's members
  taken over from the triage pass.
- `--triage-cap 0` turns the two passes off (one pass, as before). `run`
  applies the same rule within its one process.

### Disk safety

- The spool dir must be new, empty or marked by the survey
  (`.envision_survey_spool`), and it may not be, contain or sit inside the
  downloads dir or the output dir. Deletions happen only for record dirs
  directly under `<spool>/records/` named by a record id, never through a
  symlink. `data/downloads` and earlier results dirs are never written.
- One spool, one state dir: the fetch and process roles each lock
  `<spool>/.lock_fetch` and `<spool>/.lock_process` as well as their lock
  in the state dir, and the spool records the state dir that owns it
  (`<spool>/.state_dir`). A second run given the same spool with another
  state dir refuses to start, since the producer deletes half-written
  record dirs and the consumer deletes finished ones.
- High-water mark: the producer starts no record while the spool holds
  more than `--spool-max-gb` (default 300) or while the drive would keep
  less than `--disk-floor-gb` (default 80) free, counting the bytes the
  records in flight reserved; it waits (state `waiting_for_room` in its
  status) until the consumer frees space. A record larger than the spool
  budget starts once nothing else is spooled or in flight. Deep passes of
  records already in the spool are held back only by the floor (records
  waiting for their deep pass could otherwise fill the spool and wait on
  each other). A new record waiting for room gives its fetch slot to a
  pending deep or refetch request (event `triage_yielded`; the record goes
  back to the front of the queue), so a spool full of records awaiting
  their deep pass is always served and freed. The room check and the
  reservation are one step, so parallel fetch workers never count the
  same free space twice.
- Downloads also stop when the floor is crossed while they stream, and the
  consumer's own extractions (nested archives, split sets) check the
  floor minus the producer's published reservations.
- When the consumer has been gone for `--consumer-grace-s` (default 600)
  while the producer has to wait for it, the producer stops (exit code 3)
  instead of waiting for ever; the record it was about to fetch is not
  written.

### Resuming

Kill any role (or the whole pipeline) and start the same command again:

- A record is done only when its final row is in `survey_results.jsonl`
  (`started` and `awaiting_deep` lines do not count). At start the
  producer leaves out done records and records a previous run left
  `READY`, `AWAITING_DEEP` or `DEEP_READY` in the spool (the consumer and
  the request files own those, so a record the consumer finishes is never
  fetched a second time), and it skips a queued record that got a final
  row since it started. Record dirs without `READY` that a killed or
  failed fetch job left are deleted at start (event
  `partial_spool_removed`), whatever the new run's `--ids`, `--limit` or
  `--retry-status`, and the records still to do are fetched again.
- Pending deep and refetch requests (`<spool>/requests/`) survive a
  restart and are served before new records. A record left
  `AWAITING_DEEP` without a request (a role killed between the two steps)
  gets its deep request again (event `deep_request_restored`), at start
  and every 30 s. The producer removes a served request before it
  publishes the record, so a new request the consumer writes right after
  is kept, and it ends only after two checks, one poll apart, find no
  spooled record and no request.
- The consumer counts its starts of each record (`.attempts` in the
  record dir). A record it started `--max-attempts` (default 2) times
  without finishing (the process died inside it) gets status `crashed`
  (rerun with `--retry-status crashed`).
- A spooled file that is missing or changed size is not processed: the
  consumer removes the record's markers (the dir is then half written for
  the producer) and requests a refetch; the producer deletes the dir and
  fetches the record again. The consumer marks such a dir `RETURNED`
  first, so a kill between the two steps leaves a returned dir without a
  request, whose request the producer restores (event
  `refetch_request_restored`). After two refetches the record gets
  an `error` row, written before its dir is deleted.
- A record whose final row was written after its `READY` (or
  `DEEP_READY`) marker (the consumer died between the row and deleting
  the dir) is not classified again: its dir is deleted (event
  `spool_already_done`).
- A triage job that gives its slot to a request keeps its archive probe
  result, so the retry does not probe the same archives again.
- The consumer registers the token for redaction too (it never sends
  it), so everything it writes (rows, events, CMDS, predictions, the
  workbook) is redacted even when a producer's error text carried the
  token. Spool files are scratch and keep file names as they are (a
  redacted path could not be validated or read).
- `--retry-status` works as in `run`: the producer fetches those records
  again and the consumer replaces their rows.
- Adding records to a running survey (for example a gap-fill scrape):
  write one scrape file that holds the old records and the new ones (no
  id twice), stop the pipeline, point `--scrape` at the new file and start
  it again with the same out, state and spool dirs. Finished records are
  skipped, so only the new ones and the unfinished ones are fetched. They
  share the one producer and its Zenodo rate limit, and their rows go into
  the same results file and workbook. A second pipeline on the same
  machine would compete for the same cores and needs its own out, state,
  spool and scratch dirs, so a single combined scrape is simpler.

### Keeping eye-positive records (`--keep-dir`)

By default the consumer deletes every record's spool dir once its row is
written. With `--keep-dir DIR` a record whose row holds at least one image
classified as an eye class (thresholded label, MASK excluded: `n_CFP`,
`n_IR`, `n_PSC`, `n_FAF`, `n_OCT`, `n_OCTA`; the row's `n_eye_images`)
has its fetched files moved into `DIR/<record id>/` instead; every other
record is deleted as before. This holds at every pass: a record whose
triage sample has no eye image is deleted after triage; a record that
goes on to its deep pass keeps its triage files in the spool meanwhile,
and at its final row the triage and the deep files are kept together.

What is kept, relative paths preserved: `dl/` (files downloaded whole:
archives, top-level images), `remote/` (zip members and nested archives
read remotely), `deep/...` (the same for the deep pass), and
`manifest.json`, `manifest_deep.json`, `listings.json` (which Zenodo file
each path is). Left out: the spool markers, temporary files and the
scratch dirs `work/` and `.fetchwalk/`. Extractions (nested archive
members, joined split sets) are copies of what the kept archives hold, so
keeping the archive is enough; the predictions file names each classified
member inside it. Local originals (the discovery downloads) are never
moved or copied: `DIR/<id>/KEPT.json` lists their paths (and the row the
first 20, `kept_local_paths`).

Moves are renames, one per file, never copies: the keep dir must be on the
spool's filesystem (checked at start) and, like the spool, new, empty or
marked by the survey (`.envision_survey_keep`), not inside or around the
spool, downloads, output, scratch or state dir. The survey never deletes
anything in it.
Order: the row (with `kept` true) is written and fsynced, each file is
renamed, `KEPT.json` is written, the spool dir is deleted. A kill anywhere
in between leaves the spool dir with its `READY` marker; on restart the
consumer finds the row newer than the marker with `kept` true, moves what
is left and writes `KEPT.json` again, without classifying the record a
second time. A file is at every moment in the spool or in the keep dir,
never in both, and a second move onto the same path replaces it, so
nothing is lost or duplicated.

A move that fails (an OS error) is tried again after 2 s and 4 s. After a
third failure the record is parked: its spool dir gets a `KEEP_FAILED`
marker (spool state `keep_failed`, shown in the monitor's spool counts,
event `keep_move_gave_up`) and stays with every file that did not move.
Neither role touches a parked record and nothing deletes it; the next
process role started with `--keep-dir` removes the markers and finishes
the moves. A spool dir is deleted only after its move is complete. A
consumer started without `--keep-dir` parks (never deletes) a record whose
row says `kept` while some of its files are still in the spool.

Guards. Kept files are never freed, so keeping is bounded: `--keep-max-gb`
(default 200; 0 for no cap) caps the keep dir, and a record is kept only
while the free space, counted as if the spool were empty, stays at least
`--disk-floor-gb` + `--spool-max-gb`. The producer's disk floor keeps
working as before (it pauses at the floor), and because of the second
guard the spool always keeps its full budget above the floor, so keeping
never turns records into `skipped_disk` rows. A record refused by a guard
gets `kept` false with `kept_reason`, event `keep_refused`, and its files
are deleted; records already kept stay.

Row columns (also in the workbook): `n_eye_images`, `kept`, `kept_reason`
(`no eye image` or the guard), `kept_path`, `kept_files` and `kept_bytes`
(data files, manifests not counted), `kept_local_files`,
`kept_local_paths`. The consumer writes `<state>/keep_status.json` (records
and bytes in the keep dir); the monitor adds a `keep` block (dir, GB,
records, cap, kept and refused this run) and `disk.keep_gb` to
`status.json` and its log line.

Backfill (records finished before `--keep-dir` was given, whose files were
deleted): `keep-backfill-ids` lists the records whose last final row has
no `kept` field and holds an eye image, by the row's counts or by its
predictions file, leaving out ids that already have a `KEPT.json` and
records that fetched nothing (`spool_bytes` 0: local originals only, read
in place, so nothing was deleted; listed under
`eye_local_only_not_listed` in the summary):

```bash
envision-survey keep-backfill-ids --out-dir $OUT --keep-dir $KEEP --output $STATE/backfill_keep_ids.txt
```

Then start the pipeline with `--refetch-keep-ids $STATE/backfill_keep_ids.txt`
(and `--keep-dir`): the producer fetches those records first and the
consumer classifies them again (same seed, same sample) and keeps their
files. The new row replaces the old one (the last line of a record wins,
in the producer, the monitor and the workbook). Only ids whose last final
row has no `kept` field are redone, so the option can stay on across
restarts: a record already backfilled is never fetched again.

### Operating guide

From the envision-discovery checkout root, with the 30,501-record scrape
copied next to the old results (do not overwrite the older scrape):

```bash
MODEL=/path/outside/repo/regnety004_synthonly_distill_s2.onnx
OUT=results/survey_pull; STATE=results/survey_pull_state; SPOOL=/big/disk/survey_spool
mkdir -p $OUT $STATE
# optional pass 0: record and DataCite JSON for every record (about 2 requests each,
# resumable; the producer otherwise fetches them record by record)
envision-survey metadata --scrape results/zenodo_full.json --metadata-cache-dir data/metadata/zenodo_full \
    --state-dir $STATE
setsid nohup envision-survey pipeline --model $MODEL --scrape results/zenodo_full.json \
    --metadata-cache-dir data/metadata/zenodo_full --out-dir $OUT --state-dir $STATE --spool-dir $SPOOL \
    > $STATE/pipeline.log 2>&1 < /dev/null &
cat $STATE/status.json                        # or: envision-survey monitor --state-dir $STATE --once ...
tail -f $STATE/logs/fetch.log $STATE/logs/process.log
envision-survey excel --model $MODEL --results-dir $OUT --out $OUT/zenodo_modality_survey.xlsx
```

- The token is read from `~/.config/envision-survey/zenodo_token` (mode
  600) when that file exists; nothing token related goes on the command
  line.
- Stop: `kill -TERM <pipeline pid>` forwards the signal to the roles;
  killing the roles one by one is safe too. Start the same command again
  to resume. On Linux the roles also get SIGTERM when the pipeline process
  itself dies without forwarding it (SIGKILL, an OOM kill): process
  finishes its current record, then exits, so no role keeps running
  without a supervisor. A new start refuses while a role of the same
  state dir is still alive; wait for it to end.
- The roles can also run on their own (`fetch`, `process`, `monitor` with
  the same options): one process per role and state dir (a lock in
  `<state>/locks/`, which also tells the others whether a role is alive).
  A consumer without a producer drains the spool and stops; a producer
  without a consumer fetches until the spool is full.
- `excel` can run at any time; rows still in progress show as `started`
  or `awaiting_deep`.

Order of work: `--order cost` (default) fetches records with little to
download first (no pixel files, then remote zips at a nominal 50 MB each,
then whole-file downloads by size, from the cached record JSON when there
is one, else from the scrape's file names and size), so the catalogue
fills early and the large archives come last; `--order scrape` keeps the
scrape order.

Throughput (measured from the VM): Zenodo allows 133 requests per 60 s
window per IP and User-Agent whatever the token; the survey keeps to 120.
A single download stream runs at about 1 to 2 MB/s and 8 streams at about
17 MB/s together, so four fetch workers with two or three streams each
fill most of the link. Expect about 30,000 metadata requests, about
120,000 triage requests and a deep pass for the records with eye images:
two to three days end to end at 120 requests per minute, with the
consumer (two cores) rarely the bottleneck.

Files in the state dir:

| File | Written by | Content |
|---|---|---|
| `status.json`, `monitor.jsonl` | monitor | the latest snapshot; one line per snapshot |
| `fetch_status.json`, `process_status.json` | each role, every 15 s | state, counters, current records, heartbeat |
| `fetch_events.jsonl` | fetch | fetched, fetched_deep, archive_probe, skipped_disk, fetch_job_error, ... |
| `pipeline_status.json`, `pipeline_events.jsonl` | pipeline | each role's pid, start, end, exit code and restarts; role_crashed, role_restarted, role_given_up |
| `logs/<role>.log` | pipeline | each role's output |
| `locks/<role>.lock` | each role | one process per role; liveness |
| `zenodo_requests`, `zenodo_not_before`, `disk_reserve.*` | fetch (and process) | shared request budget, 429 cooldown, in-flight disk reservations |
| `keep_status.json` | process (with `--keep-dir`) | records and bytes in the keep dir |

Pipeline options (besides the `run` options above, which all roles
accept):

| Option | Default | Meaning |
|---|---|---|
| `--spool-dir` | required | where the producer writes records for the consumer |
| `--state-dir` | required | status, locks, logs, events, request budget and 429 cooldown |
| `--spool-max-gb` | 300 | the producer starts no record while the spool holds more |
| `--fetch-workers` | 4 | records fetched at once |
| `--triage-cap` | 50 | two-pass sampling (0: one pass) |
| `--order` | cost | cost or scrape |
| `--max-attempts` | 2 | consumer starts of a record before it is `crashed` |
| `--consumer-grace-s` | 600 | the producer stops when the consumer has been gone this long and it has to wait |
| `--max-role-restarts` | 20 | pipeline: restarts of fetch or process (each) before that role is given up |
| `--restart-backoff-s` | 10 | pipeline: first restart delay, doubled per restart, at most 300 s |
| `--poll-s` | 10 | polling interval of the roles |
| `--interval` | 60 | monitor snapshot interval |
| `--once`, `--exit-when-idle` | off | monitor: one snapshot; or stop when neither role runs |
| `--keep-dir` | none | process: move the fetched files of records with an eye image here instead of deleting them |
| `--keep-max-gb` | 200 | process: keep no new record past this size of the keep dir (0: no cap) |
| `--refetch-keep-ids` | none | fetch: redo these pre-retention records first so their files reach the keep dir |

## How a record is processed

1. **Metadata.** The legacy record JSON comes from `--metadata-dir` when it
   is there, otherwise from the Zenodo API. The DataCite JSON comes from
   `GET /api/records/<id>` with `Accept: application/vnd.datacite.datacite+json`.
   Both are cached under `<out-dir>/cache/` (or `--metadata-cache-dir`). The
   legacy JSON lists every file of the record (checked on a record with
   1778 files); the scrape's `file_names` stops at 20 and its
   `zip_file_types` is empty, so the survey never plans from the scrape's
   file lists. Every Zenodo request (metadata
   calls, downloads, zip listings, member fetches and range reads, across
   all worker threads) goes through one shared throttle: at most 120
   requests in any sliding 60 s window (`--rpm`) and at least 0.5 s
   between two requests (`--zenodo-interval`). A 429 sets a shared cooldown
   from Retry-After (seconds or HTTP-date; at least 60 s when the header is
   missing) that every worker waits on, and with `--shared-state-dir` every
   other process sharing that dir as well; with that dir the processes
   together also stay under `--shared-rpm` (120) per sliding minute. 5xx
   responses and transport errors
   back off exponentially per request (2 s doubling up to 120 s). Zenodo's
   own `X-RateLimit-Remaining` is read on every response: at 3 or fewer
   requests left, every request pauses until `X-RateLimit-Reset` (epoch
   seconds), whatever the local count says.
2. **Files.** Downloading everything is not an option: Zenodo serves the
   VM at 2 to 7.5 MB/s, so the 1.75 TB of records not yet on disk would take
   3 to 4 days. Each file is handled one of three ways, and the record's
   `sampling_mode` lists the ones used (`local`, `remote_zip`, `download`,
   joined with `+` when mixed; `none` when nothing was read):
   - **local**: a file with a complete copy under `downloads_dir/<id>/` is
     used in place.
   - **remote_zip**: a missing `.zip` (not a spanned `.zip` + `.z01` set) is
     never downloaded. Its members are listed with Zenodo's container API
     (`GET /api/records/<id>/files/<name>.zip/container`). That listing is
     capped near 1000 items (files plus directories), has no pagination
     (`page`, `size`, `limit`, `offset` and `from` are ignored) and reports
     `truncated: true` when the cap is hit. Because the cap and the flag are
     empirical, a listing is also treated as truncated when its `total` is
     above the items returned or when it holds 1000 items or more. A
     truncated or failed listing
     falls back to HTTP range reads of the zip's central directory: Python's
     `zipfile` runs over a seekable file object whose reads become
     `Range` requests with a small block cache. A 7.8 GB zip with 9035
     members was listed with four range requests after the capped
     container call. macOS junk (`__MACOSX/`, `._*`, `.DS_Store`) is
     dropped. Image members are picked by extension, and a
     seeded sample of at most `--remote-cap` (300) images is spread over the
     record's zips in proportion to their image counts (largest remainder).
     Only those members are fetched into the scratch dir, one request each
     through `.../container/<member path>` (size and CRC checked). Each one
     is saved under a flat, unique scratch name (index, name hash and
     suffix; the member path is kept only for display), so a long
     directory name, a file named like a directory or two names that
     differ only in `:` cannot collide or abort the record; a member that
     still fails is counted in `remote_fetch_failures`. When the
     endpoint refuses a member, it is read with a range request of its own
     bytes instead. When the container listing itself failed for a zip,
     its members go straight to range reads rather than paying a full
     retry cycle on the container endpoint per member. A member of 16 MB
     or more that is at least twice as large inflated as compressed is
     read by range too, since the container endpoint sends it inflated. A
     stored, deflated or Deflate64 (method 9, through `inflate64`) member
     is read with one streamed range request and inflated on the fly into
     the scratch file, so memory stays flat for any member size; other
     compression methods go through `zipfile` in block-sized reads. A zip
     over 4 GiB written without ZIP64 records stores 32-bit offsets that
     wrapped, and `zipfile` then shifts every member offset by a multiple of
     4 GiB (record 4005629 lost 79 of 280 fetches to this): a member whose
     local header is not at the listed offset is looked up at the offsets
     4 GiB apart, accepted only when the signature and file name match. A
     volume header member (`x.mhd`, `x.hdr`, `x.nhdr`) is fetched together
     with its data member (`x.raw`, `x.zraw`, `x.img`) into a directory of
     their own, under their own names.
     A range body that breaks off mid-read or comes back short is retried
     with backoff. A zip whose listing fails both ways counts in
     `remote_listing_failures` (see Statuses). A zip whose images sit in
     nested archives (a zip of per-set zips) gets its nested archives
     fetched whole, in seeded random order, until `--remote-cap` images are
     found, `--remote-nested-max` archives were opened, `--remote-nested-gb`
     (2 GB, a budget of its own) is used up, or 5 archives in a row held no
     image; they are then walked like local archives. Nested archives over
     500 MB are not fetched (`n_remote_nested_over_cap`), and archives that
     no longer fit the budget are counted in
     `n_remote_nested_over_budget`. At 2 to 7.5 MB/s, the 2 GB budget bounds
     the nested fetch of one record to about 5 to 17 minutes;
     `remote_nested_bytes` and `remote_nested_stop` record what was used
     and why it stopped, and each record with nested fetches logs a
     `remote_nested` event. A zip holding only extensionless files
     gets a sample of those, sniffed for DICOM. The container API answers
     HTTP 500 for RAR, so it is used for zip only.

     Two byte bounds keep one record from pulling gigabytes. Members whose
     transfer is over `--remote-max-member-mb` (200 MB) are never sampled.
     The transfer is the compressed size (the bytes a range read moves),
     not the inflated size: a 794 MB confocal stack stored in a 188 MB zip
     (record 4943680) is sampled, and its decode is bounded separately (a
     reduced read, see Format conversion; a member may inflate to at most
     4 GiB on disk). Oversize members are left out
     of the candidates, so the zip's quota goes to its other members
     (`n_members_skipped_oversize` counts them). They are outside the
     sampling frame, so they are left out of the population as well: the
     class fractions, the per-class file estimates and `n_image_files`
     describe the members up to the cap only, since large members can hold
     other images than the small ones (full scans next to masks or
     thumbnails). They are reported unclassified as `n_images_unsampleable`,
     and `fraction_scope_note` says so on the row. Empty members (size 0,
     `n_members_zero_size`) are left out the same way; they are not counted
     in `n_images_unsampleable`. `--remote-record-budget-gb` (2 GB) caps
     the bytes received per record from remote zips, nested archives and
     direct members together, counted as they arrive, so bytes of failed
     transfers and retries count too (a body that breaks off or fails its
     CRC check, the range read that follows, a failed nested archive). A
     nested archive that no longer fits is skipped
     (`n_remote_nested_over_budget`, stop reason "record byte budget used
     up"); the sampled direct members are taken in seeded random order
     until the next one would pass the budget (sizes from the listing, so
     this cut happens before any request); during the fetches a member is
     not started when it no longer fits the bytes left, and a transfer that
     would pass the budget is stopped. So a record receives at most the
     budget plus one 1 MB read chunk per fetch worker. Members not fetched
     for the budget (`n_remote_over_budget`, `remote_budget_stop`) count as
     listed, unfetched images, so the population estimate is unchanged and
     the classified sample is a random subsample. `remote_bytes_fetched`
     records the bytes received for members and nested archives, failed
     transfers included (the listing requests are not counted), and
     `remote_bytes_fetched_ok` the listed sizes of the fetches that worked.
     For records without a remote zip, `n_members_skipped_oversize`,
     `n_images_unsampleable` and `remote_bytes_fetched` are 0, and
     `n_remote_over_budget`, `remote_budget_stop` and
     `remote_bytes_fetched_ok` are left out (blank in the workbook).
   - **download**: other missing files that can hold images are streamed
     whole into `scratch_dir/records/<id>/dl/`: rar, 7z, tar (zstd
     included) and gz archives and their split parts, every pixel-bearing
     kind the survey converts (images, SVG, DICOM, volumes and volume
     headers with their data file, vendor OCT, video, numeric arrays,
     microscopy), extensionless files, and single-file
     `.gz`/`.bz2`/`.xz`/`.zst` files whose inner name is one of those or an
     archive (`x.tif.gz`, but not `x.csv.gz`).
     Top-level image files beyond `--remote-cap` are sampled (seeded); the
     others are counted and their names used for laterality and hints.
     Large top-level files of one homogeneous series (same kind, extension
     and name with digit runs masked, each at least `--series-min-mb`,
     more than `--series-sample-k` of them) are sampled the same way, 3 per
     series by default: record 15116835 has 30 land-cover GeoTIFFs of
     913 MB each, and three say what the others are (the old run fetched
     16 of them, 14.6 GB, and read none). `n_series_files_not_fetched`
     counts the rest.
     Each record may download up to `--max-download-gb` (15 GB). When its
     files need more, the budget is filled smallest first (the parts of a
     split archive count as one unit, since a partial set cannot be read),
     so one huge RAR does not also drop the small TIFFs next to it. The
     files that did not fit are listed with key, size and link
     (`skipped_size_files`) and `size_note` says so. A record where nothing
     at all is readable gets status `skipped_size`: the file list is kept in
     the row, the weblinks are catalogued and a metadata-only
     `dataset_description.json` is written. The download resumes with HTTP Range, and
     its size and md5 are checked. Truncated or oversize bodies are retried
     with exponential backoff, and an md5 mismatch restarts the file at most
     twice. Before any download, the record's download size is checked
     against the disk floor (free space minus the other survey processes'
     reservations with `--shared-state-dir`), and while a file streams the
     floor is checked again every 64 MB: a file that would cross it is
     stopped and deleted ("disk floor reached"), since another process may
     have used the space since the first check.

   Documents, code, tables and genomics files are not fetched unless
   `--download-all` is set, because nothing reads them.

   **Archive probe.** Before a non-zip archive is downloaded whole, its
   listing is read with a few HTTP range requests (`probe.py`), and the
   archive is downloaded only when the listing shows a member the walker
   would use: a pixel-bearing file, a nested archive or a compressed image,
   or (in tar) an extensionless member that sniffs as DICOM or an image.
   An uncompressed `.tar` is listed by hopping from header to header; a
   compressed tar (`.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`, `.tar.zst`)
   by streaming its first `--archive-probe-mb` (64 MB) and listing on the
   fly (a compressed tar has no index, so a listing cut off by that limit
   without an image member is undecided); `.7z` by reading its end header
   (encoded headers included); `.rar` (4 and 5) by walking the block
   headers and stopping at the first member that counts (`rarfile` takes
   over for layouts the walker does not parse); a single-file `x.gz` by its inner name (an extensionless inner
   name gets its first bytes decompressed and sniffed). Each probe stops at
   `--archive-probe-mb` bytes and `--archive-probe-max-requests` (50)
   requests, and a record at 1 GiB and 400 requests of probing. Outcomes:
   `images` (downloaded as before), `no_images` (the whole listing was
   read and holds nothing usable: not downloaded, its member names and
   kinds are catalogued, and a record with nothing else to read gets
   status `no_images_in_archives`), `unknown` (limit reached, encrypted
   header, error: downloaded as before, so an image-bearing archive is
   never skipped) and `not_probed` (archives of 1 MB or less, split sets:
   one part never proves a set empty). The row keeps each archive's
   outcome, method, member counts, bytes and requests (`archive_probes`,
   the Archive_Probes sheet) and `archive_probe_bytes_avoided`.
   `--no-archive-probe` turns it off.

   **Split sets.** Numbered and `split`-style sets (`x.tar.001`,
   `x.zip.partaa`, `x.part01`) are fetched as one unit and joined into one
   file in the scratch dir before the walk (`split_join_detail`); rar,
   spanned zip and `x.7z.001` sets are read by their tools as before.
3. **Enumerate.** Archives are listed, not unpacked:
   - zip is read through its central directory;
   - tar goes through one streaming pass;
   - 7z, rar and spanned zip (`.zip` + `.z01`) use `7z l`. RAR members
     are extracted with `unrar`, because many 7z builds list RAR but cannot
     decode it.

   Nested archives and single-file `.gz`/`.bz2`/`.xz`/`.zst` members are extracted
   one at a time into the scratch dir and walked, up to depth 3. An
   extracted archive with no image members, and a decompressed file that
   is not an image, is deleted right after it is walked, so per-patient
   inner zips cannot fill the disk during listing.
   Extensionless members are sniffed by their first bytes: DICOM with the
   `DICM` preamble or without it (a group 0002 or 0008 element at offset
   0, as in ACR-NEMA style files), PNG, JPEG, TIFF and BMP. A volume header
   is paired with its data member from the same directory, and both are
   extracted together when it is sampled. Local zips get the same 4 GiB
   offset lookup as remote ones, and Deflate64 members are read through
   `zipfile-deflate64` (or 7z). `member_ext_counts` counts every archive
   member listed by extension (local, remote and probed archives), so a
   record whose zips hold `.mat` or `.nd2` files shows it.
   Password-protected archives never stop the run: `7z` gets a dummy
   `-pnone` password and `unrar` gets `-p-`, with stdin closed, so they fail
   at once instead of waiting for a password prompt. An archive with an
   encrypted header becomes a walk error, encrypted members are skipped, and
   both are counted under `encrypted` in the file kinds.
4. **Sample and classify.** At most `--max-images` (2000) image files per
   record are drawn at random; a record read only through remote zip
   sampling uses `--remote-cap` (300) instead, which also bounds the images
   found in fetched nested archives. The RNG is seeded with
   `<seed>:<record id>`, so the sample does not depend on run order.
   A record's images can come from up to three sampling strata with
   different rates: `files` (local or downloaded files, all images up to
   2000), `remote` (direct image members of remote zips, at most 300 of
   those listed) and `remote_nested` (every image inside the nested
   archives fetched from remote zips, which are a sample of the listed
   nested archives; population scaled up by listed / fetched archives).
   When the classified images come from more than one stratum, class
   fractions are weighted:
   each classified image counts with weight (images in its stratum) /
   (images classified from that stratum), and the `frac_*`,
   `argmax_frac_*`, mean confidence, mean probabilities, dominant classes,
   present modalities and per-class file estimates come from the weighted
   counts. The `n_*` and `argmax n *` columns stay raw sample counts;
   `class_weighting` says `stratified` or `uniform` and
   `n_classified_by_source` gives the classified images per stratum. Ties
   between classes always go to the class listed first (CFP, IR, PSC, FAF,
   OCT, OCTA, NEG, UNCERTAIN), in the run and in the workbook backfill.
   `n_images_listed` counts every image seen in the listings;
   `n_image_files` is the population the per-modality file estimates are
   scaled to (it adds, for unopened nested archives, the mean image count
   of the opened ones, and for extensionless members, their DICOM share in
   the fetched sample). Only those members are read: zip
   and tar members are read directly, and 7z members are extracted to a
   temp dir in batches of at most 400 members and at most min(20 GB, free
   space minus the disk floor); members that do not fit are reported as
   `skipped (disk floor)`.

   **Format conversion.** Every pixel-bearing format is turned into one
   8-bit RGB frame (`images.py`); size never refuses a file, a large one
   is read at a reduced resolution instead:
   - TIFF of every kind (BigTIFF, tiled, pyramids, OME-TIFF, ImageJ and
     LSM stacks, SVS and NDPI slides; 1 to 64 bit, float, planar) goes
     through `tifffile` (codecs from `imagecodecs`). A pyramid gives its
     smallest level whose long side is at least 1024 px; a stack of
     same-shape pages gives its middle Z (or time) plane and first channel,
     not page 0, which in an OCT B-scan stack is an edge scan and in a
     confocal stack is often blank (`frame_index`). A page over the pixel
     budget (80 MP) or the decode budget (1 GiB) is decoded one strip or
     tile at a time, keeping every s-th row and column (s chosen to land
     under 16 MP), so memory holds one segment plus the output: the
     228 MP float32 GeoTIFFs of 15116835 and the 96523 x 153811 BigTIFF of
     4571628 are read, where the old run refused them (`too_large`).
   - JPEG is decoded at a reduced DCT scale when large, JPEG 2000 at a
     reduced resolution level; PNG, BMP, GIF, WebP, PSD (the composite),
     HEIC and AVIF (`pillow-heif`) go through Pillow.
   - SVG: the largest embedded raster image (a figure wrapping a photo),
     else the drawing rendered 512 px wide on white with `resvg`. Every
     reference that is not an in-document fragment or an embedded data URI
     is removed first and resources resolve in an empty directory, so
     rendering reads nothing outside the file and fetches nothing. Memory
     is bounded: a file over 64 MB is refused before it is read, an `.svgz`
     is decompressed only up to 64 MB (a gzip bomb never expands whole),
     and an embedded image over the pixel budget is not decoded (the file
     is then `too_large` unless a smaller embedded image can be used).
   - DICOM: the middle frame of a multi-frame file, with the compressed
     transfer syntaxes common in ophthalmology decoded (JPEG lossless,
     JPEG-LS and JPEG 2000 through `pylibjpeg`, RLE natively).
   - NIfTI, NRRD, MHA and header + data pairs (MHD + RAW through
     SimpleITK, Analyze HDR + IMG through nibabel, detached NRRD): the
     middle slice along the shortest axis. The uncompressed voxel size is
     computed from the header first, and volumes over 2 GiB uncompressed
     are not loaded (a small gzip NRRD can expand to many GB). NIfTI is
     sliced through the nibabel proxy and MHA/NRRD through a SimpleITK
     extract region, so only the slice is materialised.
   - Numeric arrays (`.npy`, `.npz`, HDF5 incl. Imaris `.ims`, MAT v7.3
     through h5py, MAT v5 through scipy): among the numeric datasets, the
     image-shaped ones (after dropping size-1 axes and a last axis of 3 or
     4 channels, a 2D plane, or the middle slice of the shortest of the
     last three axes, with both sides at least 64 px and an aspect ratio of
     at most 32, so signals and tables are not images); a name such as
     image, oct, bscan, vol, fundus or frame wins over a larger plane. Only
     the chosen slice is read (h5py and memory-mapped NumPy slice lazily,
     strided when large). Network parameters are never candidates: a
     dataset named like a layer parameter (kernel, bias, weight, gamma,
     beta, running_mean, moving_variance, embeddings, with an optional
     `:0`) is skipped, and an HDF5 that Keras wrote (root attribute
     keras_version, model_config or training_config, or a group
     model_weights) is not searched at all. A file without such a dataset
     (model weights, tables) is reported as `not_image_shaped`, which does not count as a
     pixel file for the record status.
   - Microscopy: CZI (`pylibCZIrw`, read at a zoom under the budget), LIF
     (`readlif`), ND2 (`nd2`, lazily), OIB (`oiffile`), MRC (`mrcfile`,
     memory-mapped): middle Z plane, first channel, time 0.
   - Vendor OCT: Heidelberg E2E, Topcon FDS and FDA (checked for the FOCT
     signature, since `.fds` is also the Fire Dynamics Simulator format)
     and Bioptigen OCT through `oct-converter`; Heidelberg VOL (HSF-OCT)
     and Thorlabs OCT (a zip of Header.xml and `data\*.data`) through the
     survey's own readers. The middle B-scan is classified, and the fundus
     or SLO image (Thorlabs: the camera image) is classified as a
     prediction of its own (path ending `#fundus`). A Thorlabs file with
     only raw spectra gets a B-scan from 1024 A-scans of its middle
     spectral file (mean spectrum removed, Hann window, FFT, log
     magnitude; no chirp correction). Heidelberg SDB has no open reader and
     is catalogued. `oct-converter` imports OpenCV; the survey extra
     installs the headless build, which needs no libGL.
   - Video: frames at 25, 50 and 75% of the duration (the ffmpeg bundled
     with `imageio-ffmpeg`), classified together: their probabilities are
     averaged into one prediction (`n_frames_averaged`).

   Integer data deeper than 8 bits and floats are windowed between the 0.5
   and 99.5 percentiles (on at most 1M samples), except data with 256 or
   fewer distinct values (label maps keep exact levels, min-max); float
   nodata (below -1e30 or non-finite) is left out of the window. An alpha
   channel (RGBA, LA, P or L with a transparent color) is composited over
   white, so a transparent figure no longer turns black. A constant frame
   is `blank`: it is not classified (`n_blank`). Each prediction line
   carries the source format and the conversion path, and the row
   summarises them (`formats_classified`, `conversion_counts`,
   `formats_unread`). A file that gives no image is written to the
   predictions file with its reason and header facts (rows, columns,
   format, compression), and counted in `n_pixel_files_unread_by_reason`
   (`decode_error`, `codec`, `too_large`, `companion_missing`,
   `no_reader`, `not_image_shaped`, `wrong_format`, `fetch_failed`,
   `blank`, `not_fetched`, `not_sampled`, `not_sampled_like_sample`, `oversize_member`,
   `over_record_budget`, `over_download_budget`, ...).

   Preprocessing rebuilds the training input exactly. First comes a
   content crop (the bounding box of pixels brighter than
   max(12, 0.08 x max gray), skipped if under 2% of pixels are foreground).
   Then the image is squashed to 256 x 256 without keeping its aspect ratio,
   as the tier-3 training PNGs were stored, resized to 224 x 224 (what the
   training Resize(224) + CenterCrop(224) does to a square), and ImageNet
   normalisation is applied. Nothing is cut off: an earlier version kept
   the aspect ratio and center-cropped, which dropped the sides of wide
   images (a third of a 768 x 496 OCT B-scan) and changed 15 of 160 labels
   on real non-square images. Inference runs on onnxruntime CPU in batches
   of 16.

   **Masks.** Segmentation masks and label maps are common next to eye
   images (vessel maps, lesion masks), and the classifier, which has no
   mask class, tends to call them OCTA. A pixel check runs on every decoded
   frame: a 128 x 128 nearest-neighbour sample (no interpolation, so a mask
   keeps its exact values) is a mask when it is gray (channels within
   +-2) with at most 4 gray levels, or in color with at most 8 distinct
   colors. Anti-aliased masks (a binary mask whose edges add 16 or so
   levels) and label maps are caught by a dominance test on the same
   sample: the 4 most common gray levels (gray) or the 8 most common
   colors (color) cover at least 95% of the pixels, and the values after
   the most common one (the background) cover at least 60% of the pixels
   outside it. The second part keeps a small textured object on a flat
   black canvas (a padded crop, a dark frame) from counting as a mask:
   its background alone can cover 98% of the sample, but its object
   pixels are spread over many values. The check is tuned to miss rather
   than to over-flag: a gray image is judged by the gray limits only, so a
   dark 5 to 8 level frame is never taken for a color label map, and a
   JPEG-saved mask is flagged only while its flat regions still dominate
   (not when compression ringing spreads over more than 5% of the pixels).

   Validation, every tier-3 training image of all 7 classes (7136 images):
   no image of CFP (2000), FAF (194), IR (392), NEG (1750), OCT (1050) or
   PSC (350) is flagged; 64 of 1400 OCTA are, all from one source and all
   genuine binary masks filed under OCTA (vessel segmentation maps and
   foveal avascular zone masks, checked by eye). The highest top-k
   coverage of a real image was 0.81 (CFP), 0.53 (FAF), 0.40 (IR), 0.55
   (OCT), 0.61 (PSC) and 0.91 (a mostly black OCTA frame), against the 0.95
   threshold; 5 NEG images of small objects on black reach 0.99 and are
   kept out by the 60% rule. On HRF-Seg+ (record 16744782) the check flags
   all 225 masks in Folders 1 to 5 (optic disc, optic cup, vessels,
   alpha/beta zones, ground truth; anti-aliased color label maps among
   them, where the exact-count rule caught 128) and none of the 45 color
   fundus photographs in Folder 6.

   Single-channel data deeper than 8 bits is tested on its native values
   (a 128 x 128 nearest sample, same rule) instead of the 8-bit frame:
   scaling to 8 bits only merges values, so a few saturated pixels could
   squeeze a real 12- or 16-bit image into a few gray levels (in a
   simulation on tier-3 images, 136 of 150 pseudocolor frames were flagged
   that way; none are on native values, and 200 of 200 synthetic 16-bit
   masks still are).

   The pixel check also flags flat-colour graphics: in the audit of the 65
   `mask_dominated` records of the previous run, 460 of 10,189 flagged
   images were plots, word clouds, map GIFs, logos and flowcharts, and 38
   records were `mask_dominated` only because of them. The model settles
   it: a flagged image gets the label MASK only when the model's argmax is
   an eye class; with argmax NEG it keeps its model label (NEG or
   UNCERTAIN) and counts as a non-mask image (`n_mask_like_veto`,
   `mask_like_veto: true` in the predictions file). Validated on the
   tier-3 set (the 64 known masks stay MASK, no real image flagged), on
   HRF-Seg+ (213 of 225 masks stay MASK, no photo flagged) and on a replay
   of all 288 classified records of the previous run (no `ok` record loses
   an eye class; 35 graphic records move from `mask_dominated` to
   `no_eye_images`). A flagged graphic whose argmax is an eye class (a pie
   chart as CFP, a scatter plot as OCTA) stays MASK, which is what keeps a
   false eye class out.

   A MASK image counts in the
   thresholded counts (`n_MASK`, `frac_MASK`) and in the argmax counts
   (`argmax_MASK`, `argmax_frac_MASK`) alike; the predictions file keeps
   the model's `top` class and its thresholded `model_label` next to
   `label: MASK` and `mask_like: true`. Like UNCERTAIN, MASK is not a
   modality: the dominant classes, the mean confidence and mean
   probabilities, the eye image fraction and eye class presence are taken
   over the non-mask images (`n_classified_non_mask`), and the dominant
   class is MASK only when every classified image is a mask.

   **Eye images.** A record has eye images (status `ok`) only when the eye
   classes (CFP, IR, PSC, FAF, OCT, OCTA; thresholded labels, stratum
   weights as above) make up at least `--min-eye-fraction` (5%) of its
   non-mask classified images; `eye_image_fraction` is that share. An eye
   class is then listed in `present_eye_classes` (and gets a CMDS
   directory) at `--min-class-fraction` (1%) of the non-mask images.
   Below 5% the record gets `no_eye_images`, no eye class is listed and
   its weblinks are catalogued. This replaces the earlier rule, under which
   a single eye class at 1% of all classified images (masks and UNCERTAIN
   included) made a record `ok`: a handful of stray hits among hundreds of
   NEG or UNCERTAIN images no longer turns a record into an eye dataset,
   and masks neither add to nor dilute the eye share. Rows from runs
   before this rule carry `survey_version` 2.

   **Mask-dominated records.** When masks are at least half of the
   classified images (`frac_MASK` >= 0.5, stratum-weighted) and fewer than
   50 non-mask images are classified, the eye decision would rest on a
   small remainder, often masks the pixel check missed. Such a record is
   never `ok` from that remainder: its status is `mask_dominated`
   (`mask_dominated` true), the eye classes the rule above finds in the
   remainder go to `review_eye_classes` (review only) and not to
   `present_eye_classes`, the CMDS structure description gets no modality
   directories, the Record_Classes sheet no rows, and its weblinks are
   catalogued. A record with 50 or more non-mask images follows the
   normal rule whatever its mask share, and so does a paired segmentation
   dataset: when the non-mask images hold at least 10 images of one eye
   class other than OCTA with a mean top-1 confidence of at least 0.75
   (`mask_exempt_class`), the record is judged by the normal rule (7678656:
   28 fundus photos next to 28 vessel masks; HRF-Seg+). OCTA never exempts,
   because missed masks come out as OCTA. Rows from runs before this rule
   (and the dominance mask test) carry `survey_version` 3; rows before the
   format conversion, the mask veto and the status split carry version 5
   or lower.
5. **Describe.** Per-image facts are rolled up into DICOM-aligned columns
   (see below). The two CMDS JSON files are written to
   `<out-dir>/cmds/<id>/` and validated against the bundled CMDS v0.1.1
   schemas. Validation errors are recorded and never raised.
6. **Weblinks.** When a record has no files, none of its files gave a
   usable image, its eye images are under 5% of its non-mask images
   (status `no_eye_images`) or it is mostly masks (status
   `mask_dominated`), its links are catalogued. They come from the scrape's `external_links`,
   discovery's `_weblinks` and `_dataset_links`, and the DataCite
   `relatedIdentifiers`. Each link is cleaned (the scraper glued words such
   as "The" or "(updated" onto some URLs) and de-duplicated. Links to the
   record itself are dropped. Each remaining link gets a category
   (`dataset_repository`, `code`, `paper_doi`, `institutional`,
   `cloud_drive` or `other`), a `dataset_likely` flag and an optional HTTP
   status. The status check uses a 10 s timeout and allows one request per
   second. A 429 sets a cooldown for that host from Retry-After and is
   retried once; a host cooling down for more than 2 minutes is not queried
   again in that run. Zenodo links and Zenodo DOIs go through the same
   shared throttle and 429 cooldown as the metadata calls and downloads.
   Only definitive answers (2xx, 3xx, 4xx other than 408 and 429) are
   cached; 5xx, 429 and transport errors are checked again on the next run.
7. **Result.** One JSON line per record is appended to
   `<out-dir>/survey_results.jsonl`, and the record's scratch dir is
   deleted. If a killed run left the file without a final newline, one is
   added at start so the next line is not glued onto the broken one.
   Per-image predictions go to `<out-dir>/predictions/<id>.jsonl.gz`.
   Records already in the JSONL are skipped on restart. The Excel step uses
   the last line for each record.

   Before a record starts, a `started` line is appended. If the process is
   killed inside a record (OOM killer, hang, manual stop), the next run
   turns that line into a `crashed` row, deletes the record's scratch dir
   and moves on; use `--retry-status crashed` to try it again. At start the
   run also removes any leftover per-record scratch dir (sub-directories of
   `<scratch-dir>/records/` named by a record id), so run one survey process
   per scratch dir. The survey holds a lock file in its out dir and its
   scratch dir while it runs, and a second process pointed at either stops
   at start with an error (concurrent processes need their own; see Two
   processes at once).

   The survey only deletes inside a scratch dir it owns. On first use it
   writes a `.envision_survey_scratch` marker; it refuses to start on an
   existing, non-empty scratch dir without that marker, and it deletes only
   under `<scratch-dir>/records/`. Relative paths are resolved before the
   scratch dir is checked against the downloads dir, so a data folder
   passed by mistake is never emptied.

Memory stays small on a 7 GB VM:
- archive members are read one at a time;
- members over 256 MB spill to a temp file instead of RAM;
- large JPEGs are decoded at a reduced DCT scale, large TIFFs strip or
  tile at a time, large arrays and volumes by strided slices; a TIFF whose
  one strip or tile is itself over the 1 GiB decode budget (ImageJ and
  some scanners write the whole plane as one strip) is strided through a
  memory map when uncompressed and gets `too_large` when compressed;
- 16-bit and float planes are windowed to 8 bits in one float32 working
  copy, scaled in place (percentiles from a 1M-sample stride);
- only compact facts and probabilities are kept per image;
- remote members and nested archives stream to disk in both fetch paths;
  the range reader keeps at most 64 MB in its block cache and prefetches
  at most 8 MB in one request.

### Statuses

| Status | Meaning |
|---|---|
| `ok` | eye classes make up at least `--min-eye-fraction` (5%) of the non-mask classified images |
| `no_eye_images` | images were classified, but eye classes are under 5% of the non-mask images (weblinks catalogued) |
| `mask_dominated` | masks are at least half of the classified images (`frac_MASK` >= 0.5) and fewer than 50 non-mask images are classified (an all-mask record included); eye classes found in the remainder are listed in `review_eye_classes` only, the CMDS structure has no modality directories (weblinks catalogued) |
| `no_image_files` | nothing pixel-bearing was listed, or every candidate turned out not to be one (no image-shaped dataset, wrong format; when every sampled file is such a file, the unsampled ones of the record count the same way, `not_sampled_like_sample`); weblinks catalogued |
| `images_unreadable` | pixel-bearing files were listed but none gave a classified image (decode or codec failure, fetch failure, over a size cap, not fetched or not sampled, blank frames); the reasons are in `n_pixel_files_unread_by_reason`; weblinks catalogued |
| `no_images_in_archives` | every file to read was a non-zip archive whose probed listing holds nothing usable, so nothing was downloaded (see Archive probe) |
| `no_usable_images` | runs before survey version 6 only: the two statuses above in one |
| `no_files` / `restricted` | Zenodo lists no files (weblinks catalogued) |
| `skipped_disk` | download would breach the disk floor; retry with `--retry-status skipped_disk` |
| `skipped_size` | none of the record's non-zip files fit `--max-download-gb` and nothing else was readable; file list and weblinks catalogued |
| `remote_listing_failed` | the record is read only through remote zips and every zip listing failed (429s, 5xx, broken bodies); retry with `--retry-status remote_listing_failed` |
| `error` | unexpected failure; message and traceback are kept in the JSONL (pipeline: also a record whose spooled files failed validation again after two refetches) |
| `crashed` | the process died while this record was running (pipeline: `--max-attempts` starts without a finish); skipped until `--retry-status crashed` |
| `started` | only while a run is in progress: the record being processed now |
| `awaiting_deep` | pipeline only, while in progress: the triage sample holds eye classes and the deep pass is being fetched; replaced by the record's final row |
| `metadata_unavailable` | the record JSON could not be fetched from Zenodo (retries used up, or the record is gone); weblinks from the scrape catalogued; retry with `--retry-status metadata_unavailable` |
| `*_partial_download` | suffix added when some files or remote members failed to download, or some remote zip listings failed (`remote_listing_failures`); retry with e.g. `--retry-status ok_partial_download` |

`skipped_disk`, `skipped_size`, `error` and `crashed` records still get a metadata-only
`dataset_description.json` built from the Zenodo metadata, with an empty
structure description; the `cmds_note` column says why.

## DICOM-aligned attributes

These are DICOM-aligned attributes, not DICOM conformance: a JPEG is never a
DICOM instance. The rules live in `survey/resources/dicom_mapping.json`, which was
checked against DICOM PS3 2026d. `dicom_mapping_status` records where the
values came from:
- `read_from_header`: sampled DICOM files of the dominant class exist, and
  the record's Modality, SOP Class UID and Image Type are their majority
  header values. Body Part Examined, Anatomic Region, Acquisition Device
  Type and Ophthalmic Image Type are taken from those headers too when they
  carry them (listed in `dicom_header_attributes_used`), else from the
  class mapping;
- `mixed`: DICOM files were sampled, but none of the dominant class (for
  example a CT series next to fundus JPEGs), so those values come from the
  class mapping;
- `inferred_from_classifier`: no DICOM was sampled; the values follow from
  the predicted class;
- `not_applicable`: the record has no classified images.

Patient, institution, UID and device serial tags are never read into the
outputs. Header values of every sampled DICOM file are also listed in the
`dicom_header_*` columns and compared with the classifier in a QC column.

- **Modality and SOP Class.** OP with Ophthalmic Photography 8 Bit, or 16
  Bit when the images store 9 to 16 bits, for CFP, IR, PSC and FAF. Float
  or deeper than 16-bit data fit no OP SOP class without rescaling; they
  keep the 8-bit class and are flagged in `dicom_pixel_depth_note`. OPT
  with Ophthalmic Tomography for OCT. OPTENF with Ophthalmic OCT En Face for
  OCTA. OT with Secondary Capture for NEG.
- **Codes.** Body Part Examined EYE. Anatomic Region: Retina, Optic nerve
  head or Anterior chamber for OCT, switched by keywords in the paths of
  the OCT images, or by explicit phrases in the record text ("anterior
  segment", "AS-OCT", "anterior chamber", "peripapillary", ...; single words
  such as "angle" or "cornea" in prose do not count). Acquisition
  Device Type: Fundus Camera, SLO or OCT scanner. Illumination Type
  (0022,0016) where the mapping has one (CFP). OCTA slab codes (CID 4271)
  come from path keywords (svp, dcp, cc, ...), else from explicit phrases in
  the record text ("deep capillary plexus", "choriocapillaris", ...); single
  words such as "deep" or "cc" in prose never count ("deep learning",
  "CC BY"). When more than one slab is named, the generic Retina
  vasculature flow code is kept.
- **Pixel facts, over the sample.** Rows and Columns ranges, Samples per
  Pixel, Photometric Interpretation, Bits Allocated, Bits Stored, High Bit,
  the maximum Number of Frames, Lossy Image Compression and its Method
  (ISO_10918_1 for JPEG, ISO_15444_1 for JPEG 2000), and Image Comments
  from the TIFF ImageDescription. From real DICOM headers: Software
  Versions, Horizontal Field of View, Illumination Wave Length and Series
  Description. Pixel Spacing is read only from real physical resolution:
  TIFF resolution tags in cm or inch units (screen dpi values are ignored),
  ImageJ units or OME PhysicalSize.
- **Manufacturer.** Taken from the DICOM header first, then EXIF Make and
  Model, then proprietary extensions (`.e2e`, `.fds`, ...), then text
  patterns in the title, description and file names.
- **Laterality.** For sampled DICOM files, the header ImageLaterality
  (0020,0062), then Laterality (0020,0060). For every other image file, the
  path: OD/OS/OU, right/left, and a delimited R/L when the name has another
  field. Conflicts become U. `laterality_source` counts the files per
  source (dicom_header, basename, dirname).

## CMDS / AI-READI metadata

The standard is cited as the Clinical Multimodal Data Structure (CMDS)
v0.1.1, <https://cmds.aireadi.org> (it resolved when checked on 2026-09-24).
CDS was renamed CMDS; the v0.1.1 schemas have the same validation rules as
CDS v0.1.1, and they are bundled in `survey/resources/schemas/`.

- `dataset_description.json` (`schema`
  `https://schema.aireadi.org/v0.1.0/dataset_description.json`) is filled
  mostly from DataCite JSON:
  - creators with ORCID and ROR;
  - rights, with the id mapped to its canonical SPDX spelling
    (`cc-by-4.0` becomes `CC-BY-4.0`) and labelled SPDX only then; other ids
    keep the scheme "Zenodo license id";
  - affiliation and managing organisation identifiers with their declared
    scheme (ROR, ISNI, GRID) and matching scheme URI;
  - dates with their DataCite dateType (Issued, Updated, ...), related
    identifiers, descriptions and funding.

  `subject` merges the DataCite subjects, the Zenodo keywords and the
  classifier's modality terms, one entry per subject text. A DataCite
  subject with a scheme and a classification code (or a value URI, whose
  last segment becomes the code) keeps them as `subjectIdentifier`.
  `resourceType.resourceTypeValue` and the extra `subject` terms come from
  the classifier.
- `dataset_structure_description.json` holds the classifier's composition
  as dataType and modality directories with NCIT and MeSH terms. The
  DICOM standard is added to a dataType directory only when its own
  classes had sampled DICOM files. Its
  `schema` must be the v0.1.0 URL even in CMDS v0.1.1. The schemas forbid
  extra keys, so the numbers are written into `directoryDescription`:
  - `numberOfFiles` is estimated as total images x class fraction in the
    sample;
  - the sample size and mean confidence go in the description text.
- De-identification and consent. Zenodo cannot supply
  `datasetDeIdentLevel` or `datasetConsent`, and the schema requires them.
  It removed ECRIN's "NotKnown" option from both enums, so every allowed
  value asserts something. The survey states what it bases each value on:
  - De-identification (project decision, applied to every record as a
    default because the depositors do not report their methods). Direct
    identifiers are taken as removed: `deIdentType` `DeIdentificationApplied` ("some
    de-identification measures have been applied; details in the boolean
    fields and/or separate documents") with `deIdentDirect` true
    ("direct identifiers were removed from the data set"). The depositors
    do not report their methods, so `deIdentHIPAA`, `deIdentDates`,
    `deIdentNonarr` and `deIdentKAnon` are false, which here means "not
    reported", not "confirmed absent". The enum and the five booleans are
    the same for every record, metadata-only documents included; only the
    `deIdentDetails` text and the provenance depend on the record, so the
    document never contradicts its own `accessDetails`:
    - open records with image files (images counted or classified):
      "Publicly released on Zenodo as image files", the methods not
      reported by the depositor, false flags meaning not reported, not
      confirmed absent. Provenance `derived (public release on Zenodo as
      image files; methods not reported by the depositor)`.
    - restricted, closed and embargoed records: "Access restricted by the
      depositor on Zenodo" ("Embargoed by the depositor on Zenodo until
      <date>" when the legacy JSON or enrich-access gives the embargo end;
      "Access closed ..." for closed records) "(access_right: <x>; files
      not publicly released, no image files read), so de-identification is
      not reported", followed by the statement that the enum and flags are
      only the project default the required field needs, not a statement
      about these files. Provenance `derived (Zenodo deposit, access_right
      <x>; access restricted by the depositor, de-identification not
      reported)`.
    - open records with no image file read (no files, no image files, or
      files not read): "Deposited on Zenodo (access_right: open; no image
      files read)", followed by the statement that the flags are a project
      default because the depositor does not report the methods, and the
      same not-reported meaning of the false flags. Provenance `derived
      (Zenodo deposit, access_right open; no image files read; methods not
      reported)`.

    Both provenance strings start with `derived`, so the field is listed
    with the derived fields. `NoDeIdentification` would claim it was
    confirmed that nothing was done, and
    `DeIdentificationAppliedPrimaryOutcomesReAssessed` claims a
    re-analysis.
  - Consent stays a placeholder with the least assertive value:
    `consentType` `ConsentSpecifiedNotElsewhereCategorised` ("a
    descriptive statement regarding consent is available, but it does not
    fit into categories 1 to 5"), whose statement is the
    `consentsDetails` text. Every other value names a consent level
    (`NoExplicitConsent`, `NoRestriction`, `GeneralResearchUse`, ...). All
    restriction booleans are false, meaning none was reported, not that
    none exists.
    `consentsDetails` starts with "Not reported by source (Zenodo)" and
    explains the choice; the field's provenance is `placeholder (not
    reported by source)`.

  Each record lists its placeholder and derived fields
  (`cmds_placeholder_fields`, `cmds_derived_fields`).
- `accessDetails.url` is never emitted. The schema's pattern rejects any
  URL containing the letter "s" (an upstream bug), so the landing URL goes
  into the description text and `alternateIdentifier` instead.
  `urlLastChecked` is omitted with it, since no URL was checked.

## Restricted and embargoed records (`enrich-access`)

A restricted Zenodo record has public metadata but hidden files: the file
list is empty and the owner grants access case by case through "Request
access" on the landing page. The survey gives such records the status
`restricted` and reads no file. `enrich-access` is a side command that adds
what a person needs to decide whether to ask for access, without touching
the survey's results:

```bash
envision-survey enrich-access --scrape results/zenodo_full.json \
    --survey-out-dir results/survey_pull --out-dir results/survey_pull_access \
    --rpm 10 --shared-state-dir results/survey_pull_state
envision-survey excel --results-dir results/survey_pull --results-dir results/survey_pull_access \
    --out results/zenodo_modality_survey.xlsx
```

Targets are the scrape records with `access_type` restricted or
embargoed, and the records whose last results row has status restricted,
embargoed or closed, or an `access_right` of restricted, embargoed or
closed. For each it fetches the InvenioRDM record JSON (`Accept:
application/vnd.inveniordm.v1+json`, one request, cached under
`<out-dir>/cache/invenio`) and writes one line to
`<out-dir>/access_results.jsonl`:

| Field | From |
|---|---|
| `access_record`, `access_files`, `access_status` | the record's `access` block (`public` / `restricted`; status `open`, `restricted`, `embargoed`, `metadata-only`) |
| `access_embargo_active`, `access_embargo_until`, `access_embargo_reason` | `access.embargo` |
| `access_files_public_now` | files `public` and no active embargo |
| `access_allow_user_requests`, `access_allow_guest_requests`, `access_accept_conditions_text` | the parent record's `access.settings`, which Zenodo shows to any reader: whether the owner accepts requests from logged-in users, from guests, and the owner's conditions. Empty when Zenodo does not expose them (`access_settings_exposed` false), never guessed |
| `access_request_url`, `access_request_api` | the landing page (the form is there) and `links.access_request` |
| `access_request_note` | one line on how to get the files |
| `access_license_ids`, `access_license`, `access_license_use` | `metadata.rights` (the scrape's license when the record has none): `train` (CC0, CC-BY, CC-BY-SA, ODC-BY, PDDL, MIT, BSD, Apache, public domain), `evaluate_only_nc_nd` (any NC or ND license: internal evaluation only, never training), `no_license` (do not train), `check_license` (anything else) |
| `access_resource_type`, `access_title` | `metadata` |
| `access_setfit_label`, `access_setfit_prob` | the discovery scrape |
| `access_keyword_terms`, `access_keyword_match` | eye imaging terms in title, keywords, subjects and description |
| `access_eye_relevant` | SetFit label EYE_IMAGING, p(eye imaging) at least 0.5, or a keyword match |

For records whose files are still not public it also writes the two CMDS
documents again from the full metadata (legacy and DataCite JSON, read
from the survey's `cache/` when there, else fetched into its own cache)
into `<out-dir>/cmds/<id>/`, with the non-public de-identification text
above and an `accessDetails` text that says whether the owner takes
requests and until when an embargo runs. Records whose files have become
public keep the survey's own CMDS.

The job is resumable (records with a successful row are skipped; a failed
fetch is tried again on the next run; `--refresh` fetches all again). It
never writes into the survey's out dir: an `--out-dir` that is, contains or
lies inside `--survey-out-dir`, or holds a `survey_results.jsonl`, is
refused. Every request goes through the survey's Zenodo client, so with
the running survey's `--shared-state-dir` both stay under `--shared-rpm`
together and share a 429 cooldown; `--rpm` (default 10) is the job's share.
Progress and counts go to `<out-dir>/access_summary.json`.

In `excel`, a `--results-dir` holding `access_results.jsonl` is a
supplement: its rows set only the `access_*`, `cmds_*`, `dd_*` and `dsd_*`
fields of the survey rows with the same id (the last supplement dir given
wins; its CMDS folders resolve against its own dir), never the status or
classification columns, and it adds the Access_Requests sheet.

Access is granted to a Zenodo account: once an owner accepts a request,
the files open for that account on the website and for API requests made
with a personal access token of the same account (`--token-file`), so a
later survey run with that token reads them.

## Workbook sheets

| Sheet | Content |
|---|---|
| README | method, sampling cap, threshold, model and checkpoint hash, status counts, mask-dominated record count |
| Records | one row per record: sampling mode, file and image counts (images seen in listings, estimated population, remote members sampled and fetched), remote byte columns (members never sampled for size, sampled members not fetched for the record budget, remote bytes fetched), thresholded class counts and fractions (incl. UNCERTAIN and MASK) next to argmax (threshold-free) counts and fractions (incl. MASK), images classified without masks, eye image fraction, mask-dominated flag and review-only eye classes, dominant class both ways, dominant modality, mean confidence, IR device hint and IR non-Spectralis candidate flag, SetFit label, DICOM-aligned columns, laterality and its source, the main dataset_description fields (dates, subjects, related identifiers, funding, language, format, description), the CMDS JSON folder (relative to the results dir, and in full as the results dir plus `cmds/<id>`, so a merged workbook is unambiguous), CMDS validity, placeholder, derived and classifier-set fields |
| Record_Classes | one row per record and present modality with its DICOM mapping and CMDS directory |
| Weblinks | catalogued links of records without usable images or without eye images |
| DICOM_Mapping | reference table of the class to DICOM mapping |
| Schema_Fields | each AI-READI / CMDS field, its mapping rule and how many records took it from which source (directoryList counts only records with at least one modality directory) |
| Archive_Probes | one row per archive probed before a download: outcome, method, members, images, bytes, requests |
| CMDS_JSON | the two CMDS JSON documents of every record, as text, with the full CMDS JSON folder (results dir plus `cmds/<id>`); above 2000 records (`--cmds-json auto`, the default) their file paths and sizes instead, so the workbook stays small; `--cmds-json embed` or `paths` forces one form |
| Access_Requests | only with an `enrich-access` supplement dir: eye-relevant records whose files are not public, ranked by the discovery SetFit p(eye imaging): title, SetFit label and probability, eye keywords, license and license use, resource type, access status, embargo, whether the owner accepts requests, the owner's conditions, the landing page to request access on, the API link and the survey status |
| Formats | classified images by source format, files that gave no image by source format, and classified images by conversion path, each with its record count, over all records |
| Model_Training_Datasets | only with `--training-sources <csv>`: the classifier's training and evaluation sources, one row per (model_version, source, class), copied from the CSV as given (license, role, image counts; a retrain appends rows under a new model_version). The CSV needs a `model_version` column; the README gets a per-version summary (sources, classes, training and test images, synthetic and evaluation-only sources, and sources the CSV's `training_policy` does not mark `allowed`). The CSV is kept with the model, not in this repository |

The workbook is written in openpyxl's write-only mode from an index of the
results file (byte offsets of each record's winning line), one row in
memory at a time, so the 30,000-record pull builds in bounded memory. The
Records sheet carries the discovery SetFit label and p(eye imaging) as
informational columns, labelled as not used to select records.

### IR device columns

For a record whose dominant class is IR (thresholded or argmax) or whose IR
fraction of the non-mask images is at least 20% (thresholded or argmax;
`frac_IR_non_mask`, `argmax_frac_IR_non_mask`, weighted like `frac_*`, so
masks cannot dilute it), `ir_device_hint` lists the
manufacturer evidence found. The dominant eye class does not count: in a
record that is 97% NEG, 3% IR would win it. Strong evidence is the DICOM
header or EXIF maker of the IR images themselves, where an image counts as
IR when its thresholded label or its argmax class is IR (the same
population as the trigger, so IR images below the threshold still give
their maker) (labelled
"(IR images, dicom_header)" or "(IR images, exif)", also in the
`ir_Manufacturer` columns); the record-level maker, which may come from
other images such as Zeiss OCT DICOMs, is not used. Weak evidence is the
record-level text, file name and vendor extension hints, labelled
"(record-level text)" and so on. It says "no manufacturer hint found"
when there is none. `ir_non_spectralis_candidate` is TRUE when the evidence
names a maker that is not Heidelberg (Spectralis, HRA, HRT, `.e2e`,
`.sdb`), so IR from other devices stands out. When the IR images carry a
maker (strong evidence), that maker alone decides: a Spectralis or `.e2e`
mention in the record text, often about its OCT, cannot hide a Topcon or
Optos IR device. The weak record-level hints decide only when the IR
images carry no maker. A record without any evidence is not a candidate,
since its device is unknown. Rows written before these columns existed
are filled in when the workbook is built.

### Eye modality review flags

The image classifier knows only the six eye modalities and NEG, so non-eye
grayscale images can land in an eye class. The known false-positive
pattern is OCTA (sometimes OCT) at a mean confidence near 0.7 for cell
microscopy and binary masks (7634536, mtFociCounter: 89% OCTA, 90%
mask-like), X-ray and CT slices (14847200, NACHOS: 71% OCTA) and tissue OCT
(10782863, colorectal tissue OCT: every image argmax OCT, 87% above the threshold). Such records still get schema-valid
CMDS documents that name an eye modality. For every record with an eye
modality, `modality_review_flags` lists:
- `setfit_not_eye`: the SetFit metadata classifier gives p(eye imaging)
  under 0.2 (or labels the record NEGATIVE when it has no probability);
- `mask_like`: at least 30% of the classified images are mask-like (gray
  with at most 4 levels, or color with at most 8 colors), by `frac_MASK`
  (weighted like the other `frac_*` columns). `mask_like_frac` is the
  unweighted share of the decoded sampled images and can differ. Those images are
  labelled MASK and count in no eye class, but a high share still marks a
  segmentation dataset whose other images (cell microscopy, for example)
  may not be eye images either;
- `low_confidence`: the dominant eye class has a mean top-1 confidence
  under 0.75.

`modality_low_trust` is TRUE on `setfit_not_eye` or `mask_like` (low
confidence alone is reported but does not set it). Nothing is removed: the
counts and CMDS documents stay, and the flags are also written into the
`directoryDescription` of every CMDS directory of that record, prefixed
with "REVIEW". Rows written before these columns existed are filled in
when the workbook is built.

## Throughput and tests

Measured on the 2-core, 7 GB Azure VM:
- 7505822 (RFMiD 2.0, nested zips): 860 JPEGs classified in 26 s.
- 4521044: 186 images in 35 s.
- Peak RAM stayed under 2 GB.
- A record at the 2000-image cap takes about a minute plus its download
  time.

`tests/test_survey.py` runs offline, with no network and no model. It
builds synthetic zips, a tar.gz nested in a zip, an extensionless
multi-frame DICOM, a 16-bit TIFF and a float GeoTIFF stack. It covers the
walker, the loaders, laterality, the manufacturer guards, weblink cleaning,
split-set co-location and CMDS schema validity. It also covers the
robustness fixes: crash recovery, the volume and 7z disk caps, nested
archive cleanup, no duplicate read errors, parity with the training
squash (and that the sides of wide images are kept), DICOM status and header
laterality, dateType, SPDX and scheme mapping, the shared download throttle
and the workbook sheets. The second review round added tests for the
scratch-dir marker, relative-path guard, truncated JSONL lines, OCTA slab
phrases, DICOM header attributes, the pixel-module columns and 16-bit rule,
compressed non-image downloads, planar TIFFs, DataCite subject codes, the
link checker's 429 and cache rules, metadata-only CMDS documents, and
password-protected 7z and zip archives (with the 7z binary installed).
The full-run hardening added tests, all with a mocked Zenodo HTTP server,
for: the sliding-window throttle (never more than the limit in 60 s, on
a fake clock) and the 429 cooldown; retries with exponential backoff and
no retry on 404; container listings, the range-read fallback for truncated
listings and member fetches by range when the container endpoint refuses;
that a range read never pulls a whole file; proportional, seeded sampling
over several zips; nested archives; a full remote_zip record (sampling
mode, listed versus sampled counts, argmax columns, estimates scaled to the
listed population, laterality from unfetched names, IR device fields);
top-level image sampling; `skipped_size`; the least assertive CMDS
placeholders; and the new workbook columns with backfill of older rows.
The third review round added tests for: range reads retried after a
broken body, a cache bounded by bytes and a capped prefetch; streamed
range fetches of stored and deflated members; nested archives over the
cap never fetched; no container retries for a zip whose container listing
failed; `remote_listing_failed` and listing failures in
`_partial_download`; stratified class weights for mixed records; the
smallest-first download budget with split sets kept whole; the IR device
trigger ignoring the dominant eye class and using IR-image makers only;
one tie rule for dominant classes; the throttle not holding its lock while
it sleeps; and the model hash check against the sidecar. The fourth
review round added tests for: container listings that are short without
saying so (total above the page, or 1000 items); flat, unique scratch names
that keep the suffix, with long, clashing and colliding member names all
fetched and a scratch path error failing one member only; the separate
nested budget, per-archive cap and empty-archive stop; separate `remote`
and `remote_nested` strata with weighted fractions; IR maker evidence from
below-threshold IR images; the IR images' own maker deciding the
non-Spectralis flag; the eye modality review flags in the row, the CMDS
text and the workbook; and the export's survey-path parity module loading
both in the package and standalone. Each new guard was mutation-checked:
reverting it makes its test fail. The fifth round added tests for: the
de-identification values (`deIdentDirect` true, the other flags false,
the details text and provenance) with consent unchanged; members over the
per-member cap replaced by other members, and the record byte budget
stopping member fetches (no request past it, dropped members still
listed) with the new row columns; the mask check on label maps, a mask
with chroma residue, a dark low-contrast gray frame, a mostly black
OCT-like frame and a JPEG-saved mask; the MASK label in the thresholded
and argmax counts, the predictions file keeping the model's class, and an
all-mask record (dominant MASK; `no_eye_images` then, `mask_dominated`
since the seventh round); the eye status at 4%
and 6% eye images with masks not diluting the share; `partition` (every
mode, a mixed record, a record without metadata, `--no-remote-zip`);
`--ids-file`, `--rpm` and the refusal of an empty id list; the per-dir
process lock; and a workbook merged from two results dirs. The sixth round added tests for:
a no-eye record (mostly UNCERTAIN, argmax IR) getting no IR device fields;
the record byte budget counting the bytes of failed transfers (container
bodies that fail their CRC check before a range read) and stopping the
fetches once they are used up; the combined request budget of two
processes sharing a state dir; disk reservations of other processes
counted on the same file system and removed when their owner is gone; the
download check subtracting them; a download stopped at the disk floor
while it streams; the lock fallback (warning, no scratch sweep); and the
de-identification text of closed, restricted and image-less open records.
Each of these guards was mutation-checked as well. The seventh round added
tests for: the dominance mask test (an anti-aliased binary mask with 16
edge levels and an anti-aliased 5-color label map flagged; a small textured
object on a black canvas and a dark 8-level frame not flagged; the 95%
coverage bar between four flat bands with 2% and 10% noise; a JPEG-saved
blocky mask now flagged, a JPEG of a textured frame not); a
mask-dominated record with a small non-mask remainder (status
`mask_dominated`, review-only eye classes, no CMDS modality directories,
no Record_Classes rows) next to a record with the same mask share and 50
non-mask images (status `ok`); and the full, results-dir-qualified CMDS
JSON folder in the workbook. Each new guard was mutation-checked.

`tests/test_pipeline.py` covers the full-pull architecture, offline as
well (the producer and consumer run in two threads of the test process
against a mocked Zenodo; the supervisor test starts real child
processes): the token from the environment, an explicit file or the
default file, and its redaction from log records (tracebacks included),
result rows, events, predictions and state files, with the token carried
as a file name and a title through a whole pipeline run; the triage
sample as a prefix of the deep sample (also where largest-remainder
rounding would shrink a zip's share), the deep pass reusing the triage
members and listings, eye-only deep passes for remote, local and
top-level images, triage decisions taken on triage-pass images only, and
no second deep request; the spool guards (overlap with the downloads and
output dirs, the marker, deletions under `records/` only and never through
a symlink, size and path validation, the state machine); the disk gate
(spool budget, floor, deep bypass, stop without a consumer); the
producer and consumer end to end (deep pass, NEG triage, no pixel files,
a local original read in place and left untouched, empty spool and no
requests at the end); the row written before the spool dir is deleted;
the crash-loop guard, refetch on invalid spool items and the give-up
after repeated refetches; skipping finished and spooled records and
redoing partial ones; the stop when the consumer is gone; the monitor's
counts, ETA, incremental reads and dead-role state; the supervisor
restarting a crashed role up to its cap, and a consumer killed inside
one record twice leaving a `crashed` row for it and classifying the
next; the resume leaving spooled records to the consumer and skipping
records finished since start; restored deep requests; requests written
right after publishing surviving; the double end check; the room gate's
atomic reservation and its yield to requests (a full spool of records
awaiting their deep pass); single-strip TIFFs over the budget viewed or
refused, never decoded whole; the in-place windowing memory bound; the
metadata prefetch and
its resume; the cost order; the streamed workbook (sheet order, Formats,
CMDS_JSON as paths or text, the SetFit column label); two downloads with
one base name; and the pipeline options. Each of these guards was
mutation-checked.

## Known limits

- Remote zip sampling reads at most `--remote-cap` images per record, so
  class fractions of big records rest on 300 images. In a record with
  local files and remote zips the remote part is weighted up to its
  population, so its fractions rest on those 300 images too. The same
  goes for nested archives: a few fetched archives stand for all listed
  ones.
- No record in the current corpus is partly downloaded (every record on
  disk is complete), so the mixed local plus remote_zip mode was checked on
  a smoke copy with one zip of a remote record placed in a separate
  downloads dir.
- The classifier has no "other medical image" class: non-eye grayscale
  images can be predicted as OCTA or OCT (see Eye modality review flags).
- The mask check sees few-valued images, exact or dominated by a few
  values (anti-aliased edges). Masks whose JPEG ringing or interpolated
  edges hold more than 5% of the pixels (thin vessel maps saved as JPEG,
  large blobs resampled with interpolation), label maps with more
  than 8 colors or jittered colors, and probability maps keep a model
  label. A figure or diagram with 8 or fewer dominant flat colors is
  labelled MASK rather than NEG, which leaves it out of the non-mask
  images the eye share is taken over.
- Images inside nested archives that were not opened are extrapolated from
  the opened ones; `n_images_listed` counts only what was seen.
- The container API serves zip only. A remote RAR, 7z or tar is downloaded
  whole when the record is under `--max-download-gb`, else it is not read.

- Split archives need every part. When some parts are already downloaded
  and others are streamed, the local parts are symlinked next to the
  streamed ones. A split set spread over several Zenodo records cannot be
  read. For example, ULS23 Part 5 (10057471) holds only `images.z12` to
  `.z20` of another split set, so it reports no usable images.
- Vendor OCT containers (`.e2e`, `.fds`, `.fda`, `.vol`, `.sdb`) are
  counted and used as manufacturer hints but not decoded. The same goes for
  video, `.h5`, `.mat` and `.npy`.
- Class counts come from a sample of at most `--max-images` files, so the
  estimated per-modality file counts are extrapolations.
- The classifier was trained on synthetic FAF, and its real-FAF recall is
  modest. FAF counts should be read with that in mind.
