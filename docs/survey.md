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
  sent fewer than `--shared-rpm` (110) in the last 60 s. Two processes left
  at the default `--rpm 110` therefore cannot send 220 per minute; `--rpm`
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
| `--rpm` (or `--zenodo-rpm`) | 110 | Zenodo requests per sliding minute for this process, all threads together (Zenodo allows about 133 per IP); with `--shared-state-dir` it sets this process's share, e.g. 90 + 20 |
| `--shared-state-dir` | none | dir shared by processes running at the same time: a 429 in one pauses all of them, their requests together stay under `--shared-rpm`, and each subtracts the others' in-flight downloads on the same disk before checking the disk floor |
| `--shared-rpm` | 110 | Zenodo requests per sliding minute of all the processes sharing `--shared-state-dir` together; ignored without it |
| `--zenodo-interval` | 0.5 | minimum seconds between two Zenodo requests |
| `--download-all` | off | also fetch documents, tables, code and arrays |
| `--min-eye-fraction` | 0.05 | a record has eye images (status `ok`) only when the eye classes make up at least this fraction of its non-mask classified images (thresholded labels) |
| `--min-class-fraction` | 0.01 | in a record with eye images, fraction of the non-mask classified images for an eye class to be listed as present |
| `--ids`, `--ids-file` | all records | only these record ids; `--ids-file` (repeatable) reads them from a file, one per line, e.g. from `partition` |
| `--no-link-check` | off | skip HTTP HEAD checks of weblinks |
| `--retry-status` | none | reprocess records with these statuses, e.g. `error,skipped_disk,skipped_size,remote_listing_failed,ok_partial_download` |

## How a record is processed

1. **Metadata.** The legacy record JSON comes from `--metadata-dir` when it
   is there, otherwise from the Zenodo API. The DataCite JSON comes from
   `GET /api/records/<id>` with `Accept: application/vnd.datacite.datacite+json`.
   Both are cached under `<out-dir>/cache/`. Every Zenodo request (metadata
   calls, downloads, zip listings, member fetches and range reads, across
   all worker threads) goes through one shared throttle: at most 110
   requests in any sliding 60 s window (`--rpm`) and at least 0.5 s
   between two requests (`--zenodo-interval`). A 429 sets a shared cooldown
   from Retry-After (seconds or HTTP-date; at least 60 s when the header is
   missing) that every worker waits on, and with `--shared-state-dir` every
   other process sharing that dir as well; with that dir the processes
   together also stay under `--shared-rpm` (110) per sliding minute. 5xx
   responses and transport errors
   back off exponentially per request (2 s doubling up to 120 s).
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
     retry cycle on the container endpoint per member. A stored or deflated
     member is read with one streamed range request and inflated on the
     fly into the scratch file, so memory stays flat for any member size;
     other compression methods go through `zipfile` in block-sized reads.
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

     Two byte bounds keep one record from pulling gigabytes. Members over
     `--remote-max-member-mb` (200 MB) are never sampled: they are left out
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
     whole into `scratch_dir/records/<id>/dl/`: rar, 7z, tar and gz
     archives and their split parts, images, DICOM, volumes, extensionless
     files, and single-file `.gz`/`.bz2`/`.xz` files whose inner name is an
     image or an archive (`x.tif.gz`, but not `x.csv.gz` or `x.h5.gz`).
     Top-level image files beyond `--remote-cap` are sampled (seeded); the
     others are counted and their names used for laterality and hints.
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

   Documents, code, tables, arrays, genomics files, video and vendor OCT
   containers are not fetched unless `--download-all` is set, because
   nothing reads them.
3. **Enumerate.** Archives are listed, not unpacked:
   - zip is read through its central directory;
   - tar goes through one streaming pass;
   - 7z, rar and spanned zip (`.zip` + `.z01`) use `7z l`. RAR members
     are extracted with `unrar`, because many 7z builds list RAR but cannot
     decode it.

   Nested archives and single-file `.gz`/`.bz2`/`.xz` members are extracted
   one at a time into the scratch dir and walked, up to depth 3. An
   extracted archive with no image members, and a decompressed file that
   is not an image, is deleted right after it is walked, so per-patient
   inner zips cannot fill the disk during listing.
   Extensionless members are sniffed for the DICOM `DICM` marker.
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
   `skipped (disk floor)`. Each file gives one frame:
   - raster images give their first frame (the frame count is recorded);
   - multi-frame DICOM gives its middle frame, read with `pydicom`;
   - NIfTI, NRRD and MHA give the middle slice along the shortest axis.
     The uncompressed voxel size is computed from the header first, and
     volumes over 2 GiB uncompressed are inventoried but not loaded (a small
     gzip NRRD can expand to many GB). NIfTI is sliced through the nibabel
     proxy and MHA/NRRD through a SimpleITK extract region, so only the
     slice is materialised where the format allows it.

   Images over 80 MP are inventoried but not decoded, and so are TIFFs
   whose decoded page would exceed 1 GiB (the tifffile fallback reads the
   image length and width, not the shape, so planar RGB pages are sized
   right). Preprocessing rebuilds the training input exactly. First comes a
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

   A flagged image gets the label MASK instead of its model label, in the
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
   normal rule whatever its mask share. Rows from runs before this rule
   (and the dominance mask test) carry `survey_version` 3.
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
- large JPEGs are decoded at a reduced DCT scale;
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
| `no_usable_images` | files exist but none decoded into an image (weblinks catalogued) |
| `no_files` / `restricted` | Zenodo lists no files (weblinks catalogued) |
| `skipped_disk` | download would breach the disk floor; retry with `--retry-status skipped_disk` |
| `skipped_size` | none of the record's non-zip files fit `--max-download-gb` and nothing else was readable; file list and weblinks catalogued |
| `remote_listing_failed` | the record is read only through remote zips and every zip listing failed (429s, 5xx, broken bodies); retry with `--retry-status remote_listing_failed` |
| `error` | unexpected failure; message and traceback are kept in the JSONL |
| `crashed` | the process died while this record was running; skipped until `--retry-status crashed` |
| `started` | only while a run is in progress: the record being processed now |
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
    - restricted, closed and embargoed records, and open records with no
      image file read (no files, no image files, or files not read):
      "Deposited on Zenodo (access_right: <x>; files not publicly
      released, no image files read)" (or "; no image files read" for open
      records), followed by the statement that the flags are a project
      default because the depositor does not report the methods, and the
      same not-reported meaning of the false flags. Provenance `derived
      (Zenodo deposit, access_right <x>; ...; methods not reported)`.

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

## Workbook sheets

| Sheet | Content |
|---|---|
| README | method, sampling cap, threshold, model and checkpoint hash, status counts, mask-dominated record count |
| Records | one row per record: sampling mode, file and image counts (images seen in listings, estimated population, remote members sampled and fetched), remote byte columns (members never sampled for size, sampled members not fetched for the record budget, remote bytes fetched), thresholded class counts and fractions (incl. UNCERTAIN and MASK) next to argmax (threshold-free) counts and fractions (incl. MASK), images classified without masks, eye image fraction, mask-dominated flag and review-only eye classes, dominant class both ways, dominant modality, mean confidence, IR device hint and IR non-Spectralis candidate flag, SetFit label, DICOM-aligned columns, laterality and its source, the main dataset_description fields (dates, subjects, related identifiers, funding, language, format, description), the CMDS JSON folder (relative to the results dir, and in full as the results dir plus `cmds/<id>`, so a merged workbook is unambiguous), CMDS validity, placeholder, derived and classifier-set fields |
| Record_Classes | one row per record and present modality with its DICOM mapping and CMDS directory |
| Weblinks | catalogued links of records without usable images or without eye images |
| DICOM_Mapping | reference table of the class to DICOM mapping |
| Schema_Fields | each AI-READI / CMDS field, its mapping rule and how many records took it from which source (directoryList counts only records with at least one modality directory) |
| CMDS_JSON | the two CMDS JSON documents of every record, as text, with the full CMDS JSON folder (results dir plus `cmds/<id>`) |

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
for: the sliding-window throttle (never more than 110 requests in 60 s, on
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
