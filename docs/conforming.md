# Conformer

Turns raw downloaded archives (e.g. from [envision-discovery](https://github.com/EyeACT/envision-discovery)'s
`--download` step) into structured directory trees following the
[AI-READI dataset description schema](https://schema.aireadi.org/v0.1.0/) (v0.1.0).
Output lands in `data/actionable/{source}/{source_id}/` with a
`dataset_structure_description.json` that describes the layout.

## Pipeline

```
downloaded record
      |
      v
   unpack    extract every .zip/.tar/.7z/.rar/multi-part zip (recursive)
      |
      v
  inventory  walk tree, tag every file by format/modality/readability,
             collect README / PDF / DOCX / label-hint files
      |
      v
   agent     Gemma 4 E4B analyzes the inventory + README + scraped metadata
             and proposes a Recipe(task_type, placements[])
      |
      v
 materialize hardlink (default) or copy source files into target tree
      |
      v
  validate   check dataset_structure_description.json, sample-open one
             file per modality with the right reader
```

## CLI

```bash
# Conform every downloaded zenodo record
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf --source zenodo

# One specific record
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo --source-id 4521044

# Every source directory under ./data/downloads/
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf --all-sources

# Put conformed trees elsewhere (e.g. external disk)
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo --output-dir /mnt/bigdisk/envision-actionable

# Copy files instead of hardlinking (uses ~2x disk, portable across filesystems)
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo --copy

# Re-run only records that failed on a previous run
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo --rerun-status failed
```

Or use the module entry point:

```bash
python -m envision_eye_actionable --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo
```

## On-disk layout

```
data/actionable/
+-- zenodo/
    +-- _conform_log.json                           # run summary (status per record)
    +-- 4521044/
    |   +-- dataset_structure_description.json      # AI-READI schema descriptor
    |   +-- conform_report.json                     # placements + inventory + stats
    |   +-- _unplaced/                              # files the recipe didn't place
    |   +-- retinal_photography/
    |       +-- 1_healthy_young_raw_good_quality/
    |       |   +-- <image files hardlinked from downloads/>
    |       +-- 2_healthy_young_segmented/
    |       +-- ...
    +-- 16744782/
        +-- dataset_structure_description.json
        +-- conform_report.json
        +-- retinal_photography/
            +-- ...
```

Hardlinks mean the conformed tree does not double disk usage. The downloaded
file and the placed file are the same inode; deleting either keeps the
other alive.

## Agent

Every record is processed by a local Gemma 4 E4B model (via llama-cpp-python).
The agent receives:

- the inventory summary (file-type histogram, modality counts, depth, etc.)
- sampled README text (up to 3 KB)
- sample file paths (first 30 files)
- dataset metadata from the scrape (title, description, keywords)

...and returns a JSON Recipe: a list of glob-to-directory placements plus a
`task_type` (classification, segmentation, detection, or raw).

The model is loaded once per process and cached in memory (~7 GB for the
Q4_K_M quantization). Typical inference takes 30-90 seconds per record on
a 2-core CPU.

### Model setup

```bash
pip install envision-eye-actionable[agent]

# Download the recommended GGUF (~5 GB)
# https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
# Place it somewhere stable, e.g. ~/models/gemma-4-e4b-it-q4.gguf
```

## Modality tagging

The inventory maps file extensions to modality tags that correspond to
`relatedTerm` entries in the AI-READI schema:

| Modality tag          | Triggered by                     | Ontology                          |
|-----------------------|----------------------------------|-----------------------------------|
| `retinal_photography` | .jpg .png .tif .bmp .gif         | MeSH D005654 (Fundus Photography) |
| `retinal_imaging`     | .dcm .dicom                      | NCIT C168215 (Retinal Imaging)    |
| `retinal_oct`         | .fds .fda .e2e .img .oct .vol    | MeSH D041623 (OCT)                |
| `volumetric_imaging`  | .nii .nii.gz .nrrd .mha .mhd     | NCIT C188577                      |
| `tabular_data`        | .csv .tsv .xlsx                  | --                                |
| `derived_data`        | .h5 .hdf5 .mat .npy .npz         | --                                |
| `documentation`       | .pdf .docx .txt .md              | --                                |
| `metadata`            | .json .xml .yaml                 | --                                |

Unreadable vendor formats (.vol, .nce, .nck, .zrx, .ims, .tco) are still
placed into the correct modality directory — we just don't decode them. See
[readers.md](readers.md) for the full list and remediation paths.

## Validation

For every conformed record, the validator:

1. Loads `dataset_structure_description.json`.
2. Checks every `directoryName` exists on disk.
3. Samples one file per top-level directory and tries to open it with the
   appropriate open-source reader (Pillow, tifffile, pydicom, nibabel, h5py,
   mat73/scipy, pynrrd, SimpleITK, pypdf, python-docx, `oct-converter`).

Validation results appear in `_conform_log.json` and on stdout:

```
  [3/83] Conforming zenodo/4521044
    status=ok  confidence=0.70  validation_ok=True
```

## Orphan policy

Files a recipe doesn't place (e.g. stray .DS_Store, unrecognized formats, or
files in unusual locations) land under `_unplaced/`. Nothing is silently
dropped. `conform_report.json.orphan_count` + `orphan_extensions` make the
gap visible for followup.

## What the conformer does NOT do (yet)

- **Decode vendor OCT** — we only verify `oct-converter` imports. Actual
  conversion to DICOM / numpy arrays is a followup once we decide the
  canonical on-disk form (DICOM-wrapped volumes vs. extracted B-scans).
- **Extract labels from PDFs / DOCX** — the readers are installed and text
  extraction works, but label extraction isn't wired into the agent prompt yet.
- **Splits** — no train/val/test split handling yet. Most datasets provide
  splits explicitly; the conformer passes them through as directories but
  doesn't tag them.
- **Deduplicate across sources** — that lives upstream in envision-discovery.
