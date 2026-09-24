# envision-eye-actionable

Turns raw downloaded eye-imaging archives into HuggingFace-loadable training
data. Uses a local LLM (Gemma 4 E4B via llama.cpp, runs on CPU) to analyze
each dataset's file layout and produce structured directory trees with
per-modality subdirectories, MeSH/NCIT ontology tags, and validated file
layouts compatible with `datasets.load_dataset(...)` or any ML training loop.

Metadata follows the [AI-READI dataset description schema](https://schema.aireadi.org/v0.1.0/)
(v0.1.0) — each conformed dataset includes a `dataset_structure_description.json`
describing its on-disk layout.

Part of the [EyeACT](https://github.com/EyeACT) (Eye Aging, Cognition, and
Imaging) project by the [FAIR Data Innovations Hub](https://fairdataihub.org)
at the [California Medical Innovations Institute (CalMI2)](https://calmi2.org).

EyeACT aims to make eye imaging datasets across the scientific literature
discoverable, classifiable, and directly usable for ML/AI research. Discovered
datasets are registered on the [Envision Portal](https://envisionportal.org).

## Where this sits in the pipeline

[**envision-discovery**](https://github.com/EyeACT/envision-discovery) handles
the metadata side:

- scrapes 7 repositories (Zenodo, Figshare, Dryad, OSF, DataCite, Kaggle, NEI)
- classifies records as eye-imaging or not
- downloads raw files for EYE_IMAGING records
- exports the dataset *metadata* to the AI-READI schema (`dataset_description.json`) — describing what the dataset *claims* to contain

**envision-eye-actionable** (this repo) handles the data side:

- unpacks every archive in each download (zip, tar, 7z, rar, multi-part zip)
- classifies every file by format and modality (fundus / OCT / OCTA / DICOM / ...)
- uses a local LLM agent to analyze the dataset layout and propose a placement recipe
- hardlinks files into a structured on-disk tree
- emits a **post-materialization** `dataset_structure_description.json` — what we actually laid down on disk, distinct from what envision-discovery's exporter produced
- validates the tree by sample-opening one file per modality with the right reader (pydicom, nibabel, Pillow, oct-converter, ...)

The output trees are structured so HuggingFace loaders work out of the
box: `retinal_photography/<class_name>/*.jpg` is an ImageFolder; paired
`image/`+`mask/` subdirectories are a segmentation dataset; etc.

## Installation

```bash
git clone https://github.com/EyeACT/envision-eye-actionable.git
cd envision-eye-actionable
pip install -e '.[agent]'
```

Requires Python >= 3.10. Installs all open-source readers needed for conform
runs: pydicom, nibabel, h5py, mat73, tifffile, pynrrd, SimpleITK, pypdf,
pdfplumber, python-docx, Pillow, oct-converter, py7zr, rarfile, and
llama-cpp-python (for the Gemma 4 agent). See
[docs/readers.md](docs/readers.md) for the full format matrix.

You also need a Gemma 4 E4B GGUF model file:

```bash
# Recommended: Q4_K_M quantization (~5 GB, 5-15 tok/s on CPU)
# Download from https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
```

## Usage

```bash
# Conform every downloaded zenodo record
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf --source zenodo

# One specific record
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo --source-id 4521044

# Every source directory under ./data/downloads/
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf --all-sources

# Re-run only records that failed
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo --rerun-status failed

# Custom input/output paths (if not running next to an envision-discovery checkout)
envision-conform --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo \
    --downloads-dir /mnt/bigdisk/envision-downloads \
    --output-dir    /mnt/bigdisk/envision-actionable
```

Or use the module entry point:

```bash
python -m envision_eye_actionable --agent-model ~/models/gemma-4-e4b-it-q4.gguf \
    --source zenodo
```

## How it works

```
downloaded record
      |
      v
   unpack    extract every .zip/.tar/.7z/.rar/multi-part zip (recursive)
      |
      v
  inventory  walk tree; tag every file by format/modality/readability;
             collect README / PDF / DOCX / label-hint files
      |
      v
   agent     Gemma 4 E4B analyzes inventory + README + metadata
             and proposes a Recipe(task_type, placements[])
      |
      v
 materialize hardlink (default) or copy source files into target tree
      |
      v
  validate   check dataset_structure_description.json, sample-open one
             file per modality with the right reader
```

Full details in [docs/conforming.md](docs/conforming.md).

## On-disk layout (output)

```
data/actionable/
+-- zenodo/
    +-- _conform_log.json                           # run summary (status per record)
    +-- 4521044/
    |   +-- dataset_structure_description.json      # AI-READI schema descriptor
    |   +-- conform_report.json                     # placements + inventory + stats
    |   +-- _unplaced/                              # orphan files the recipe didn't place
    |   +-- retinal_photography/
    |       +-- 1_healthy_young_raw_good_quality/
    |       |   +-- <files hardlinked from data/downloads/>
    |       +-- 2_healthy_young_segmented/
    |       +-- ...
    +-- 16744782/
        +-- dataset_structure_description.json
        +-- conform_report.json
        +-- retinal_photography/
            +-- ...
```

Hardlinks mean the conformed tree **does not double disk usage** — the
downloaded file and the placed file share an inode.

## Relationship to envision-discovery

By default, this tool expects envision-discovery's layout:

| Directory                         | What we read from it                             |
|-----------------------------------|---------------------------------------------------|
| `./data/downloads/{source}/{id}/` | Downloaded files + per-record `manifest.json`    |
| `./results/{source}_eye_imaging.json` | Classification metadata (optional; enriches agent context) |
| `./data/actionable/{source}/{id}/` | **Output** — conformed trees                     |

If you're not running this alongside an envision-discovery checkout, use
`--downloads-dir`, `--results-dir`, and `--output-dir` to point elsewhere.

## What the conformer does NOT do (yet)

- **Decode** vendor OCT. We only verify `oct-converter` imports; extraction to
  DICOM / numpy is a followup once the canonical on-disk form is decided.
- **Extract labels** from PDFs / DOCX automatically. Readers are installed,
  but label extraction isn't wired into the agent prompt yet.
- **Handle splits** (train/val/test). Most datasets provide splits explicitly
  as subdirectories; the conformer passes them through but doesn't tag them.
- **Cross-source deduplication** — that lives upstream in envision-discovery.

## Results (Zenodo, first run)

Applied to 83 downloaded Zenodo eye imaging datasets (644 GB, 421 files):

| Status | Count | % |
|--------|-------|---|
| ok     | 82    | 99% |
| failed | 1     | 1% |

The Gemma 4 E4B agent (running locally on CPU via llama.cpp, Q4_K_M
quantization) produced validated directory trees for 82 of 83 records.

## Zenodo modality survey (`envision-survey`)

A second, lighter pipeline surveys every record of an envision-discovery
Zenodo scrape. The metadata classifier does not filter records first. The
survey classifies a seeded sample of each record's images, inside archives
too, with an ONNX export of the EyeACT 7-class modality classifier (CFP, IR,
PSC, FAF, OCT, OCTA, NEG) on CPU. It records DICOM-aligned attributes
(Modality, SOP Class, laterality, dimensions, manufacturer). It writes
AI-READI / CMDS `dataset_description.json` and
`dataset_structure_description.json` per record and validates both. Records
without usable images, or without any eye-modality image, get a catalogue of
their external dataset links.
Missing files are streamed and deleted after each record, so the whole
corpus never has to fit on disk. An Excel workbook summarises the run.

```bash
pip install -e '.[survey]'
# Run from the envision-discovery checkout, never from this repo: the
# survey writes results/, data/ and caches relative to the working dir.
cd /path/to/envision-discovery
envision-survey run --model /path/outside/repo/regnety004_synthonly.onnx --limit 3
envision-survey excel --out results/survey/zenodo_modality_survey.xlsx
```

See [docs/survey.md](docs/survey.md).

## Related

- [envision-discovery](https://github.com/EyeACT/envision-discovery) — dataset scraping + classification + download pipeline
- [envision-classifier](https://github.com/EyeACT/envision-classifier) — the SetFit eye-imaging classifier
- [Model weights on HuggingFace](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)
- [Envision Portal](https://envisionportal.org) — searchable catalog of discovered eye imaging datasets
- [AI-READI dataset description schema](https://schema.aireadi.org/v0.1.0/)
- [EyeACT Study](https://eyeactstudy.org) — Eye Aging, Cognition, and Imaging study
- [FAIR Data Innovations Hub](https://fairdataihub.org)

## Citation

If you use this tool in your research, please cite the EyeACT project:

> FAIR Data Innovations Hub, California Medical Innovations Institute (CalMI2).
> *Envision: Eye imaging dataset discovery and curation pipeline.*
> EyeACT Study, [eyeactstudy.org](https://eyeactstudy.org).

## License

MIT, see [LICENSE](LICENSE). Individual dataset licenses vary; check each
dataset before use.

The two AI-READI schema files under
`envision_eye_actionable/survey/resources/schemas/` are not MIT: they are
Copyright (c) 2024 AI-READI Consortium, licensed CC BY 4.0, and copied
unmodified (see `schemas/LICENSE.md`, which ships with every wheel and sdist,
and `survey/resources/README.md`).
