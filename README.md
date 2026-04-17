# envision-eye-actionable

Turns raw downloaded eye-imaging archives into HuggingFace-loadable training
data. Produces ADDF v0.1.0 compliant directory trees with per-modality
subdirectories, MeSH/NCIT ontology tags, and validated file layouts that can
be fed directly to `datasets.load_dataset(...)` or any ML training loop.

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
- exports the dataset *metadata* to ADDF v0.1.0 (`dataset_description.json`) — describing what the dataset *claims* to contain

**envision-eye-actionable** (this repo) handles the data side:

- unpacks every archive in each download (zip, tar, 7z, rar, multi-part zip)
- classifies every file by format and modality (fundus / OCT / OCTA / DICOM / …)
- sniffs common ML-ready layouts (ImageFolder classification, paired images+masks, per-subject eye codes, …)
- hardlinks files into an ADDF-compliant on-disk tree
- emits a **post-materialization** `dataset_structure_description.json` — what we actually laid down on disk, distinct from what envision-discovery's exporter produced
- validates the tree by sample-opening one file per modality with the right reader (pydicom, nibabel, Pillow, oct-converter, …)

The ADDF tree we emit is structured so HuggingFace loaders work out of the
box: `retinal_photography/<class_name>/*.jpg` is an ImageFolder; paired
`image/`+`mask/` subdirectories are a segmentation dataset; etc.

## Installation

```bash
git clone https://github.com/EyeACT/envision-eye-actionable.git
cd envision-eye-actionable
pip install -e .
```

Requires Python ≥ 3.10. Installs all open-source readers needed for conform
runs: pydicom, nibabel, h5py, mat73, tifffile, pynrrd, SimpleITK, pypdf,
pdfplumber, python-docx, Pillow, oct-converter, py7zr, rarfile. See
[docs/readers.md](docs/readers.md) for the full format matrix.

Optional extras:

```bash
# Local LLM agent fallback (Gemma 4 E4B via llama.cpp) for unusual layouts
pip install -e '.[agent]'
```

## Usage

```bash
# Conform every downloaded zenodo record
envision-conform --source zenodo

# One specific record
envision-conform --source zenodo --source-id 4521044

# Every source directory under ./data/downloads/
envision-conform --all-sources

# Enable the Gemma 4 agent fallback for low-confidence layouts
envision-conform --source zenodo --agent-model ~/models/gemma-4-e4b-it-q4.gguf

# Re-run only records that need the agent (after a first pass)
envision-conform --source zenodo \
    --rerun-status agent_needed,failed \
    --agent-model ~/models/gemma-4-e4b-it-q4.gguf

# Custom input/output paths (if not running next to an envision-discovery checkout)
envision-conform --source zenodo \
    --downloads-dir /mnt/bigdisk/envision-downloads \
    --output-dir    /mnt/bigdisk/envision-actionable
```

Or use the module entry point:

```bash
python -m envision_eye_actionable --source zenodo
```

## How it works

```
downloaded record
      │
      ▼
   unpack    extract every .zip/.tar/.7z/.rar/multi-part zip (recursive)
      │
      ▼
  inventory  walk tree; tag every file by format/modality/readability;
             collect README / PDF / DOCX / label-hint files
      │
      ▼
   sniff     rule-based layout detection → Recipe(task_type, placements[])
      │
      ▼  (if confidence < 0.7 AND --agent-model is configured)
   agent     Gemma 4 E4B proposes a Recipe from inventory + README + metadata
      │
      ▼
 materialize hardlink (default) or copy source files into ADDF tree
      │
      ▼
  validate   check dataset_structure_description.json, sample-open one
             file per modality with the right reader
```

Full details in [docs/conforming.md](docs/conforming.md).

## On-disk layout (output)

```
data/actionable/
└── zenodo/
    ├── _conform_log.json                           # run summary (status per record)
    ├── 4521044/
    │   ├── dataset_structure_description.json      # ADDF v0.1.0 descriptor
    │   ├── conform_report.json                     # placements + inventory + stats
    │   ├── _unplaced/                              # orphan files the recipe didn't place
    │   └── retinal_photography/
    │       ├── 1_healthy_young_raw_good_quality/
    │       │   └── <files hardlinked from data/downloads/>
    │       ├── 2_healthy_young_segmented/
    │       └── …
    └── 16744782/
        ├── dataset_structure_description.json
        ├── conform_report.json
        └── retinal_photography/
            └── …
```

Hardlinks mean the conformed tree **does not double disk usage** — the
downloaded file and the ADDF-placed file share an inode.

## Relationship to envision-discovery

By default, this tool expects envision-discovery's layout:

| Directory                         | What we read from it                             |
|-----------------------------------|---------------------------------------------------|
| `./data/downloads/{source}/{id}/` | Downloaded files + per-record `manifest.json`    |
| `./results/{source}_eye_imaging.json` | Classification metadata (optional; enriches agent context) |
| `./data/actionable/{source}/{id}/` | **Output** — ADDF-compliant trees                |

If you're not running this alongside an envision-discovery checkout, use
`--downloads-dir`, `--results-dir`, and `--output-dir` to point elsewhere.

## What the conformer does NOT do (yet)

- **Decode** vendor OCT. We only verify `oct-converter` imports; extraction to
  DICOM / numpy is a followup once the canonical on-disk form is decided.
- **Extract labels** from PDFs / DOCX automatically. Readers are installed,
  but no rule consumes them. This is the first thing the Gemma 4 agent pass
  will unlock.
- **Handle splits** (train/val/test). Most datasets provide splits explicitly
  as subdirectories; the conformer passes them through but doesn't tag them.
- **Cross-source deduplication** — that lives upstream in envision-discovery.

## Results (Zenodo, first run)

Applied to 83 downloaded Zenodo eye imaging datasets (644 GB):

| Pass | ok | agent_needed | failed |
|------|-----|-------------|--------|
| Rule-based only | 16 (19%) | 60 (72%) | 7 (9%) |
| + Gemma 4 E4B agent | **82 (99%)** | 1 (1%) | 0 (0%) |

The agent (running locally on CPU via llama.cpp, Q4_K_M quantization) upgraded
66 of 67 non-ok records to `ok` with validated ADDF trees. Zero failures on the
re-run.

## Related

- [envision-discovery](https://github.com/EyeACT/envision-discovery) — dataset scraping + classification + download pipeline
- [envision-classifier](https://github.com/EyeACT/envision-classifier) — the SetFit eye-imaging classifier
- [Model weights on HuggingFace](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)
- [Envision Portal](https://envisionportal.org) — searchable catalog of discovered eye imaging datasets
- [ADDF schema (AI-READI)](https://schema.aireadi.org/v0.1.0/)
- [EyeACT Study](https://eyeactstudy.org) — Eye Aging, Cognition, and Imaging study
- [FAIR Data Innovations Hub](https://fairdataihub.org)

## Citation

If you use this tool in your research, please cite the EyeACT project:

> FAIR Data Innovations Hub, California Medical Innovations Institute (CalMI2).
> *Envision: Eye imaging dataset discovery and curation pipeline.*
> EyeACT Study, [eyeactstudy.org](https://eyeactstudy.org).

## License

MIT — see [LICENSE](LICENSE). Individual dataset licenses vary; check each
dataset before use.
