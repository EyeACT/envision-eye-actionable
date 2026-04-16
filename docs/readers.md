# File format readers

This document tracks which proprietary eye imaging formats we can decode with
open-source tools versus which ones need vendor SDKs or further effort. The
ADDF Conformer relies on these readers to turn raw archives into canonical
ADDF directory trees.

## Open-source readers we can use today

All of these ship under permissive licenses and can be bundled without friction.

### Standard / open formats

| Format          | Extensions            | Reader                   | License     |
|-----------------|------------------------|---------------------------|-------------|
| DICOM           | `.dcm`, `.dicom`       | [`pydicom`](https://github.com/pydicom/pydicom)       | MIT-style   |
| NIfTI           | `.nii`, `.nii.gz`      | [`nibabel`](https://nipy.org/nibabel/)                | MIT         |
| HDF5            | `.h5`, `.hdf5`         | [`h5py`](https://www.h5py.org/)                       | BSD-3       |
| MATLAB ≤ v7     | `.mat`                 | `scipy.io.loadmat`        | BSD-3       |
| MATLAB v7.3     | `.mat`                 | [`mat73`](https://github.com/skjerns/mat_v7.3)        | BSD-2       |
| NumPy           | `.npy`, `.npz`         | `numpy`                   | BSD-3       |
| Common images   | `.jpg/.png/.bmp/.gif/.tif` | `Pillow` + `tifffile` | HPND / BSD  |
| Multi-page TIFF stacks | `.tif`, `.tiff`  | [`tifffile`](https://pypi.org/project/tifffile/)      | BSD-3       |
| NRRD            | `.nrrd`, `.nhdr`       | [`pynrrd`](https://github.com/mhe/pynrrd)             | MIT         |
| Many medical    | `.mha`, `.mhd`, etc.   | [`SimpleITK`](https://simpleitk.org/)                 | Apache-2.0  |
| PDF text/tables | `.pdf`                 | [`pypdf`](https://pypdf.readthedocs.io/) + [`pdfplumber`](https://github.com/jsvine/pdfplumber) | BSD / MIT |
| DOCX            | `.docx`                | [`python-docx`](https://python-docx.readthedocs.io/)  | MIT         |
| Video           | `.avi`, `.mp4`, `.mov` | [`imageio-ffmpeg`](https://github.com/imageio/imageio-ffmpeg) / `opencv` | BSD / Apache-2 |

### Vendor OCT formats via [`oct-converter`](https://github.com/marksgraham/OCT-Converter) (MIT)

| Format  | Vendor / Device              | Notes                                    |
|---------|-------------------------------|------------------------------------------|
| `.fds`  | Topcon                        | Full support                             |
| `.fda`  | Topcon (newer)                | Experimental for OCT data; fundus works  |
| `.e2e`  | Heidelberg Spectralis         | Full support + contour (layer) extraction |
| `.img`  | Zeiss Cirrus (raw B-scan export) | Full support with de-interlacing option |
| `.oct`  | Bioptigen                     | Full support                             |
| `.OCT`  | Optovue (legacy)              | Full support                             |
| `.dcm`  | Any DICOM-compliant device    | Can also **write** DICOM with correct headers |

> Caveat from the `oct-converter` authors: extracted images are not guaranteed
> to match what the vendor's software renders. Fine for dataset curation /
> training pipelines; treat with caution for clinical use.

## What we do not have open access to

These formats either have no known open reader or only partial coverage.
Listed in rough order of how often they show up in public ophthalmology
datasets.

### 1. Newer Zeiss Cirrus / Forum exports (proprietary containers)

- **Extensions seen:** custom DICOM-wrapped containers, `.ZRX`, `.zcx`, proprietary
  Forum archive formats.
- **Status:** `oct-converter` handles the legacy raw `.img` export, but
  newer firmware and Forum exports use vendor-private containers that aren't
  reverse-engineered.
- **What we need:** Zeiss Cirrus SDK (commercial; access gated by a clinical
  partnership) or vendor-converted DICOM output from the uploader. Practical
  workaround — require datasets to be uploaded as DICOM.

### 2. Nidek RS/Navis-EX raw

- **Extensions seen:** `.nce`, `.nck`, `.ndb`.
- **Status:** no public reader.
- **What we need:** Nidek Navis-EX SDK (commercial). Usable workaround —
  request DICOM export upstream; Nidek supports DICOM export when configured.

### 3. Canon Xephilio / OCT-HS100 proprietary raw

- **Status:** no public reader for the proprietary raw stream. The Canon CX-1
  export pipeline can produce DICOM when configured, but raw device files
  remain closed.
- **What we need:** Canon SDK access (limited; research partnership) or force
  DICOM upstream.

### 4. CSO (Costruzione Strumenti Oftalmici)

- **Devices:** MAIA microperimeter, Antares corneal tomographer, MS-39 AS-OCT.
- **Status:** closed proprietary container; no public reader.
- **What we need:** CSO Phoenix export to DICOM / CSV, or the CSO SDK
  (commercial).

### 5. Heidelberg `.vol` (exported volumes, not `.e2e` raw)

- **Status:** older `.vol` spec is partially documented and some third-party
  readers exist, but post-HEYEX v2 `.vol` files include variant packing that
  breaks older readers. `oct-converter` focuses on `.e2e`; `.vol` coverage
  remains uneven in the wild.
- **What we need:** up-to-date `.vol` spec or Heidelberg HEYEX SDK
  (commercial). Mitigation — detect `.vol` header version and fall back to
  marking the record "needs human" when the version isn't supported.

### 6. Topcon IMAGEnet exports

- **Extensions seen:** `.ims`, `.tco`, `.imj`.
- **Status:** no public reader. `.fds` / `.fda` from the raw device are fine
  via `oct-converter`; IMAGEnet's repackaged exports are a different animal.
- **What we need:** Topcon IMAGEnet SDK (commercial) or require authors to
  upload the device-native `.fds` / `.fda` instead.

### 7. Optovue AngioVue (newer multi-volume archives)

- **Status:** `oct-converter` handles the legacy `.OCT`. Newer AngioVue
  exports bundle multiple volumes plus metadata in proprietary archives that
  need parsing work.
- **What we need:** a small amount of RE on sample files, or vendor docs.

### 8. MATLAB `.mat` with custom classes / toolbox objects

- **Status:** `mat73` and `scipy.io.loadmat` handle numeric arrays and structs
  fine, but custom user-defined classes (e.g., Simulink objects, custom
  toolbox types) round-trip poorly — often returning opaque structs.
- **What we need:** the dataset author's MATLAB class definitions, or a
  MATLAB runtime (commercial) to re-export the data as HDF5/NumPy. Mitigation
  — if the author also published a `.npy` / `.h5` version, prefer that.

### 9. Slit-lamp / anterior-segment vendor videos

- **Extensions seen:** `.sli`, `.slc`, plus proprietary-wrapped MP4s with
  encrypted metadata.
- **Status:** the media stream is usually readable via FFmpeg once you
  strip the wrapper. The timing/metadata layer isn't.
- **What we need:** spec for the wrapper or conversion to plain MP4 + CSV.

### 10. Heidelberg raytracing / SPECTRALIS OCTA recent private packing

- **Status:** partial coverage in `oct-converter` (`.e2e` works for B-scans);
  OCTA flow volumes and retinal layer metadata from recent firmware revisions
  are partial.
- **What we need:** Heidelberg SDK (commercial) for full layer metadata.

## Recommended approach

1. **Ship with everything under "open-source readers"** as hard dependencies.
2. **`oct-converter` is a peer dependency** — add it to `pyproject.toml` and
   wire it into the Conformer's modality detection step. For each record,
   emit a `readers_available.json` noting which files were converted to
   DICOM-equivalent form vs. kept as raw pass-through.
3. **Unreadable vendor formats** get marked `status: "pass_through"` in the
   ADDF output — the files live in the ADDF tree under their modality
   directory, but no decoding is attempted. The `relatedStandard` entry in
   `dataset_structure_description.json` should reference the vendor format
   name (e.g. `heidelberg_spectralis_vol`). Consumers then know they need
   vendor tooling.
4. **Commercial SDK gaps** are tracked in this document. Any time we want to
   unlock one, the options are: (a) reverse-engineer on sample files, (b)
   acquire vendor SDK access through a research collaboration, or (c) push
   DICOM-export as a requirement for EyeACT-curated datasets.

## Action items (SDK / access gaps)

| Gap                              | Impact                                              | Action                                                                 |
|----------------------------------|------------------------------------------------------|-----------------------------------------------------------------------|
| Zeiss Cirrus Forum / newer `.vol` | Blocks Zeiss-hospital datasets                      | Pursue Zeiss research SDK; DICOM fallback in intake requirements      |
| Nidek raw (`.nce`, `.nck`)        | Blocks Nidek-sourced datasets                       | Nidek Navis-EX SDK request; DICOM fallback                            |
| Canon raw                        | Blocks Canon-sourced datasets                       | Canon research SDK request; DICOM fallback                            |
| CSO files                        | Blocks MAIA/MS-39 datasets                          | Require DICOM upload                                                   |
| Heidelberg `.vol` post-HEYEX v2  | Partial `.vol` coverage                             | Version-sniff + fallback; request Heidelberg SDK                       |
| Topcon IMAGEnet (`.ims`, `.tco`) | Affects IMAGEnet-repackaged datasets                | Require device-native `.fds` / `.fda` upload                           |
| Newer Optovue AngioVue bundles   | Affects OCTA datasets                                | Small RE effort on samples                                             |
| MATLAB custom classes            | A few percent of `.mat` files return opaque structs | Prefer `.h5` / `.npy` companions when provided; document the gap       |
