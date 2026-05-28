# Zenodo Eye Imaging Dataset Guide

Summary of the 83 Zenodo eye imaging datasets downloaded and conformed by the
Envision pipeline. These datasets were discovered by
[envision-discovery](https://github.com/EyeACT/envision-discovery) and
classified as `EYE_IMAGING` by the
[envision-classifier](https://github.com/EyeACT/envision-classifier) SetFit
model.

## Overview

| Metric | Value |
|--------|-------|
| Total datasets | 83 |
| Total download size (compressed) | 644 GB |
| Total unpacked size | ~730 GB |
| Total files (compressed, on disk) | 421 |
| Total files (after unpacking) | 1,581,174 |
| Avg download size per dataset | ~7.8 GB |
| Avg files per dataset (unpacked) | ~19,050 |
| Smallest dataset | 739 B (292994) |
| Largest dataset | 77 GB (14478210) |
| Conform status: ok | 82 (99%) |
| Conform status: needs review | 1 (1%) |

## File Types (after unpacking all archives)

Breakdown of the 1,581,174 files inside the downloaded archives. Compressed
formats (zip, rar, 7z, gz, multi-part zip fragments) are excluded — these are
the actual data files.

### Images (97.5% of all files)

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.jpg` | 867,447 | 54.9% | JPEG retinal photographs, OCT B-scans |
| `.png` | 445,964 | 28.2% | PNG retinal photos, masks, segmentations |
| `.tiff` | 215,652 | 13.6% | TIFF images (OCT volumes, fundus, histology) |
| `.tif` | 12,225 | 0.8% | TIFF images (fundus, adaptive optics) |
| `.gif` | 11,840 | 0.7% | GIF images (robustness benchmark augmentations) |
| `.ppm` | 2,980 | 0.2% | PPM raw images (vessel segmentation benchmarks) |
| `.jpeg` | 1,490 | 0.1% | JPEG (alternate extension) |
| `.bmp` | 22 | <0.1% | Bitmap images |
| **Subtotal** | **1,557,620** | **98.5%** | |

### Volumetric / Medical Imaging

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.nii.gz` | 1,942 | 0.1% | NIfTI compressed volumes (CT/MR segmentation) |
| `.nrrd` | 1,497 | 0.1% | NRRD volumes (head/neck segmentation) |
| `.mha` | 884 | 0.1% | MetaImage volumes (OCT glaucoma) |
| `.hdr` / `.img` | 864 | 0.1% | Analyze 7.5 format pairs (functional OCT) |
| `.mhd` / `.raw` | 36 | <0.1% | MetaImage header + raw data pairs |
| `.oct` | 6 | <0.1% | Vendor OCT volume files |
| **Subtotal** | **5,229** | **0.3%** | |

### Tabular / Labels

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.txt` | 5,137 | 0.3% | Text files (labels, annotations, READMEs) |
| `.xls` | 2,801 | 0.2% | Legacy Excel spreadsheets |
| `.csv` | 280 | <0.1% | CSV data/label files |
| `.xlsx` | 255 | <0.1% | Excel spreadsheets |
| `.tsv` | 12 | <0.1% | Tab-separated values |
| **Subtotal** | **8,485** | **0.5%** | |

### Scientific / Derived Data

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.tdms` | 1,884 | 0.1% | National Instruments TDMS (cell mechanics) |
| `.mat` | 728 | <0.1% | MATLAB data files |
| `.npy` / `.npz` | 630 | <0.1% | NumPy arrays |
| `.mask` / `.meta` / `.points` | 1,719 | 0.1% | GlottisNet annotation files |
| `.pkl` / `.pickle` | 145 | <0.1% | Python pickle files |
| `.h5` / `.hdf5` | 30 | <0.1% | HDF5 data files |
| `.dat` | 182 | <0.1% | Raw binary data |
| `.p` / `.pt` / `.pth` | 38 | <0.1% | PyTorch model weights |
| `.rdata` | 5 | <0.1% | R data files |
| **Subtotal** | **5,361** | **0.3%** | |

### Video

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.mp4` | 1,161 | 0.1% | MP4 video (endoscopy, glottis) |
| `.avi` | 827 | 0.1% | AVI video (retinal imaging, fundus) |
| `.mov` / `.mpeg` | 4 | <0.1% | Other video formats |
| **Subtotal** | **1,992** | **0.1%** | |

### Documentation / Metadata

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.json` | 755 | <0.1% | Metadata, manifests, configs |
| `.pdf` | 38 | <0.1% | Papers, documentation |
| `.svg` | 47 | <0.1% | Vector figures |
| `.xml` | 18 | <0.1% | XML metadata |
| `.md` | 9 | <0.1% | Markdown documentation |
| `.docx` | 7 | <0.1% | Word documents |
| `.pptx` | 10 | <0.1% | PowerPoint presentations |
| **Subtotal** | **884** | **0.1%** | |

### Code / Config

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.py` | 15 | <0.1% | Python scripts |
| `.m` | 211 | <0.1% | MATLAB scripts |
| `.r` | 10 | <0.1% | R scripts |
| `.ini` | 196 | <0.1% | Configuration files |
| `.java` / `.class` / `.lua` | 3 | <0.1% | Other code |
| **Subtotal** | **435** | <0.1% | |

### Other / Junk

| Extension | Count | % | Description |
|-----------|------:|----:|-------------|
| `.ds_store` | 327 | <0.1% | macOS Finder metadata (junk) |
| `._*` (resource forks) | ~150 | <0.1% | macOS resource fork artifacts (junk) |
| misc | ~100 | <0.1% | Assorted vendor-specific, unlabeled |

## Per-Dataset Breakdown

Each row shows one Zenodo record: its title, download size (compressed files
on disk), unpacked size (after extracting all archives), file count after
unpacking, and the dominant file type.

| ID | Title | DL Size | Unpacked | Files | Primary Type |
|----|-------|--------:|----------:|------:|-------------|
| 292994 | Concept detection scores for the IACC.3 dataset | 739 B | 739 B | 1 | json |
| 4957619 | Development of ML models for diagnosis of glaucoma | 13 KB | 39 KB | 4 | csv |
| 12646034 | Comparing Clinical Viability of Automated Fundus Segmentation | 265 KB | 2.7 MB | 3 | csv |
| 14926762 | FAIR AMD OCT Datasets Paper | 480 KB | 994 KB | 11 | xlsx |
| 14305302 | Fetal Ultrasound Semi-Supervised Cervical Segmentation | 3.5 MB | 3.5 MB | 3 | png |
| 18463827 | Dynamic neurovascular adaptation of the retina (hypoxia) | 1.2 MB | 1.2 MB | 3 | docx / tif |
| 17958965 | Motion enhances camouflage: retinal flicker fusion | 1.6 MB | 6.2 MB | 21 | csv |
| 15524472 | RAS Dataset | 1.6 MB | 12 MB | 157 | nrrd |
| 6506353 | Misclassified images of papilledema severity (DL) | 1.7 MB | 1.7 MB | 3 | png / txt |
| 3813684 | Point-wise correlations: 10-2 Humphrey VF and OCT | 2.0 MB | 3.2 MB | 68 | png / m |
| 16744782 | HRF-Seg+: Multi-Structure Annotated Fundus Dataset | 3.4 MB | 3.8 MB | 273 | png |
| 8108809 | Heart and Clavicle Segmentation - Montgomery Dataset | 4.0 MB | 9.0 MB | 277 | png |
| 3708064 | Type-specific dendritic integration in mouse RGCs | 3.3 MB | 3.3 MB | 7 | pickle |
| 8289533 | VIP Cup 2023: Ophthalmic Biomarker Detection (sample) | 21 MB | 21 MB | 252 | jpeg |
| 4994030 | Human foveal cone photoreceptor topography | 33 MB | 44 MB | 59 | tif |
| 17408396 | Human fovea shape in infantile nystagmus | 50 MB | 50 MB | 51 | tif |
| 7505822 | Retinal Fundus Multi-Disease Image Dataset (RFMiD) 2.0 | 64 MB | 64 MB | 4 | zip (nested) |
| 5592621 | OCTA image quality assessment dataset | 78 MB | 136 MB | 602 | png |
| 4992944 | Human retinal pigment epithelium: in vivo cell morphometry | 84 MB | 198 MB | 1,265 | tif |
| 7678656 | Retinal microvascular network (AOSLO) — topology | 90 MB | 90 MB | 59 | png / tif |
| 15105761 | OCTA-Mosaicking Dataset | 96 MB | 103 MB | 395 | png |
| 7618624 | Automatic Choroid Vascularity Index (OCT) | 114 MB | 136 MB | 538 | jpg |
| 17708044 | Sonogenetic Vision Restoration Dataset (EMC/MscL-G22S) | 131 MB | 137 MB | 129 | tif |
| 14201693 | IMA++: ISIC Multi-Annotator Dermoscopic Segmentation | 174 MB | 352 MB | 22,481 | png |
| 56515 | Retinal status analysis: feature extraction & grading | 181 MB | 189 MB | 44 | bmp / m |
| 13916845 | Periorbital Segmentation and Distance Prediction | 190 MB | 501 MB | 12,208 | jpg / png |
| 4980308 | All-optical recording & stimulation of retinal neurons (mice) | 222 MB | 238 MB | 130 | mat |
| 16983454 | Differential topographic organization & retinal inheritance | 324 MB | 327 MB | 186 | m / mat |
| 4304781 | High-Res Fundus Image Database (Monomodal Registration) | 330 MB | 3.2 GB | 254 | tif |
| 10537424 | DeepEye: DL for Automated Retinal Disease Detection | 347 MB | 347 MB | 201 | png |
| 7957454 | Retina OCT glaucoma | 376 MB | 377 MB | 887 | mha |
| 5790134 | Dataset for RSOS-211108 | 384 MB | 632 MB | 311 | tif / csv |
| 4521044 | Retinal microvascular nEtwork visualized by AOSLO | 380 MB | 380 MB | 189 | tif / png |
| 10839556 | OCTDL: OCT Dataset for Image-Based Deep Learning | 399 MB | 407 MB | 2,066 | jpg |
| 15054026 | OCT images: sodium iodate retinal degeneration + ML models | 701 MB | 801 MB | 665 | tiff |
| 3626854 | Self-supervised retinal thickness prediction (unlabeled OCT) | 838 MB | 956 MB | 4,009 | png / jpeg |
| 4935639 | Functional OCT: abnormal retinal rod function | 844 MB | 5.9 GB | 1,113 | hdr / img |
| 17219542 | IDRiD: Indian Diabetic Retinopathy Image Dataset | 1.0 GB | 1.0 GB | 1,489 | jpg / tif |
| 10913446 | Fiber and vessel dataset for segmentation | 1.0 GB | 1.1 GB | 9,236 | jpg / txt |
| 6476639 | UTHealth Fundus and Synthetic OCT-A (UT-FSOCTA) | 1.1 GB | 1.2 GB | 3,912 | png |
| 6325549 | RIGA+ Dataset: Unsupervised Domain Adaptation (Segmentation) | 1.1 GB | 1.1 GB | 5,939 | tif |
| 4962379 | Rat retinal vasomotion (laser speckle imaging) | 1.1 GB | 1.1 GB | 2 | mat |
| 6635685 | Automatic Processing of Adaptive Optics Flood Illumination | 1.2 GB | 1.2 GB | 853 | png |
| 14580071 | OCTAVE: OCT Annotated Volume Experiment | 1.2 GB | 25.7 GB | 1,240 | tif / json |
| 3832094 | Absolute Retinal Blood Flow (Laser Doppler) — part 2 | 1.4 GB | 1.4 GB | 2,097 | xls |
| 3635402 | Absolute Retinal Blood Flow (Laser Doppler) — part 1 | 1.4 GB | 1.4 GB | 3,175 | png / xls |
| 6582341 | MAP natural scene reconstruction from RGCs (ML models) | 1.5 GB | 1.9 GB | 60 | p / pt / hdf5 |
| 4891308 | Fundus images: study of diabetic retinopathy | 1.5 GB | 1.7 GB | 1,439 | jpg |
| 7769863 | Photodynamic Ocular Drug Delivery (OCT-oriented) | 1.8 GB | 2.2 GB | 410 | tif / xlsx |
| 1410499 | Retinal Blur from Natural Scenes and Eye Shape | 1.9 GB | 2.0 GB | 197 | txt / m |
| 12170637 | OCT Human Kidney Dataset for PRB Guidance | 2.1 GB | 2.2 GB | 30,001 | png |
| 17151869 | SYN-OCT (synthetic OCT images) | 3.3 GB | 3.2 GB | 800,014 | jpg |
| 4931882 | Airway segmentation & centerline extraction (thoracic CT) | 3.3 GB | 5.4 GB | 65 | mhd / raw |
| 15225243 | OCT 3D images | 4.8 GB | 6.5 GB | 901 | tiff |
| 15783866 | 3D Cancer Organoids (Phase Contrast Microscopy) | 4.9 GB | 4.9 GB | 41,881 | png |
| 7442914 | HaN-Seg: Head & Neck OAR CT/MR Segmentation | 4.9 GB | 5.1 GB | 1,348 | nrrd |
| 7432971 | Cell membrane tension (traction force microscopy) | 4.9 GB | 11.3 GB | 3,143 | tdms / tif |
| 5030552 | Orbit Image Analysis: whole slide image analysis tool | 5.0 GB | 45.3 GB | 91 | tiff |
| 5659573 | Computer-aided Veress needle guidance (endoscopic OCT) | 5.4 GB | 5.6 GB | 88,073 | png |
| 17657914 | Flexible corneal neurotechnology (retinal oscillations) | 5.9 GB | 6.3 GB | 126 | svg / xlsx / fif |
| 6938457 | GlottisNetV2: Time-variant training/testing data | 590 MB | 690 MB | 2,866 | mp4 / mask |
| 8238140 | OCT Image Dataset of Radiation Dermatitis | 7.0 GB | 7.0 GB | 4 | zip (nested) |
| 8009107 | Fundus Image Dataset: Joint Segmentation of OD/OC (DG) | 616 MB | 685 MB | 4,343 | tif / png |
| 7113948 | OCT porcine kidney dataset (percutaneous nephrostomy) | 658 MB | 657 MB | 30,001 | jpg |
| 7105232 | OLIVES: Ophthalmic Labels for Visual Eye Semantics | 33.9 GB | 34.0 GB | 10 | xlsx / zip (nested) |
| 10258100 | Probabilistic volumetric speckle suppression in OCT (DL) | 8.2 GB | 8.2 GB | 10 | mat |
| 12659652 | Robustness benchmark: retinal vessel segmentation | 8.2 GB | 9.6 GB | 50,913 | png / gif |
| 17298706 | Multidimensional Data Analysis using SMIAL | 10.9 GB | 11.3 GB | 329 | mat |
| 17359557 | CeraMIRScan: Mid-infrared OCT (Ceramic QA) | 10.5 GB | 10.5 GB | 43,815 | png |
| 7806898 | High-res structural/functional retinal imaging (awake primate) | 14.1 GB | 61.8 GB | 554 | avi / dat / mat |
| 5903672 | MICCAI 2021 FLARE Challenge Dataset | 18.5 GB | 18.7 GB | 773 | nii.gz |
| 8425185 | Wavenumber-dependent DLS-OCT measurements | 18.4 GB | 18.4 GB | 25 | pdf / oct |
| 13759203 | Nonlinear spatial integration: retina defocus detection | 19.7 GB | 41.5 GB | 3,510 | png / npy |
| 8287928 | RVD: Handheld Fundus Video Dataset (Vessel Segmentation) | 26.1 GB | 26.5 GB | 7,070 | png / avi |
| 15476771 | Multi-Institutional Benchmark: Volumetric Segmentation (pt 2) | 29.9 GB | 31.3 GB | 537 | nii.gz |
| 7105232 | OLIVES Dataset (full) | 33.9 GB | 34.0 GB | 10 | zip (nested) |
| 8040573 | VIP Cup 2023: Ophthalmic Biomarker Detection (full) | 34.6 GB | 34.6 GB | 3 | zip (nested) |
| 6425084 | Scanning DLS-OCT: Measurement of Lateral Flow | 36.2 GB | 36.2 GB | 17 | py / pdf |
| 15467847 | Multi-Institutional Benchmark: Volumetric Segmentation (pt 1) | 39.5 GB | 41.2 GB | 635 | nii.gz |
| 5030552 | Orbit Image Analysis (whole slide images) | 42.6 GB | 45.3 GB | 91 | tiff |
| 10057471 | ULS23 Challenge Public Training Dataset Part 5 | 48.3 GB | 48.3 GB | 13 | multi-part zip |
| 15224240 | Mycelium Dataset: Edge-Precise Semantic Segmentation | 56.5 GB | 56.5 GB | 20,868 | jpg |
| 14697783 | CARLA Equirectangular Dataset | 73.4 GB | 78.0 GB | 22,952 | png |
| 14478210 | OCT Atherosclerotic Plaque Morphological Features | 82.1 GB | 84.3 GB | 346,597 | tiff / png |

## Notes

- **Compression ratio**: Most datasets are already compressed (zip containing
  JPEGs), so unpacked size is typically close to download size. Exceptions
  include datasets with highly compressible formats like TIFF or NIfTI, where
  unpacked size can be 2-20x larger.

- **Nested archives**: Some datasets (7505822, 8238140, 8040573, 7105232)
  contain zip files inside zip files. The conformer unpacks recursively, but
  some inner zips use compression methods Python's zipfile doesn't support
  (deflate64) — these show up as still-compressed in the output.

- **Multi-part zip** (10057471): One dataset uses spanned zip archives
  (.z01-.z11). Python's zipfile doesn't support these natively; they require
  p7zip or manual reassembly.

- **SYN-OCT** (17151869): This single dataset accounts for 800,000 of the
  1.58M total files — over half the file count. It contains synthetic OCT
  images for glaucoma classification training.

- **Non-eye datasets**: Some records classified as EYE_IMAGING are tangentially
  related (e.g., cell mechanics, airway segmentation, ceramic OCT). The
  classifier catches these at the metadata level because their descriptions
  mention OCT or retinal imaging techniques. They're still valid for the
  pipeline since the conformer handles any file layout.
