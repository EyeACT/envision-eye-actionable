"""Build the survey workbook from ``survey_results.jsonl``.

Sheets:
    README          method, parameters, model provenance, status counts
    Records         one row per Zenodo record
    Record_Classes  one row per (record, present modality) with its DICOM mapping
    Weblinks        external links of records with no usable image files
    Archive_Probes  one row per archive probed before a download
    DICOM_Mapping   reference: classifier class -> DICOM attributes and CMDS dirs
    Schema_Fields   which AI-READI / CMDS fields were filled from what
    CMDS_JSON       the two CMDS JSON documents of every record, as text (or,
                    for large runs, their file paths)
    Formats         source formats and conversion paths over all records
    Access_Requests eye-relevant restricted and embargoed records, ranked by
                    the discovery SetFit probability (only with an
                    enrich-access supplement, see below)
    Model_Training_Datasets
                    the classifier's training and evaluation sources, one row
                    per (model_version, source, class), copied from the CSV
                    given as --training-sources (only then)

Supplements: a results dir holding ``access_results.jsonl`` (written by
``envision-survey enrich-access``) is a supplement, not a run. Its rows are
laid over the survey rows of the same record id, but only their access and
CMDS fields (``access_*``, ``cmds_*``, ``dd_*``, ``dsd_*``; see
SUPPLEMENT_PREFIXES): the survey's status and classification columns are
never replaced. Among several supplements the last dir given wins. Its CMDS
folders are resolved against the supplement's own dir. Supplement rows of
records not in any results file only reach the Access_Requests sheet.

The workbook is written in openpyxl's write-only mode from an index of the
results JSONL (one row in memory at a time), so it scales to the full
30,000-record pull.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .cmds import CLASS_DIRS
from .constants import (CLASS_LABELS, CLASSES, MASK, MASK_DOMINATED_FRAC, MASK_DOMINATED_MIN_NON_MASK, UNCERTAIN,
                        dominant_of)
from .dicom_map import ir_device_fields, mapping, modality_review
from .results import load_results

RECORD_COLUMNS: list[tuple[str, str]] = [
    ("record_id", "Zenodo record id"),
    ("doi", "DOI"),
    ("title", "Title"),
    ("url", "Landing page"),
    ("status", "Survey status"),
    ("license", "License (scrape)"),
    ("access_right", "Zenodo access_right"),
    ("size_mb", "Size MB (scrape)"),
    ("sampling_mode", "Sampling mode (local, remote_zip, download)"),
    ("sampling_pass", "Sampling pass (single, triage, deep)"),
    ("triage_eye_fraction", "Triage eye fraction (non-mask, thresholded)"),
    ("n_triage_classified", "Images classified on the triage pass"),
    ("deep_pass_reason", "Deep pass reason"),
    ("n_files_zenodo", "Files on Zenodo"),
    ("n_files_local", "Files used in place"),
    ("n_files_remote_zip", "Zips sampled remotely"),
    ("n_files_to_download", "Files streamed"),
    ("n_files_not_needed", "Files not fetched (non-image types)"),
    ("n_files_skipped_size", "Files not fetched (over --max-download-gb)"),
    ("download_failures", "Download failures"),
    ("n_archives_probed", "Archives probed before download"),
    ("n_archive_probe_images", "Probed archives with image members (downloaded)"),
    ("n_archive_probe_no_images", "Probed archives without image members (not downloaded, listing catalogued)"),
    ("n_archive_probe_unknown", "Probed archives undecided (downloaded as before)"),
    ("archive_probe_bytes", "Archive probe bytes read"),
    ("archive_probe_requests", "Archive probe requests"),
    ("archive_probe_bytes_avoided", "Download bytes avoided by the archive probe"),
    ("n_series_files_not_fetched", "Large top-level files of a homogeneous series not fetched (a seeded few were)"),
    ("split_join_detail", "Split archive sets joined"),
    ("n_archives_listed", "Archives listed"),
    ("n_images_listed", "Images seen in listings"),
    ("n_image_files", "Image files (total, incl. archive members; estimate when sampled remotely; excludes unsampleable members)"),
    ("member_ext_counts", "Archive members by extension (top 25; local, remote and probed archives)"),
    ("formats_classified", "Source formats of the classified images"),
    ("conversion_counts", "Conversion paths of the classified images (digits masked)"),
    ("formats_unread", "Source formats of the files that gave no image"),
    ("n_pixel_files_unread_by_reason", "Pixel-bearing files that gave no image, by reason"),
    ("n_blank", "Blank (constant) frames, not classified"),
    ("n_mask_like_veto", "Mask-like images kept as NEG or UNCERTAIN (model argmax NEG: graphics)"),
    ("mask_exempt_class", "Eye class exempting the record from mask_dominated"),
    ("n_remote_sampled", "Remote members sampled"),
    ("n_remote_fetched", "Remote members fetched"),
    ("n_remote_reused", "Remote members of the deep sample reused from the triage pass"),
    ("remote_fetch_failures", "Remote fetch failures"),
    ("n_remote_nested_fetched", "Remote nested archives fetched"),
    ("remote_nested_bytes", "Remote nested archive bytes"),
    ("remote_nested_stop", "Remote nested fetching stopped"),
    ("remote_listing_failures", "Remote zip listings failed"),
    ("n_members_skipped_oversize", "Remote members never sampled (over --remote-max-member-mb)"),
    ("n_images_unsampleable", "Unsampleable remote members (over --remote-max-member-mb; unclassified, not in Image files or est. files)"),
    ("fraction_scope_note", "Scope of the class fractions (when members were unsampleable)"),
    ("n_remote_over_budget", "Remote sampled members not fetched (--remote-record-budget-gb reached)"),
    ("n_members_zero_size", "Remote members left out (size 0)"),
    ("remote_bytes_fetched", "Remote bytes fetched (members + nested archives, failed transfers included)"),
    ("remote_bytes_fetched_ok", "Remote bytes of successful fetches (listed sizes)"),
    ("n_sampled", "Images sampled"),
    ("n_classified", "Images classified"),
    ("n_classified_non_mask", "Images classified, masks excluded"),
    ("class_weighting", "Class weighting (uniform, or stratified by source)"),
    ("n_classified_by_source", "Images classified per source"),
    ("n_load_errors", "Sampled images not decodable"),
    ("dominant_class", "Dominant class"),
    ("argmax_dominant_class", "Dominant class (argmax, no threshold)"),
    ("dominant_eye_class", "Dominant eye modality"),
    ("present_eye_classes", "Eye modalities present"),
    ("mask_dominated", f"Mask-dominated (frac MASK >= {MASK_DOMINATED_FRAC:g} and under "
                       f"{MASK_DOMINATED_MIN_NON_MASK} non-mask images)"),
    ("review_eye_classes", "Eye modalities seen in a mask-dominated record (review only, not present)"),
    ("eye_image_fraction", "Eye image fraction (of non-mask images; status ok needs >= --min-eye-fraction)"),
    ("modality_low_trust", "Eye modality low trust (review)"),
    ("modality_review_flags", "Eye modality review flags"),
    ("mean_confidence", "Mean top-1 confidence"),
    *[(f"n_{c}", f"n {c}") for c in CLASSES + [UNCERTAIN, MASK]],
    *[(f"argmax_{c}", f"argmax n {c}") for c in CLASSES + [MASK]],
    *[(f"frac_{c}", f"frac {c}") for c in CLASSES + [UNCERTAIN, MASK]],
    *[(f"argmax_frac_{c}", f"argmax frac {c}") for c in CLASSES + [MASK]],
    *[(f"mean_prob_{c}", f"mean p({c})") for c in CLASSES],
    ("frac_IR_non_mask", "frac IR of non-mask images"),
    ("argmax_frac_IR_non_mask", "argmax frac IR of non-mask images"),
    ("ir_device_hint", "IR device hint (IR dominant or >= 20% of non-mask images)"),
    ("ir_non_spectralis_candidate", "IR non-Spectralis candidate"),
    ("ir_Manufacturer", "Maker of the IR-classified images"),
    ("ir_Manufacturer_source", "IR maker source (dicom_header, exif)"),
    ("setfit_label", "Discovery SetFit metadata label (informational only; NOT used to select records)"),
    ("setfit_prob_eye_imaging", "Discovery SetFit p(eye imaging) (informational only; NOT used to select records)"),
    ("dicom_mapping_status", "DICOM mapping status"),
    ("dicom_header_attributes_used", "Attributes read from DICOM headers"),
    ("dicom_Modality", "Modality (0008,0060)"),
    ("dicom_Modalities_present", "Modalities of present classes"),
    ("dicom_SOPClassUID", "SOP Class UID (0008,0016)"),
    ("dicom_SOPClassName", "SOP Class name"),
    ("dicom_ImageType", "Image Type (0008,0008)"),
    ("dicom_BodyPartExamined", "Body Part Examined (0018,0015)"),
    ("dicom_AnatomicRegion_code", "Anatomic Region (0008,2218)"),
    ("dicom_AcquisitionDeviceType_code", "Acquisition Device Type (0022,0015)"),
    ("dicom_LightPathFilter_code", "Light Path Filter (0022,0017)"),
    ("dicom_ChannelDescription_code", "Channel Description (0022,001A)"),
    ("dicom_OphthalmicImageType_code", "Ophthalmic Image Type (0022,1615)"),
    ("dicom_IlluminationType_code", "Illumination Type (0022,0016)"),
    ("dicom_ImageLaterality_dist", "Image Laterality (0020,0062) distribution"),
    ("laterality_R", "Laterality R"), ("laterality_L", "Laterality L"),
    ("laterality_B", "Laterality B"), ("laterality_U", "Laterality U"),
    ("laterality_conflicts", "Laterality conflicts"),
    ("laterality_source", "Laterality source (files per source)"),
    ("dicom_Rows_range", "Rows (0028,0010) range"),
    ("dicom_Columns_range", "Columns (0028,0011) range"),
    ("dicom_PhotometricInterpretation_set", "Photometric Interpretation (0028,0004)"),
    ("dicom_SamplesPerPixel_set", "Samples per Pixel (0028,0002)"),
    ("dicom_BitsAllocated_set", "Bits Allocated (0028,0100)"),
    ("dicom_BitsStored_set", "Bits Stored (0028,0101)"),
    ("dicom_HighBit_set", "High Bit (0028,0102)"),
    ("dicom_pixel_depth_note", "Pixel depth conformance note"),
    ("dicom_NumberOfFrames_max", "Number of Frames (0028,0008) max"),
    ("multiframe_files", "Multi-frame files (sample)"),
    ("dicom_PixelSpacing_mm", "Pixel Spacing (0028,0030) mm"),
    ("dicom_LossyImageCompression", "Lossy Image Compression (0028,2110)"),
    ("dicom_LossyImageCompressionMethod", "Lossy Image Compression Method (0028,2114)"),
    ("dicom_ImageComments", "Image Comments (0020,4000) (TIFF ImageDescription)"),
    ("rgb_is_gray_frac", "RGB files that are gray"),
    ("mask_like_frac", "Mask-like share of decoded sampled images, unweighted (gray <= 4 levels or "
                       "color <= 8 colors, or dominated by that few values; labelled MASK; weighted share: "
                       "frac MASK)"),
    ("dicom_Manufacturer", "Manufacturer (0008,0070)"),
    ("dicom_Manufacturer_source", "Manufacturer source"),
    ("dicom_ManufacturerModelName", "Manufacturer Model Name (0008,1090)"),
    ("dicom_SoftwareVersions", "Software Versions (0018,1020)"),
    ("dicom_AcquisitionDateTime_range", "Acquisition DateTime (0008,002A) range"),
    ("dicom_HorizontalFieldOfView", "Horizontal Field of View (0022,000C) (DICOM headers)"),
    ("dicom_IlluminationWaveLength", "Illumination Wave Length (0022,0055) (DICOM headers)"),
    ("dicom_SeriesDescription", "Series Description (0008,103E) (DICOM headers)"),
    ("manufacturer_hint_text", "Manufacturer hints (text, names, extensions)"),
    ("manufacturer_hint_source", "Manufacturer hint sources"),
    ("n_real_dicom_sampled", "Real DICOM files sampled"),
    ("dicom_classes_sampled", "Classes with sampled DICOM files"),
    ("dicom_header_Modality_set", "Header Modality values"),
    ("dicom_header_SOPClassUID_set", "Header SOP Class UIDs"),
    ("dicom_qc_modality_agreement", "QC header vs classifier Modality"),
    ("cmds_resourceTypeValue", "CMDS resourceTypeValue"),
    ("cmds_directories", "CMDS directories (est. files)"),
    ("cmds_creators", "Creators"),
    ("cmds_publicationYear", "Publication year"),
    ("cmds_version", "Version"),
    ("cmds_accessType", "CMDS accessType"),
    ("cmds_rights", "Rights"),
    ("cmds_managingOrganization", "Managing organization (derived)"),
    ("dd_identifier", "dataset_description identifier"),
    ("dd_publisher", "dataset_description publisher"),
    ("dd_contributors", "dataset_description contributors"),
    ("dd_dates", "dataset_description dates"),
    ("dd_language", "dataset_description language"),
    ("dd_subjects", "dataset_description subjects"),
    ("dd_relatedIdentifiers", "dataset_description relatedIdentifiers"),
    ("dd_funding", "dataset_description fundingReference"),
    ("dd_size", "dataset_description size"),
    ("dd_format", "dataset_description format"),
    ("dd_description", "dataset_description description (truncated)"),
    ("cmds_dir", "CMDS JSON folder (relative to the results dir)"),
    ("cmds_folder", "CMDS JSON folder (results dir + cmds/<id>)"),
    ("cmds_note", "CMDS note (metadata-only documents)"),
    ("dd_valid", "dataset_description valid"),
    ("dd_errors", "dataset_description errors"),
    ("dsd_valid", "dataset_structure_description valid"),
    ("dsd_errors", "dataset_structure_description errors"),
    ("cmds_placeholder_fields", "Placeholder fields (not from Zenodo)"),
    ("cmds_derived_fields", "Derived fields"),
    ("cmds_classifier_fields", "Fields set from the classifier"),
    ("n_weblinks", "Weblinks catalogued"),
    ("n_weblinks_dataset_likely", "Weblinks dataset-likely"),
    ("image_ext_counts", "Image extensions"),
    ("files_ext_counts", "Zenodo file extensions"),
    ("file_kind_counts", "File kinds (incl. archive members)"),
    ("load_error_top", "Decode errors (top)"),
    ("walk_errors", "Archive errors"),
    ("walk_error_detail", "Archive error detail"),
    ("download_failure_detail", "Download failure detail"),
    ("remote_listing_detail", "Remote zip listings"),
    ("archive_probe_detail", "Archive probe detail (per archive: outcome, method, members, images, bytes, requests)"),
    ("remote_fetch_failure_detail", "Remote fetch failure detail"),
    ("remote_requests", "Remote requests"),
    ("size_note", "Size note"),
    ("skipped_size_files", "Files over --max-download-gb (key, size, link)"),
    ("error", "Error"),
    ("results_dir", "Results dir (when several runs were merged)"),
    ("keywords", "Keywords"),
    ("elapsed_s", "Seconds (classification)"),
    ("fetch_s", "Seconds (fetch, pipeline)"),
    ("spool_bytes", "Bytes fetched into the spool (pipeline)"),
    ("deep_fetch_error", "Deep pass fetch error"),
    ("n_eye_images", "Eye-class images classified (thresholded, MASK excluded)"),
    ("kept", "Fetched files kept (keep dir)"),
    ("kept_reason", "Why not kept"),
    ("kept_path", "Keep dir of the record"),
    ("kept_files", "Files kept"),
    ("kept_bytes", "Bytes kept"),
    ("kept_local_files", "Local originals (not moved; paths in KEPT.json)"),
    ("finished_at", "Finished at (UTC)"),
    ("access_status", "Zenodo access status (enrich-access)"),
    ("access_files", "Zenodo files access (enrich-access)"),
    ("access_embargo_until", "Embargo until (enrich-access)"),
    ("access_allow_user_requests", "Owner accepts access requests (enrich-access; empty: not exposed)"),
    ("access_request_note", "How to get the files (enrich-access)"),
    ("access_request_url", "Request access on (landing page)"),
    ("access_license_use", "License use: train, evaluate_only_nc_nd, no_license, check_license (enrich-access)"),
    ("access_eye_relevant", "Eye-relevant by SetFit or keywords (enrich-access)"),
    ("access_keyword_terms", "Eye keywords matched (enrich-access)"),
    ("access_checked_at", "Access checked at (UTC)"),
]

# Fields a supplement row (enrich-access) may set on a survey row. Nothing
# else: status and classification columns always come from the survey.
SUPPLEMENT_PREFIXES = ("access_", "cmds_", "dd_", "dsd_")
SUPPLEMENT_NAME = "access_results.jsonl"

ACCESS_REQUEST_COLUMNS = [
    ("rank", "Rank (by SetFit p(eye imaging))"), ("record_id", "Zenodo record id"), ("title", "Title"),
    ("access_setfit_label", "Discovery SetFit label"), ("access_setfit_prob", "Discovery SetFit p(eye imaging)"),
    ("access_keyword_terms", "Eye keywords matched"), ("access_license", "License"),
    ("access_license_use", "License use (train, evaluate_only_nc_nd, no_license, check_license)"),
    ("access_resource_type", "Resource type"), ("access_status", "Zenodo access status"),
    ("access_files", "Files access"), ("access_embargo_active", "Embargo active"),
    ("access_embargo_until", "Embargo until"), ("access_embargo_reason", "Embargo reason"),
    ("access_allow_user_requests", "Owner accepts requests from Zenodo users (empty: not exposed)"),
    ("access_allow_guest_requests", "Owner accepts guest requests"),
    ("access_accept_conditions_text", "Owner's conditions"), ("access_request_note", "How to get the files"),
    ("url", "Request access on (landing page)"), ("access_request_api", "Access request API link"),
    ("access_survey_status", "Survey status"), ("cmds_folder", "CMDS JSON folder"),
    ("access_checked_at", "Checked at (UTC)"),
]

WEBLINK_COLUMNS = [
    ("record_id", "Zenodo record id"), ("record_title", "Record title"), ("record_status", "Record status"),
    ("url", "URL"), ("category", "Category"), ("category_detail", "Category detail"),
    ("dataset_likely", "Dataset likely"), ("http_status", "HTTP status"), ("final_url", "Redirected to"),
    ("content_type", "Content-Type"), ("content_length", "Content-Length"), ("error", "Check error"),
    ("origin", "Found in"), ("scraper_type", "Discovery link type"), ("url_raw", "Raw URL (before cleaning)"),
]

ARCHIVE_PROBE_COLUMNS = [
    ("record_id", "Zenodo record id"), ("record_status", "Record status"), ("archive", "Archive"),
    ("size", "Size (bytes)"), ("format", "Format"),
    ("outcome", "Outcome (images: downloaded; no_images: not downloaded; unknown: downloaded as before)"),
    ("method", "Probe method"), ("complete", "Whole listing read"), ("n_members", "Members listed"),
    ("n_images", "Image, DICOM or volume members"), ("n_nested", "Nested archives"),
    ("n_noext", "Extensionless members (possible DICOM)"), ("bytes", "Probe bytes"),
    ("requests", "Probe requests"), ("hit", "First image-bearing member"), ("note", "Note"),
]

SCHEMA_FIELDS = [
    # (schema, field, required, rule)
    ("dataset_description", "schema", True, "constant https://schema.aireadi.org/v0.1.0/dataset_description.json"),
    ("dataset_description", "identifier", True, "Zenodo DOI (identifierType DOI); landing URL if no DOI"),
    ("dataset_description", "title", True, "DataCite titles (titleType mapped to enum), else legacy title"),
    ("dataset_description", "version", True, "Zenodo version; else version index + 1 (derived); else '1' (placeholder)"),
    ("dataset_description", "alternateIdentifier", False, "landing URL + DataCite alternateIdentifiers (oai -> Other)"),
    ("dataset_description", "creator", True, "DataCite creators: nameType, ORCID as https://orcid.org/ URI, ROR affiliations"),
    ("dataset_description", "contributor", False, "DataCite contributors (types outside enum -> Other)"),
    ("dataset_description", "publicationYear", True, "DataCite publicationYear, else publication_date[:4]"),
    ("dataset_description", "date", False, "DataCite dates with their dateType kept (Issued, Updated, Available, ...; only values outside the schema list become Other)"),
    ("dataset_description", "resourceType", True, "classifier: 'Eye Imaging Dataset: <modalities>' or 'Dataset'; general = Dataset"),
    ("dataset_description", "datasetDeIdentLevel", True, "DERIVED, a project default for every record: "
                                                        "DeIdentificationApplied with deIdentDirect TRUE (direct "
                                                        "identifiers taken as removed); deIdentHIPAA, deIdentDates, "
                                                        "deIdentNonarr and deIdentKAnon FALSE because the depositor "
                                                        "does not report the methods (false = not reported, not "
                                                        "confirmed absent). deIdentDetails says 'Publicly released on "
                                                        "Zenodo as image files' only for open records with image "
                                                        "files; for restricted, embargoed and closed records it says "
                                                        "access is restricted by the depositor and de-identification "
                                                        "is not reported (enrich-access adds the embargo end); "
                                                        "otherwise it names the access_right and that no image "
                                                        "file was read"),
    ("dataset_description", "datasetConsent", True, "PLACEHOLDER, not reported by source: "
                                                   "ConsentSpecifiedNotElsewhereCategorised (least assertive schema "
                                                   "value) with every boolean false (no restriction reported) and a "
                                                   "'Not reported by source' consentsDetails"),
    ("dataset_description", "description", False, "DataCite descriptions, HTML stripped (Abstract/Methods/Other)"),
    ("dataset_description", "language", False, "Zenodo language, ISO 639-3 mapped to 639-1"),
    ("dataset_description", "relatedIdentifier", False, "DataCite relatedIdentifiers incl. IsVersionOf concept DOI"),
    ("dataset_description", "subject", False, "DataCite subjects (subjectScheme, classificationCode or valueUri "
                                             "kept as subjectIdentifier) + Zenodo keywords + classifier modality "
                                             "terms (NCIT/MeSH where coded); one entry per subject text"),
    ("dataset_description", "managingOrganization", True, "DERIVED first creator affiliation, with its identifier and declared scheme (ROR, ISNI, GRID), else Zenodo"),
    ("dataset_description", "accessType", True, "access_right: open -> PublicOnScreenAccessAndDownload, restricted -> CaseByCaseDownload, closed -> NonPublicAccessNoDetails, embargoed -> Other"),
    ("dataset_description", "accessDetails", True, "DERIVED text + landing URL; url and urlLastChecked omitted (url pattern rejects URLs containing 's'; nothing was checked)"),
    ("dataset_description", "rights", True, "DataCite rightsList (name, URI, id); ids mapped to canonical SPDX (cc-by-4.0 -> CC-BY-4.0) and labelled SPDX only then, else scheme 'Zenodo license id'; bare openAccess entry dropped; 'Not specified' when absent"),
    ("dataset_description", "publisher", True, "DataCite publisher (Zenodo)"),
    ("dataset_description", "fundingReference", False, "DataCite fundingReferences"),
    ("dataset_description", "size", False, "Zenodo file sizes and counts, survey image count"),
    ("dataset_description", "format", False, "MIME types of Zenodo file and archive member extensions"),
    ("dataset_structure_description", "schema", True, "constant https://schema.aireadi.org/v0.1.0/dataset_structure_description.json (v0.1.1 keeps the v0.1.0 const)"),
    ("dataset_structure_description", "directoryList", True, "classifier: dataType dirs retinal_photography / retinal_oct / retinal_octa with modality subdirs cfp, ir, uwf_pseudocolor, faf, structural_oct, enface; numberOfFiles estimated from the sample; NCIT/MeSH relatedTerm; CMDS relatedStandard, plus DICOM on a dataType dir only when its own classes had sampled DICOM files"),
    ("dataset_structure_description", "metadataFileList", True, "dataset_description.json"),
]


def _fmt(v):
    """Cell value: scalars as is, dicts as 'k: v; ...', lists joined."""
    if v is None:
        return None
    if isinstance(v, list) and v and all(isinstance(x, dict) and "key" in x for x in v):
        # file lists: "key (size) link"
        return "; ".join(f"{x['key']} ({_human(x.get('size'))}) {x.get('url') or ''}".strip() for x in v)[:32000]
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, str):
        return v[:32000]
    return v


_CELL_MAX = 32000  # Excel cells hold at most 32767 characters


def _human(n) -> str:
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "?"
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{int(n)} B"


def backfill(row: dict) -> dict:
    """Columns added after a row was written (older JSONL lines): argmax
    fractions from the argmax counts, the IR device fields, the eye modality
    review flags, sampling mode."""
    n = row.get("n_classified") or 0
    for c in CLASSES:
        if f"argmax_frac_{c}" not in row and f"argmax_{c}" in row:
            row[f"argmax_frac_{c}"] = round((row.get(f"argmax_{c}") or 0) / n, 4) if n else None
    if "argmax_dominant_class" not in row and n and any(f"argmax_{c}" in row for c in CLASSES):
        row["argmax_dominant_class"] = dominant_of({c: row.get(f"argmax_{c}") or 0 for c in CLASSES}, CLASSES)
    if "ir_device_hint" not in row or ("present_eye_classes" in row and not row["present_eye_classes"]):
        row.update(ir_device_fields(row))     # IR device fields follow the eye decision
    if "modality_review_flags" not in row:
        row.update(modality_review(row))
    if "sampling_mode" not in row:
        row["sampling_mode"] = ("local" if row.get("n_files_local") else "") + \
            ("+download" if row.get("n_files_to_download") else "")
        row["sampling_mode"] = row["sampling_mode"].strip("+") or "none"
    if "n_images_listed" not in row and "n_image_files" in row:
        row["n_images_listed"] = row["n_image_files"]
    return row


def _cmds_folder(cmds_dir: str, results_path: Path) -> Path:
    """cmds_dir is relative to the results dir (rows from older runs: absolute)."""
    p = Path(cmds_dir)
    return p if p.is_absolute() else Path(results_path).parent / p


def _rel_cmds_dir(cmds_dir: str, results_path: Path) -> str:
    p = Path(cmds_dir)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(Path(results_path).parent.resolve()).as_posix()
    except ValueError:
        parts = p.parts
        return "/".join(parts[parts.index("cmds"):]) if "cmds" in parts else p.as_posix()


UNFINISHED_STATUSES = ("started", "crashed", "awaiting_deep")   # in progress, died inside, or waiting for its deep pass


def results_dir_labels(results_paths: list[Path]) -> dict[str, str]:
    """{str(path): label} naming each results file's dir by the shortest
    trailing part of its resolved path that no other dir given shares
    (``survey_smoke/results`` next to ``survey_smoke2/results``). Never an
    absolute path: the drive or root is not part of a label."""
    parents = {str(p): Path(p).resolve().parent.parts for p in results_paths}
    distinct = set(parents.values())
    labels = {}
    for key, parts in parents.items():
        others = [o for o in distinct if o != parts]
        body = parts[1:] if len(parts) > 1 else parts      # drop the anchor ('/' or 'C:\\')
        label = "/".join(body)
        for k in range(1, len(body) + 1):
            tail = body[-k:]
            if all(tuple(o[-k:]) != tuple(tail) for o in others):
                label = "/".join(tail)
                break
        labels[key] = label
    return labels


def merge_results(results_paths: list[Path]) -> dict[str, dict]:
    """Rows of several results JSONL files merged by record id: the last
    line per id within a file, and the last file given across files, wins,
    except that an unfinished row (status started or crashed) never replaces
    a finished one from an earlier file. Each row remembers its file
    (``_results_path``) so its CMDS folder is resolved against its own
    results dir, and with several files names its dir in ``results_dir``
    (see results_dir_labels)."""
    merged: dict[str, dict] = {}
    labels = results_dir_labels(results_paths) if len(results_paths) > 1 else {}
    for path in results_paths:
        for rid, row in load_results(path).items():
            row["_results_path"] = str(path)
            if labels:
                row["results_dir"] = labels[str(path)]
            old = merged.get(rid)
            if (old is not None and row.get("status") in UNFINISHED_STATUSES
                    and old.get("status") not in UNFINISHED_STATUSES):
                continue                    # keep the finished row
            merged.pop(rid, None)           # re-insert: dict order follows the winning file
            merged[rid] = row
    return merged


def merge_index(results_paths: list[Path]) -> dict[str, tuple[int, int, str | None]]:
    """The merge of merge_results as an index, without keeping any row in
    memory: {record id: (file index, byte offset of the winning line,
    status)}. Same rules: last line per id within a file, the last file
    across files, an unfinished row never replaces a finished one."""
    merged: dict[str, tuple[int, int, str | None]] = {}
    for fi, path in enumerate(results_paths):
        last: dict[str, tuple[int, str | None]] = {}
        if not Path(path).exists():
            continue
        with open(path, "rb") as fp:
            while True:
                off = fp.tell()
                line = fp.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rid = row.get("record_id")
                if rid:
                    last[str(rid)] = (off, row.get("status"))
        for rid, (off, st) in last.items():
            old = merged.get(rid)
            if old is not None and st in UNFINISHED_STATUSES and old[2] not in UNFINISHED_STATUSES:
                continue
            merged.pop(rid, None)
            merged[rid] = (fi, off, st)
    return merged


def split_results_dirs(dirs: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """(survey results files, supplement files, dirs with neither) of the
    --results-dir values, in the order given. A dir may hold both."""
    source, supplements, empty = [], [], []
    for d in dirs:
        res, sup = Path(d) / "survey_results.jsonl", Path(d) / SUPPLEMENT_NAME
        if res.is_file():
            source.append(res)
        if sup.is_file():
            supplements.append(sup)
        if not res.is_file() and not sup.is_file():
            empty.append(Path(d))
    return source, supplements, empty


def load_supplements(paths: list[Path] | None) -> dict[str, dict]:
    """Supplement rows (enrich-access) by record id: the last line per id in
    a file and the last file given win. Each row remembers its file
    (``_supplement_path``)."""
    out: dict[str, dict] = {}
    for path in paths or []:
        for rid, row in load_results(path).items():
            row["_supplement_path"] = str(path)
            out[rid] = row
    return out


def overlay_supplement(row: dict, supp: dict | None) -> dict:
    """Lay a supplement row over a survey row: only the fields named by
    SUPPLEMENT_PREFIXES are set (a supplement CMDS folder then resolves
    against the supplement's dir); every other field stays the survey's."""
    if not supp:
        return row
    for k, v in supp.items():
        if k.startswith(SUPPLEMENT_PREFIXES):
            row[k] = v
    if supp.get("cmds_dir"):
        row["cmds_dir"] = supp["cmds_dir"]
        row["_cmds_base"] = supp["_supplement_path"]
    return row


def _iter_rows(paths: list[Path], index: dict[str, tuple[int, int, str | None]], order: list[str],
               supplements: dict[str, dict] | None = None):
    """Rows of ``index`` in ``order``, read one at a time by offset, with
    _results_path (and results_dir when several files), the backfill and the
    supplement fields (overlay_supplement)."""
    labels = results_dir_labels(paths) if len(paths) > 1 else {}
    fps = {}
    try:
        for rid in order:
            fi, off, _ = index[rid]
            fp = fps.get(fi)
            if fp is None:
                fp = fps[fi] = open(paths[fi], "rb")
            fp.seek(off)
            row = json.loads(fp.readline())
            own = Path(paths[fi])
            row["_results_path"] = str(own)
            if labels:
                row["results_dir"] = labels[str(own)]
            backfill(row)
            overlay_supplement(row, (supplements or {}).get(rid))
            if row.get("cmds_dir"):
                base = Path(row.pop("_cmds_base", None) or own)
                row["_cmds_folder"] = str(_cmds_folder(row["cmds_dir"], base))
                row["cmds_dir"] = _rel_cmds_dir(row["cmds_dir"], base)
                row["cmds_folder"] = _cmds_folder(row["cmds_dir"], base.resolve()).as_posix()
            yield row
    finally:
        for fp in fps.values():
            fp.close()


def access_request_rows(supplements: dict[str, dict]) -> list[dict]:
    """Access_Requests rows: eye-relevant records whose files are not public,
    highest SetFit p(eye imaging) first (none last), then by record id."""
    rows = []
    for rid, s in supplements.items():
        if not s.get("access_eye_relevant") or s.get("access_files_public_now"):
            continue
        d = {k: v for k, v in s.items() if not k.startswith("_")}
        d["record_id"] = rid
        d["title"] = s.get("access_title")
        d["url"] = s.get("access_request_url")
        if s.get("cmds_dir"):
            d["cmds_folder"] = _cmds_folder(s["cmds_dir"], Path(s["_supplement_path"]).resolve()).as_posix()
        rows.append(d)

    def key(d):
        p = d.get("access_setfit_prob")
        p = p if isinstance(p, (int, float)) else -1.0
        rid = str(d["record_id"])
        return (-p, int(rid) if rid.isdigit() else 0, rid)
    rows.sort(key=key)
    for i, d in enumerate(rows, 1):
        d["rank"] = i
    return rows


EMBED_CMDS_MAX_ROWS = 2000     # CMDS_JSON embeds the documents up to this many records (auto)

TRAINING_SHEET = "Model_Training_Datasets"
# Columns of a training sources CSV that the README summary reads; any other
# column is copied to the sheet as is, in the CSV's order.
TRAINING_REQUIRED = ("model_version",)


def load_training_sources(path: Path) -> tuple[list[str], list[dict]]:
    """(columns, rows) of a training sources CSV: one row per (model_version,
    source, class). Needs a model_version column so that a retrain can append
    its own rows next to the earlier model's. Blank lines are skipped."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [c for c in (reader.fieldnames or []) if c]
        missing = [c for c in TRAINING_REQUIRED if c not in columns]
        if missing:
            raise ValueError(f"{path}: training sources CSV has no {', '.join(missing)} column")
        rows = [r for r in reader if any((v or "").strip() for k, v in r.items() if k)]
    return columns, rows


def _int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def training_summary(rows: list[dict]) -> str:
    """README text: per model_version, sources, classes, training and test
    images, synthetic and evaluation-only sources, sources whose license the
    CSV marks as not cleared for training."""
    parts = []
    for mv in dict.fromkeys(r.get("model_version") or "(none)" for r in rows):
        rs = [r for r in rows if (r.get("model_version") or "(none)") == mv]
        classes = sorted({r.get("class") for r in rs if r.get("class")})
        text = f"{mv}: {len(rs)} source rows"
        if classes:
            text += f" over {len(classes)} classes ({', '.join(classes)})"
        for col, label in (("student_train_n", "training images"), ("student_test_n", "test images")):
            if any(col in r for r in rs):
                text += f", {sum(_int(r.get(col)) for r in rs)} {label}"
        synth = [r.get("source_id") or "?" for r in rs if str(r.get("synthetic", "")).lower() in ("yes", "true", "1")]
        if synth:
            text += f"; synthetic: {', '.join(synth)}"
        ev = [r.get("source_id") or "?" for r in rs if (r.get("role") or "").startswith("eval_only")]
        if ev:
            text += f"; evaluation only (never trained on): {', '.join(ev)}"
        flagged = [r.get("source_id") or "?" for r in rs
                   if (r.get("training_policy") or "allowed") != "allowed" and _int(r.get("student_train_n")) > 0]
        if flagged:
            text += f"; trained on sources not cleared by license policy: {', '.join(flagged)}"
        parts.append(text)
    return ". ".join(parts)


def build_workbook(results_path: Path | list[Path], out_path: Path, model_meta: dict | None = None,
                   cmds_json: str = "auto", training_sources: Path | None = None,
                   supplements: list[Path] | None = None) -> dict:
    """Write the workbook from one results JSONL or several (merged by
    record id, the last one wins; see merge_results). Streamed: rows are
    read one at a time by offset and written with openpyxl's write-only
    mode, so 30,000 records need little memory. ``cmds_json``: embed (the
    two CMDS documents as text), paths (their file paths and validity), or
    auto (embed up to EMBED_CMDS_MAX_ROWS records). ``training_sources``: a
    CSV of the model's training and evaluation sources (see
    load_training_sources), copied to the Model_Training_Datasets sheet and
    summarised in the README. Returns simple stats."""
    training = load_training_sources(Path(training_sources)) if training_sources else None
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    paths = [Path(p) for p in results_path] if isinstance(results_path, (list, tuple)) else [Path(results_path)]
    supp = load_supplements([Path(p) for p in supplements or []])
    index = merge_index(paths)
    order = sorted(index, key=lambda r: (0, int(r)) if str(r).isdigit() else (1, 0))
    embed = cmds_json == "embed" or (cmds_json == "auto" and len(order) <= EMBED_CMDS_MAX_ROWS)

    # ---- pass 1: statistics, column widths (first 300 rows), format totals
    stats = {"n": len(order), "status": Counter(), "passes": Counter(), "n_classified": 0, "n_eye": 0,
             "n_mask_dominated": 0, "n_kept": 0, "kept_bytes": 0, "first": {}, "n_cmds": 0, "n_dirs": 0,
             "n_supplement": len(supp), "n_supplement_matched": 0}
    prov_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    fmt = {k: (Counter(), Counter()) for k in ("formats_classified", "formats_unread", "conversion_counts")}
    counts = {"weblinks": 0, "record_classes": 0, "archive_probes": 0}
    widths: dict[str, dict[str, int]] = defaultdict(dict)
    for i, r in enumerate(_iter_rows(paths, index, order, supp)):
        if i == 0:
            stats["first"] = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        stats["status"][r.get("status")] += 1
        stats["n_supplement_matched"] += r["record_id"] in supp
        stats["passes"][r.get("sampling_pass") or "(none)"] += 1
        stats["n_classified"] += (r.get("n_classified") or 0) > 0
        stats["n_eye"] += bool(r.get("present_eye_classes"))
        stats["n_mask_dominated"] += bool(r.get("mask_dominated"))
        stats["n_kept"] += bool(r.get("kept"))
        stats["kept_bytes"] += int(r.get("kept_bytes") or 0) if r.get("kept") else 0
        stats["n_cmds"] += bool(r.get("cmds_dir"))
        stats["n_dirs"] += bool(r.get("cmds_directories"))
        for fld, src in (r.get("cmds_provenance") or {}).items():
            prov_counts[("dataset_description", fld)][src] += 1
        for key, (n_img, n_rec) in fmt.items():
            for f, n in (r.get(key) or {}).items():
                n_img[f] += n or 0
                n_rec[f] += 1
        counts["weblinks"] += len(r.get("weblinks") or [])
        counts["record_classes"] += len(r.get("_class_detail") or {})
        counts["archive_probes"] += len(r.get("archive_probes") or [])
        if i < 300:
            for key, _ in RECORD_COLUMNS:
                v = r.get(key)
                if v is not None:
                    widths["Records"][key] = max(widths["Records"].get(key, 0), len(str(_fmt(v) or "")))

    wb = Workbook(write_only=True)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="305496")
    link_font = Font(color="0563C1", underline="single")

    def clean(v):
        v = _fmt(v)
        return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v

    class Table:
        def __init__(self, title, columns, n_rows, freeze="B2", link_cols=("url",), width_hint=None):
            self.ws = wb.create_sheet(title)
            self.columns, self.link_cols = columns, set(link_cols)
            for idx, (key, label) in enumerate(columns, 1):
                w = max(min(len(label), 28), (width_hint or {}).get(key, 0))
                self.ws.column_dimensions[get_column_letter(idx)].width = max(8, min(w + 2, 60))
            self.ws.freeze_panes = freeze
            if n_rows:
                self.ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{n_rows + 1}"
            head = []
            for _, label in columns:
                c = WriteOnlyCell(self.ws, value=label)
                c.font, c.fill = header_font, header_fill
                c.alignment = Alignment(wrap_text=True, vertical="top")
                head.append(c)
            self.ws.row_dimensions[1].height = 45
            self.ws.append(head)

        def add(self, d: dict):
            out = []
            for key, _ in self.columns:
                v = clean(d.get(key))
                if key in self.link_cols and isinstance(v, str) and v.startswith("http"):
                    c = WriteOnlyCell(self.ws, value=v)
                    if len(v) <= 2000:                        # Excel's hyperlink length limit
                        c.hyperlink = v
                        c.font = link_font
                    out.append(c)
                else:
                    out.append(v)
            self.ws.append(out)

    # ---- README
    ws = wb.create_sheet("README")
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 120
    for i, (k, v) in enumerate(_readme_rows(stats, paths, model_meta, training)):
        a = WriteOnlyCell(ws, value=clean(k))
        a.font = Font(bold=True, size=14) if i == 0 else Font(bold=True)
        b = WriteOnlyCell(ws, value=clean(v))
        b.alignment = Alignment(wrap_text=True, vertical="top")
        ws.append([a, b])

    # ---- per-record sheets, written in one pass
    records = Table("Records", RECORD_COLUMNS, stats["n"], width_hint=widths["Records"])
    rc_cols = [("record_id", "Zenodo record id"), ("title", "Title"), ("class", "Class"), ("label", "Label"),
               ("n_sampled", "Sampled images"), ("frac", "Fraction of sample"), ("est_files", "Estimated files"),
               ("mean_conf", "Mean confidence"), ("cmds_dir", "CMDS directory"),
               ("dicom_Modality", "Modality"), ("dicom_SOPClassUID", "SOP Class UID"),
               ("dicom_SOPClassName", "SOP Class name"), ("dicom_ImageType", "Image Type"),
               ("dicom_BodyPartExamined", "Body Part Examined"), ("dicom_AnatomicRegion", "Anatomic Region"),
               ("dicom_AcquisitionDeviceType", "Acquisition Device Type"),
               ("dicom_LightPathFilter", "Light Path Filter"), ("dicom_ChannelDescription", "Channel Description"),
               ("dicom_SamplesPerPixelUsed", "Samples Per Pixel Used"),
               ("dicom_OphthalmicImageType", "Ophthalmic Image Type"),
               ("dicom_IlluminationType", "Illumination Type")]
    rclasses = Table("Record_Classes", rc_cols, counts["record_classes"], freeze="D2")
    weblinks = Table("Weblinks", WEBLINK_COLUMNS, counts["weblinks"], freeze="D2", link_cols=("url", "final_url"))
    probes = Table("Archive_Probes", ARCHIVE_PROBE_COLUMNS, counts["archive_probes"], freeze="C2")
    if embed:
        cj_cols = [("record_id", "Zenodo record id"), ("cmds_folder", "CMDS JSON folder (results dir + cmds/<id>)"),
                   ("dd_valid", "dataset_description valid"), ("dsd_valid", "dataset_structure_description valid"),
                   ("dataset_description", "dataset_description.json"),
                   ("dataset_structure_description", "dataset_structure_description.json")]
    else:
        cj_cols = [("record_id", "Zenodo record id"), ("cmds_folder", "CMDS JSON folder (results dir + cmds/<id>)"),
                   ("dd_valid", "dataset_description valid"), ("dsd_valid", "dataset_structure_description valid"),
                   ("dd_file", "dataset_description.json file"),
                   ("dsd_file", "dataset_structure_description.json file"),
                   ("dd_bytes", "dataset_description.json bytes"),
                   ("dsd_bytes", "dataset_structure_description.json bytes")]
    cmds_sheet = None                   # created after the reference sheets (sheet order kept below)
    cj_pending: list[dict] = []
    for r in _iter_rows(paths, index, order, supp):
        records.add(r)
        detail = r.get("_class_detail") or {}
        per_class = r.get("dicom_per_class") or {}
        for cls, info in detail.items():
            d = per_class.get(cls) or {}
            dtype, mdir, _ = CLASS_DIRS.get(cls, ("", "", ""))
            rclasses.add({
                "record_id": r["record_id"], "title": r.get("title"), "class": cls,
                "label": CLASS_LABELS.get(cls), "n_sampled": info.get("count"),
                "frac": r.get(f"frac_{cls}"), "est_files": info.get("est_files"),
                "mean_conf": round(info.get("mean_conf") or 0, 4),
                "cmds_dir": f"{dtype}/{mdir}", **{f"dicom_{k}": v for k, v in d.items()},
            })
        for link in r.get("weblinks") or []:
            weblinks.add({"record_id": r["record_id"], "record_title": r.get("title"),
                          "record_status": r.get("status"), **link})
        for a in r.get("archive_probes") or []:
            probes.add({"record_id": r["record_id"], "record_status": r.get("status"),
                        **{("archive" if k == "key" else k): v for k, v in a.items()}})
        if r.get("cmds_dir"):
            cj_pending.append({"record_id": r["record_id"], "cmds_folder": r["cmds_folder"],
                               "_cmds_folder": r["_cmds_folder"], "dd_valid": r.get("dd_valid"),
                               "dsd_valid": r.get("dsd_valid")})

    # ---- DICOM_Mapping (reference)
    m = mapping()
    dm_cols = [("class", "Class"), ("label", "Label"), ("Modality", "Modality (0008,0060)"),
               ("Modality_meaning", "Modality meaning"), ("SOPClassUID", "SOP Class UID"),
               ("SOPClassName", "SOP Class name"), ("SOPClassUID_16bit", "Alternative SOP Class UID"),
               ("ImageType", "Image Type"), ("BodyPart", "Body Part Examined"), ("Anatomic", "Anatomic Region"),
               ("Device", "Acquisition Device Type"), ("Filter", "Light Path Filter"),
               ("Channel", "Channel Description"), ("Photometric", "Expected Photometric Interpretation"),
               ("cmds_dir", "CMDS directory"), ("notes", "Notes")]
    dmt = Table("DICOM_Mapping", dm_cols, len(CLASSES))

    def code(c):
        if not c:
            return ""
        if isinstance(c, list):
            return "; ".join(code(x) for x in c)
        return f"{c.get('CodingSchemeDesignator')} {c.get('CodeValue')} {c.get('CodeMeaning')}"

    for cls in CLASSES:
        sp = m["classes"][cls]
        dtype, mdir, _ = CLASS_DIRS.get(cls, ("(none)", "", ""))
        dmt.add({
            "class": cls, "label": sp.get("label"), "Modality": sp.get("Modality_0008_0060"),
            "Modality_meaning": sp.get("Modality_meaning"), "SOPClassUID": sp.get("SOPClassUID_0008_0016"),
            "SOPClassName": sp.get("SOPClassName"),
            "SOPClassUID_16bit": "; ".join(v for v in (
                sp.get("SOPClassUID_if_16bit"), sp.get("SOPClassUID_alternative_if_native_projection")) if v),
            "ImageType": "\\".join(sp.get("ImageType_0008_0008") or []),
            "BodyPart": sp.get("BodyPartExamined_0018_0015") or "",
            "Anatomic": code(sp.get("AnatomicRegionSequence_0008_2218")),
            "Device": code(sp.get("AcquisitionDeviceTypeCodeSequence_0022_0015")),
            "Filter": code(sp.get("LightPathFilterTypeStackCodeSequence_0022_0017")),
            "Channel": code(sp.get("ChannelDescriptionCodeSequence_0022_001A")),
            "Photometric": sp.get("expected_PhotometricInterpretation") or "",
            "cmds_dir": f"{dtype}/{mdir}" if mdir else "(not a CMDS directory)",
            "notes": sp.get("notes"),
        })

    # ---- Schema_Fields
    sf_cols = [("schema", "Schema"), ("field", "Field"), ("required", "Required"), ("rule", "Mapping rule"),
               ("records", "Records filled"), ("sources", "Sources used (records)")]
    sft = Table("Schema_Fields", sf_cols, len(SCHEMA_FIELDS), freeze="C2")
    for schema, fld, req, rule in SCHEMA_FIELDS:
        c = prov_counts.get((schema, fld), Counter())
        if schema == "dataset_description":
            filled = sum(c.values())
        elif fld == "directoryList":
            filled = stats["n_dirs"]       # records without an eye modality get an empty directoryList
        else:
            filled = stats["n_cmds"]
        sft.add({"schema": schema, "field": fld, "required": "yes" if req else "no", "rule": rule,
                 "records": filled, "sources": "; ".join(f"{k} ({v})" for k, v in c.most_common())})

    # ---- CMDS_JSON: the two documents per record, as text or as file paths
    cmds_sheet = Table("CMDS_JSON", cj_cols, len(cj_pending), freeze="B2")
    for d in cj_pending:
        folder = Path(d.pop("_cmds_folder"))
        if embed:
            for name in ("dataset_description", "dataset_structure_description"):
                try:
                    txt = (folder / f"{name}.json").read_text(encoding="utf-8")
                except OSError:
                    txt = ""
                if len(txt) > _CELL_MAX:
                    txt = txt[:_CELL_MAX] + "\n... [truncated; see the JSON file]"
                d[name] = txt
        else:
            for short, name in (("dd", "dataset_description"), ("dsd", "dataset_structure_description")):
                f = folder / f"{name}.json"
                d[f"{short}_file"] = f"{d['cmds_folder']}/{name}.json"
                try:
                    d[f"{short}_bytes"] = f.stat().st_size
                except OSError:
                    d[f"{short}_bytes"] = None
        cmds_sheet.add(d)

    # ---- Formats: source formats and conversion paths over all records
    fo_cols = [("kind", "Table"), ("name", "Format or conversion path"), ("images", "Images (files)"),
               ("records", "Records")]
    fo_rows = []
    for key, label in (("formats_classified", "classified images by source format"),
                       ("formats_unread", "files that gave no image, by source format"),
                       ("conversion_counts", "classified images by conversion path")):
        n_img, n_rec = fmt[key]
        fo_rows += [{"kind": label, "name": f, "images": n, "records": n_rec[f]} for f, n in n_img.most_common()]
    fot = Table("Formats", fo_cols, len(fo_rows), freeze="C2")
    for d in fo_rows:
        fot.add(d)

    # ---- Model_Training_Datasets: the classifier's sources, as given
    if training is not None:
        t_cols, t_rows = training
        tt = Table(TRAINING_SHEET, [(c, c) for c in t_cols], len(t_rows), freeze="C2", link_cols=("url",))
        for d in t_rows:
            tt.add({k: (_int(v) if k.endswith("_n") and str(v).strip().lstrip("-").isdigit() else v)
                    for k, v in d.items() if k})

    # ---- Access_Requests (enrich-access supplements only)
    n_access = 0
    if supp:
        ar_rows = access_request_rows(supp)
        n_access = len(ar_rows)
        art = Table("Access_Requests", ACCESS_REQUEST_COLUMNS, n_access, freeze="D2", link_cols=("url",))
        for d in ar_rows:
            art.add(d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.xlsx")
    wb.save(tmp)
    tmp.replace(out_path)
    return {"records": stats["n"], "weblinks": counts["weblinks"], "record_classes": counts["record_classes"],
            "archive_probes": counts["archive_probes"], "cmds_json": "embedded" if embed else "paths",
            "supplement_rows": stats["n_supplement"], "supplement_rows_matched": stats["n_supplement_matched"],
            "access_requests": n_access,
            "training_sources": len(training[1]) if training is not None else None,
            "status": dict(stats["status"]), "path": str(out_path)}


def _readme_rows(stats: dict, paths: list[Path], model_meta: dict | None,
                 training: tuple[list[str], list[dict]] | None = None) -> list[tuple]:
    status = stats["status"]
    first = stats["first"]
    model_meta = model_meta or {}
    readme = [
        ("EyeACT Zenodo modality survey", ""),
        ("Generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Records", stats["n"]),
        ("Results merged from", "; ".join(dict.fromkeys(results_dir_labels(paths).values())) if len(paths) > 1 else "one run"),
        ("Records by status", "; ".join(f"{k}: {v}" for k, v in status.most_common())),
        ("Records with classified images", stats["n_classified"]),
        ("Records with an eye modality", stats["n_eye"]),
        ("Records mask-dominated (eye classes review only)", stats["n_mask_dominated"]),
        ("Records whose fetched files were kept (keep dir)", stats["n_kept"]),
        ("Bytes kept", _human(stats["kept_bytes"])),
        ("", ""),
        ("Scope", "Every record of the unfiltered envision-discovery Zenodo scrape (the records its eye-imaging "
                  "keyword queries found). The SetFit metadata classifier was NOT used to select, skip or order "
                  "records; its label and p(eye) are kept as informational columns only."),
        ("Records by sampling pass", "; ".join(f"{k}: {v}" for k, v in stats["passes"].most_common()) or "none"),
        ("Two-pass sampling", "Triage: at most --triage-cap (default 50) images from each source (local and "
                              "downloaded files; remote zip members and top-level image files fetched one by "
                              "one) are classified first. Only a record whose triage sample has eye classes "
                              "(thresholded, non-mask) at --min-eye-fraction or more gets the deep sample: the "
                              "rest of the same seeded order, up to --max-images for local and downloaded files "
                              "and --remote-cap for remote members, the triage images kept. 'Sampling pass' "
                              "says which: single (one pass), triage or deep; 'Triage eye fraction' and 'Deep "
                              "pass reason' give the decision. Archives that cannot be read remotely are "
                              "downloaded whole once, on the triage pass."),
        ("Method", "Files already downloaded were read in place (sampling_mode local). A zip that was not on "
                   "disk was not downloaded: its members were listed through Zenodo's container API (or, when "
                   "that listing is capped at about 1000 items, by HTTP range reads of the zip's central "
                   "directory) and only a seeded sample of image members was fetched (remote_zip). Nested "
                   "archives inside a remote zip were fetched whole, at most 500 MB each and "
                   "--remote-nested-gb (default 2 GB) per record. Other "
                   "missing image-bearing files (rar, 7z, tar, gz, top-level images) were downloaded whole to a "
                   "scratch dir and deleted after the record (download), up to --max-download-gb per record, "
                   "smallest files first; files that did not fit are listed, and a record with nothing readable "
                   "gets status skipped_size (file list and weblinks still catalogued). "
                   "Before a non-zip archive was downloaded, its listing was read with a few HTTP range "
                   "requests (archive probe: tar headers, the first 64 MB of a compressed tar, the 7z end "
                   "header, rar block headers, the inner name of x.tif.gz): an archive whose whole listing "
                   "held no image, DICOM, volume or nested archive member was not downloaded (its listing is "
                   "catalogued; a record with nothing else to read gets status no_images_in_archives), and "
                   "an undecided probe kept the download (see the Archive_Probes sheet). "
                   "Archives were listed, not unpacked: only the sampled members were read. Every "
                   "pixel-bearing format was converted to an RGB frame (see 'Format conversion'), "
                   "content-cropped and squashed to a "
                   "256 x 256 square (aspect ratio not kept) exactly as the training images were stored, resized "
                   "to 224 x 224 (the training Resize(224) + CenterCrop(224) on a square) and classified by the "
                   "ONNX export of the model."),
        ("Sampling cap", f"At most {first.get('max_images')} images per record for local and downloaded files, "
                         f"at most {first.get('remote_cap') or 300} per record read only through remote zip "
                         f"sampling (spread over the zips in proportion to their image counts); seeded random "
                         f"samples (seed {first.get('seed')}, per-record RNG keyed by record id). 'Images seen in "
                         "listings' counts every image file listed; 'Image files' is the population used for the "
                         "per-modality estimates (it extrapolates unopened nested archives and the DICOM share of "
                         "extensionless members); 'classified' counts the decoded sample. A record whose images "
                         "come from sources sampled at different rates (local or downloaded files; direct "
                         "members of remote zips; images inside nested archives fetched from remote zips) has "
                         "stratified class weights: each image counts with (images in its source) / (images "
                         "classified from that source), and the frac, argmax frac, mean, dominant and estimate "
                         "columns use the "
                         "weighted counts ('Class weighting' = stratified); the n and argmax n columns stay raw "
                         "sample counts. Ties go to the class listed first in 'Classes'."),
        ("Confidence threshold", f"Top-1 probability below {first.get('threshold')} is counted as UNCERTAIN. "
                                 "The 'argmax' columns count every classified image by its top class, ignoring "
                                 "the threshold."),
        ("Format conversion", "TIFF (BigTIFF, tiled, pyramids, OME, ImageJ, LSM, SVS; 8 to 32 bit and float) "
                              "through tifffile: a pyramid gives its smallest level with a long side of at least "
                              "1024 px, a stack of same-shape pages its middle plane (first channel), and an image "
                              "over the pixel budget a strided decode, one strip or tile at a time (never skipped "
                              "for size). JPEG at a reduced DCT scale, JPEG 2000 at a reduced resolution level, "
                              "PNG, BMP, GIF, WebP, PSD, HEIC through Pillow. SVG: the largest embedded raster "
                              "image, else rendered 512 px wide with every external reference removed. DICOM "
                              "(JPEG lossless, JPEG-LS, JPEG 2000 and RLE transfer syntaxes decoded): middle "
                              "frame. NIfTI, NRRD, MHA and header pairs (MHD + RAW, Analyze HDR + IMG): middle "
                              "slice along the shortest axis. NumPy, HDF5, MAT v5 and v7.3 arrays: an image-shaped "
                              "dataset (both sides at least 64 px, aspect at most 32; names such as image, oct, "
                              "bscan, fundus preferred), middle slice. Microscopy (CZI, LIF, ND2, OIB, MRC): middle "
                              "Z plane, first channel. Vendor OCT (Heidelberg E2E and VOL, Topcon FDS and FDA, "
                              "Thorlabs and Bioptigen OCT): middle B-scan, plus the fundus or SLO image as its own "
                              "prediction (path ending #fundus); Heidelberg SDB is catalogued only. Video: frames "
                              "at 25, 50 and 75% of the duration, probabilities averaged into one prediction. "
                              "Integer data deeper than 8 bits and floats are windowed between the 0.5 and 99.5 "
                              "percentiles (few-level data such as label maps keeps min-max), transparency is "
                              "composited over white, and a constant (blank) frame is not classified (n blank). "
                              "The source format and conversion path of every image are in the predictions file "
                              "and summarised per record. Records with no classified image are no_image_files "
                              "(nothing pixel-bearing listed) or images_unreadable (pixel files listed, none "
                              "decoded; reasons per record). Large top-level images of one homogeneous series "
                              "(same name pattern, 64 MB and more each) are sampled, 3 per series."),
        ("Masks", "Images the pixel check flags as segmentation masks or label maps (in a 128 x 128 "
                  "nearest-neighbour sample: gray with at most 4 levels, or color with at most 8 colors; or, "
                  "for anti-aliased masks and label maps, the 4 most common gray levels or the 8 most common "
                  "colors cover at least 95% of the sample and the values after the most common one cover at "
                  "least 60% of the pixels outside it; single-channel data deeper than 8 bits is tested on its "
                  "native values) AND whose model argmax is an eye class get "
                  "the label MASK instead of a "
                  "modality, in the thresholded and the argmax columns alike (n MASK, frac MASK, argmax n MASK, "
                  "argmax frac MASK). Like UNCERTAIN, MASK is not a modality: dominant classes, mean confidence, "
                  "mean probabilities, the eye image fraction and eye class presence are taken over the non-mask "
                  "images, and the dominant class is MASK only when every classified image is a mask. The model's "
                  "own prediction for a mask stays in the per-image predictions file (top, model_label). A "
                  "flagged image whose argmax is NEG keeps its model label (flat-colour graphics: plots, word "
                  "clouds, logos; mask_like_veto in the predictions file)."),
        ("Eye images", f"A record has eye images (status ok) only when the eye classes (CFP, IR, PSC, FAF, OCT, "
                       f"OCTA; thresholded labels) make up at least --min-eye-fraction "
                       f"({first.get('min_eye_fraction', 0.05)}) of its non-mask classified images; an eye class "
                       "is then listed as present at --min-class-fraction (default 0.01) of those images. "
                       "Otherwise the status is no_eye_images, no eye class is listed and the weblinks are "
                       f"catalogued. A mask-dominated record (frac MASK at least {MASK_DOMINATED_FRAC:g} and "
                       f"fewer than {MASK_DOMINATED_MIN_NON_MASK} non-mask "
                       "classified images) is never ok on the small non-mask remainder: its status is "
                       "mask_dominated, the eye classes the rule finds there are listed only in 'Eye modalities "
                       "seen in a mask-dominated record (review only, not present)', its CMDS structure has no "
                       "modality directories and its weblinks are catalogued. A record with "
                       f"{MASK_DOMINATED_MIN_NON_MASK} or more non-mask "
                       "images follows the normal rule whatever its mask share, and so does a record whose "
                       "non-mask images hold at least 10 images of one eye class other than OCTA with mean "
                       "confidence at least 0.75 (photos next to their masks). Rows from runs before survey "
                       "version 3 used a 1% per-class rule over all images; rows before version 4 have no "
                       "mask_dominated status and an older mask check."),
        ("Remote byte bounds", f"Remote zip members over --remote-max-member-mb "
                               f"({first.get('remote_max_member_mb', 200)} MB) are never sampled (another member is "
                               f"drawn instead where the zip has one); at most --remote-record-budget-gb "
                               f"({first.get('remote_record_budget_gb', 2)} GB) is received per record from remote "
                               "zips, direct members and nested archives together, counting the bytes of failed "
                               "transfers and retries too ('Remote bytes fetched' is that measured count); sampled "
                               "members past it are not fetched (they count as listed, unfetched images). Members "
                               "over the per-member cap are outside the sampling frame: they are left out of 'Image "
                               "files', the class fractions and the estimates, and counted on their own as "
                               "'Unsampleable remote members'; 'Scope of the class fractions' says so on the row, "
                               "because large members can hold different images than the small ones sampled. "
                               "Empty (size 0) members are left out the same way but not counted there ('Remote "
                               "members left out (size 0)')."),
        ("IR device", "For records whose dominant class (thresholded or argmax; not the dominant eye modality) "
                  "is IR or whose IR fraction of the non-mask images (thresholded or argmax; 'frac IR of "
                  "non-mask images') is at least 20%: the manufacturer evidence "
                  "found. Strong: the DICOM header or EXIF maker of the images classified IR, labelled "
                  "'(IR images, ...)'. Weak: record-level text, file name and vendor extension hints, labelled "
                  "'(record-level ...)'. An image counts as IR when its thresholded label or its argmax class "
                  "is IR, as in the trigger. 'IR non-Spectralis candidate' is TRUE when the evidence names a "
                  "maker that is not Heidelberg (Spectralis, HRA, HRT, .e2e, .sdb). When the IR images carry a "
                  "maker (strong evidence), that maker alone decides, so a Spectralis mention in the record text "
                  "(often about its OCT) cannot hide a Topcon or Optos IR device; the weak hints decide only "
                  "when the IR images carry no maker."),
        ("Eye modality review", "The image classifier knows only the six eye modalities and NEG, so non-eye "
                                "grayscale images can land in an eye class. Known false-positive pattern: "
                                "microscopy (cell segmentation, binary masks), X-ray and CT slices and tissue "
                                "OCT predicted as OCTA (or OCT) at a mean confidence near 0.7; such records still "
                                "get schema-valid CMDS documents that name an eye modality. 'Eye modality review "
                                "flags' lists: setfit_not_eye (SetFit p(eye) under 0.2), mask_like (frac MASK, "
                                "the weighted share of classified images that are masks or label maps, at least "
                                "30%) and low_confidence (dominant eye "
                                "class mean top-1 confidence under 0.75). 'Eye modality low trust' is TRUE on "
                                "setfit_not_eye or mask_like. Nothing is removed; the flags are also written into "
                                "the CMDS directoryDescription text. Check low-trust rows before using their "
                                "modality."),
        ("Classes", "; ".join(f"{c} = {CLASS_LABELS[c]}" for c in CLASSES + [UNCERTAIN, MASK])),
        ("Model", f"{first.get('model_model_file')} (arch {first.get('model_arch') or model_meta.get('arch')}, "
                  f"checkpoint {first.get('model_checkpoint') or model_meta.get('checkpoint')})"),
        ("Model checkpoint sha256", first.get("model_checkpoint_sha256") or model_meta.get("checkpoint_sha256") or ""),
        ("Model ONNX sha256", (f"{first.get('model_onnx_sha256')} (checked against the loaded .onnx when the "
                               "run started)") if first.get("model_onnx_sha256")
         else (model_meta.get("onnx_sha256") or "")),
        ("Model ONNX parity", _parity_text(model_meta)),
        ("Model training datasets",
         (f"See the {TRAINING_SHEET} sheet: one row per model version, source and class, with license, "
          "role (train, held-out test source, within-source holdout, synthetic, evaluation only) and image "
          "counts, as supplied by the model's maintainers. " + training_summary(training[1]))
         if training is not None else "not included (pass --training-sources)"),
        ("Estimated file counts", "CMDS numberOfFiles per modality = total image files x class fraction in the "
                                  "sample (exact when every image was classified)."),
        ("DICOM", "DICOM-aligned attributes, not DICOM conformance. dicom_mapping_status: read_from_header = "
                  "Modality, SOP Class and Image Type are the majority header values of sampled DICOM files of the "
                  "dominant class (Body Part, Anatomic Region, Acquisition Device Type and Ophthalmic Image "
                  "Type too when the headers carry them; see 'Attributes read from DICOM headers'); "
                  "mixed = DICOM files were sampled but none of the dominant class, so those values "
                  "come from the class mapping (header values are in the 'Header' columns); "
                  "inferred_from_classifier = no DICOM sampled, values from the class mapping; not_applicable = "
                  "nothing classified. Identifying DICOM tags are never read or exported. Rules: DICOM PS3 2026d, "
                  "see DICOM_Mapping."),
        ("Laterality", "DICOM ImageLaterality (0020,0062), then Laterality (0020,0060), for sampled DICOM files; file "
                       "and directory names (OD/OS/OU, right/left, delimited R/L) for every other image file. The "
                       "source column counts files per source."),
        ("CMDS / AI-READI", "Per record: dataset_description.json and dataset_structure_description.json, validated "
                            "against the bundled Clinical Multimodal Data Structure (CMDS) v0.1.1 schemas "
                            "(https://cmds.aireadi.org). datasetDeIdentLevel and datasetConsent are required but "
                            "not reported by Zenodo, and the schema has no 'not known' option. De-identification, a "
                            "project default for every record: DeIdentificationApplied with deIdentDirect TRUE; the "
                            "other flags are FALSE because the depositor does not report the methods (false = not "
                            "reported, not confirmed absent). deIdentDetails states a public release as image files "
                            "only for open records with image files, and otherwise the access_right and that no "
                            "image file was read. Consent: the "
                            "least assertive allowed value (ConsentSpecifiedNotElsewhereCategorised, every boolean "
                            "false) with a 'Not reported by source' note (see Schema_Fields and the placeholder "
                            "column). The JSON files are in the folder named in "
                            "the 'CMDS JSON folder' columns (relative to the results dir, and in full: the results "
                            "dir plus cmds/<id>, so merged runs are unambiguous) and are also copied as text on the "
                            "CMDS_JSON sheet. Records that ended without a classification (skipped_disk, error, "
                            "crashed) get metadata-only documents from the Zenodo metadata (empty directoryList; "
                            "see the CMDS note column)."),
        ("Weblinks", "Records with no files, no image files, unreadable images, images of no eye modality (status "
                     "no_eye_images) or mostly masks (status mask_dominated): every external link, cleaned, categorised by domain, flagged "
                     "dataset_likely, with an HTTP status check when enabled (429 and 5xx answers are retried "
                     "on the next run, not cached)."),
    ]
    return readme


def _parity_text(meta: dict) -> str:
    """ONNX vs PyTorch parity numbers from the model sidecar (pass --model)."""
    if not meta:
        return "not available (pass --model so the sidecar is read)"
    parts = []
    if meta.get("parity_max_abs_diff") is not None:
        parts.append(f"random input: max logit diff {meta['parity_max_abs_diff']:.2g}")
    for key, label in (("parity_real_images", "real images, training eval transform"),
                       ("parity_survey_preprocess", "real images, survey decoding and preprocessing")):
        p = meta.get(key)
        if isinstance(p, dict) and p.get("n"):
            parts.append(f"{label}: n {p['n']}, max logit diff {p.get('max_logit_diff', 0):.2g}, "
                         f"argmax agreement {p.get('argmax_agree')} of {p['n']}")
        elif isinstance(p, dict) and p.get("skipped"):
            parts.append(f"{label}: not run ({p['skipped']})")
    return "; ".join(parts) or "no parity numbers in the sidecar"


def load_model_meta(model_path: Path | None) -> dict:
    if not model_path:
        return {}
    p = Path(model_path).with_suffix(".json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
