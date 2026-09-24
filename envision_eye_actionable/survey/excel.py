"""Build the survey workbook from ``survey_results.jsonl``.

Sheets:
    README          method, parameters, model provenance, status counts
    Records         one row per Zenodo record
    Record_Classes  one row per (record, present modality) with its DICOM mapping
    Weblinks        external links of records with no usable image files
    DICOM_Mapping   reference: classifier class -> DICOM attributes and CMDS dirs
    Schema_Fields   which AI-READI / CMDS fields were filled from what
    CMDS_JSON       the two CMDS JSON documents of every record, as text
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
    ("n_files_zenodo", "Files on Zenodo"),
    ("n_files_local", "Files used in place"),
    ("n_files_remote_zip", "Zips sampled remotely"),
    ("n_files_to_download", "Files streamed"),
    ("n_files_not_needed", "Files not fetched (non-image types)"),
    ("n_files_skipped_size", "Files not fetched (over --max-download-gb)"),
    ("download_failures", "Download failures"),
    ("n_archives_listed", "Archives listed"),
    ("n_images_listed", "Images seen in listings"),
    ("n_image_files", "Image files (total, incl. archive members; estimate when sampled remotely; excludes unsampleable members)"),
    ("n_remote_sampled", "Remote members sampled"),
    ("n_remote_fetched", "Remote members fetched"),
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
    ("setfit_label", "SetFit metadata label (info only)"),
    ("setfit_prob_eye_imaging", "SetFit p(eye) (info only)"),
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
    ("remote_fetch_failure_detail", "Remote fetch failure detail"),
    ("remote_requests", "Remote requests"),
    ("size_note", "Size note"),
    ("skipped_size_files", "Files over --max-download-gb (key, size, link)"),
    ("error", "Error"),
    ("results_dir", "Results dir (when several runs were merged)"),
    ("keywords", "Keywords"),
    ("elapsed_s", "Seconds"),
    ("finished_at", "Finished at (UTC)"),
]

WEBLINK_COLUMNS = [
    ("record_id", "Zenodo record id"), ("record_title", "Record title"), ("record_status", "Record status"),
    ("url", "URL"), ("category", "Category"), ("category_detail", "Category detail"),
    ("dataset_likely", "Dataset likely"), ("http_status", "HTTP status"), ("final_url", "Redirected to"),
    ("content_type", "Content-Type"), ("content_length", "Content-Length"), ("error", "Check error"),
    ("origin", "Found in"), ("scraper_type", "Discovery link type"), ("url_raw", "Raw URL (before cleaning)"),
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
                                                        "files; otherwise it names the access_right and that no image "
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


UNFINISHED_STATUSES = ("started", "crashed")   # in progress, or died inside the record


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


def build_workbook(results_path: Path | list[Path], out_path: Path, model_meta: dict | None = None) -> dict:
    """Write the workbook from one results JSONL or several (merged by
    record id, the last one wins; see merge_results). Returns simple stats."""
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    paths = [Path(p) for p in results_path] if isinstance(results_path, (list, tuple)) else [Path(results_path)]
    rows = sorted(merge_results(paths).values(), key=lambda r: int(r["record_id"])
                  if str(r["record_id"]).isdigit() else 0)
    for r in rows:
        backfill(r)
        if r.get("cmds_dir"):
            own = Path(r["_results_path"])
            r["_cmds_folder"] = str(_cmds_folder(r["cmds_dir"], own))
            r["cmds_dir"] = _rel_cmds_dir(r["cmds_dir"], own)
            # Full folder (the results dir resolved, plus cmds/<id>), so rows
            # merged from several results dirs are unambiguous.
            r["cmds_folder"] = _cmds_folder(r["cmds_dir"], own.resolve()).as_posix()
    wb = Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="305496")

    def clean(v):
        v = _fmt(v)
        return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v

    def table(ws, columns: list[tuple[str, str]], data: list[dict], freeze="B2", link_cols=("url",)):
        ws.append([label for _, label in columns])
        for c in ws[1]:
            c.font, c.fill = header_font, header_fill
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for d in data:
            ws.append([clean(d.get(key)) for key, _ in columns])
        for idx, (key, label) in enumerate(columns, 1):
            letter = get_column_letter(idx)
            sample = [len(str(d.get(key) or "")) for d in data[:300]]
            width = max([min(len(label), 28)] + sample)
            ws.column_dimensions[letter].width = max(8, min(width + 2, 60))
            if key in link_cols:
                for cell in ws[letter][1:]:
                    if isinstance(cell.value, str) and cell.value.startswith("http"):
                        cell.hyperlink = cell.value
                        cell.font = Font(color="0563C1", underline="single")
        ws.freeze_panes = freeze
        if data:
            ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(data) + 1}"
        ws.row_dimensions[1].height = 45

    # ---- README
    ws = wb.active
    ws.title = "README"
    status = Counter(r.get("status") for r in rows)
    first = rows[0] if rows else {}
    model_meta = model_meta or {}
    readme = [
        ("EyeACT Zenodo modality survey", ""),
        ("Generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Records", len(rows)),
        ("Results merged from", "; ".join(dict.fromkeys(results_dir_labels(paths).values())) if len(paths) > 1 else "one run"),
        ("Records by status", "; ".join(f"{k}: {v}" for k, v in status.most_common())),
        ("Records with classified images", sum(1 for r in rows if (r.get("n_classified") or 0) > 0)),
        ("Records with an eye modality", sum(1 for r in rows if r.get("present_eye_classes"))),
        ("Records mask-dominated (eye classes review only)", sum(1 for r in rows if r.get("mask_dominated"))),
        ("", ""),
        ("Scope", "Every record of the unfiltered envision-discovery Zenodo scrape. The SetFit metadata "
                  "classifier was NOT used to select records; its label is kept as an informational column."),
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
                   "Archives were listed, not unpacked: only the sampled members were read. Each sampled image was decoded (first frame of multi-page TIFF, middle frame "
                   "of multi-frame DICOM, middle slice of NIfTI/NRRD/MHA), content-cropped and squashed to a "
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
        ("Masks", "Images the pixel check flags as segmentation masks or label maps (in a 128 x 128 "
                  "nearest-neighbour sample: gray with at most 4 levels, or color with at most 8 colors; or, "
                  "for anti-aliased masks and label maps, the 4 most common gray levels or the 8 most common "
                  "colors cover at least 95% of the sample and the values after the most common one cover at "
                  "least 60% of the pixels outside it) get "
                  "the label MASK instead of a "
                  "modality, in the thresholded and the argmax columns alike (n MASK, frac MASK, argmax n MASK, "
                  "argmax frac MASK). Like UNCERTAIN, MASK is not a modality: dominant classes, mean confidence, "
                  "mean probabilities, the eye image fraction and eye class presence are taken over the non-mask "
                  "images, and the dominant class is MASK only when every classified image is a mask. The model's "
                  "own prediction for a mask stays in the per-image predictions file (top, model_label)."),
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
                       "images follows the normal rule whatever its mask share. Rows from runs before survey "
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
        ("Weblinks", "Records with no files, no usable image files, images of no eye modality (status "
                     "no_eye_images) or mostly masks (status mask_dominated): every external link, cleaned, categorised by domain, flagged "
                     "dataset_likely, with an HTTP status check when enabled (429 and 5xx answers are retried "
                     "on the next run, not cached)."),
    ]
    for k, v in readme:
        ws.append([k, v])
    ws["A1"].font = Font(bold=True, size=14)
    for r in ws.iter_rows(min_row=2):
        r[0].font = Font(bold=True)
        r[1].alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 120

    # ---- Records
    table(wb.create_sheet("Records"), RECORD_COLUMNS, rows)

    # ---- Record_Classes
    rc_rows = []
    for r in rows:
        detail = r.get("_class_detail") or {}
        per_class = r.get("dicom_per_class") or {}
        for cls, info in detail.items():
            d = per_class.get(cls) or {}
            dtype, mdir, _ = CLASS_DIRS.get(cls, ("", "", ""))
            rc_rows.append({
                "record_id": r["record_id"], "title": r.get("title"), "class": cls,
                "label": CLASS_LABELS.get(cls), "n_sampled": info.get("count"),
                "frac": r.get(f"frac_{cls}"), "est_files": info.get("est_files"),
                "mean_conf": round(info.get("mean_conf") or 0, 4),
                "cmds_dir": f"{dtype}/{mdir}", **{f"dicom_{k}": v for k, v in d.items()},
            })
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
    table(wb.create_sheet("Record_Classes"), rc_cols, rc_rows, freeze="D2")

    # ---- Weblinks
    wl_rows = []
    for r in rows:
        for link in r.get("weblinks") or []:
            wl_rows.append({"record_id": r["record_id"], "record_title": r.get("title"),
                            "record_status": r.get("status"), **link})
    table(wb.create_sheet("Weblinks"), WEBLINK_COLUMNS, wl_rows, freeze="D2", link_cols=("url", "final_url"))

    # ---- DICOM_Mapping (reference)
    m = mapping()
    dm_rows = []
    for cls in CLASSES:
        s = m["classes"][cls]
        code = lambda c: "" if not c else ("; ".join(code(x) for x in c) if isinstance(c, list)  # noqa: E731
                                           else f"{c.get('CodingSchemeDesignator')} {c.get('CodeValue')} {c.get('CodeMeaning')}")
        dtype, mdir, _ = CLASS_DIRS.get(cls, ("(none)", "", ""))
        dm_rows.append({
            "class": cls, "label": s.get("label"), "Modality": s.get("Modality_0008_0060"),
            "Modality_meaning": s.get("Modality_meaning"), "SOPClassUID": s.get("SOPClassUID_0008_0016"),
            "SOPClassName": s.get("SOPClassName"),
            "SOPClassUID_16bit": "; ".join(v for v in (
                s.get("SOPClassUID_if_16bit"), s.get("SOPClassUID_alternative_if_native_projection")) if v),
            "ImageType": "\\".join(s.get("ImageType_0008_0008") or []),
            "BodyPart": s.get("BodyPartExamined_0018_0015") or "",
            "Anatomic": code(s.get("AnatomicRegionSequence_0008_2218")),
            "Device": code(s.get("AcquisitionDeviceTypeCodeSequence_0022_0015")),
            "Filter": code(s.get("LightPathFilterTypeStackCodeSequence_0022_0017")),
            "Channel": code(s.get("ChannelDescriptionCodeSequence_0022_001A")),
            "Photometric": s.get("expected_PhotometricInterpretation") or "",
            "cmds_dir": f"{dtype}/{mdir}" if mdir else "(not a CMDS directory)",
            "notes": s.get("notes"),
        })
    dm_cols = [("class", "Class"), ("label", "Label"), ("Modality", "Modality (0008,0060)"),
               ("Modality_meaning", "Modality meaning"), ("SOPClassUID", "SOP Class UID"),
               ("SOPClassName", "SOP Class name"), ("SOPClassUID_16bit", "Alternative SOP Class UID"),
               ("ImageType", "Image Type"), ("BodyPart", "Body Part Examined"), ("Anatomic", "Anatomic Region"),
               ("Device", "Acquisition Device Type"), ("Filter", "Light Path Filter"),
               ("Channel", "Channel Description"), ("Photometric", "Expected Photometric Interpretation"),
               ("cmds_dir", "CMDS directory"), ("notes", "Notes")]
    table(wb.create_sheet("DICOM_Mapping"), dm_cols, dm_rows, freeze="B2")

    # ---- Schema_Fields
    prov_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        for fld, src in (r.get("cmds_provenance") or {}).items():
            prov_counts[("dataset_description", fld)][src] += 1
    sf_rows = []
    for schema, fld, req, rule in SCHEMA_FIELDS:
        c = prov_counts.get((schema, fld), Counter())
        if schema == "dataset_description":
            filled = sum(c.values())
        elif fld == "directoryList":
            # Records without an eye modality get an empty directoryList.
            filled = sum(1 for r in rows if r.get("cmds_directories"))
        else:
            filled = sum(1 for r in rows if r.get("cmds_dir"))
        sf_rows.append({"schema": schema, "field": fld, "required": "yes" if req else "no", "rule": rule,
                        "records": filled,
                        "sources": "; ".join(f"{k} ({v})" for k, v in c.most_common())})
    sf_cols = [("schema", "Schema"), ("field", "Field"), ("required", "Required"), ("rule", "Mapping rule"),
               ("records", "Records filled"), ("sources", "Sources used (records)")]
    table(wb.create_sheet("Schema_Fields"), sf_cols, sf_rows, freeze="C2")

    # ---- CMDS_JSON: the two conformant documents per record, as text
    cj_rows = []
    for r in rows:
        if not r.get("cmds_dir"):
            continue
        folder = Path(r["_cmds_folder"])
        docs = {}
        for name in ("dataset_description", "dataset_structure_description"):
            try:
                txt = (folder / f"{name}.json").read_text(encoding="utf-8")
            except OSError:
                txt = ""
            if len(txt) > _CELL_MAX:
                txt = txt[:_CELL_MAX] + "\n... [truncated; see the JSON file]"
            docs[name] = txt
        cj_rows.append({"record_id": r["record_id"], "cmds_folder": r["cmds_folder"],
                        "dd_valid": r.get("dd_valid"), "dsd_valid": r.get("dsd_valid"),
                        "dataset_description": docs["dataset_description"],
                        "dataset_structure_description": docs["dataset_structure_description"]})
    cj_cols = [("record_id", "Zenodo record id"), ("cmds_folder", "CMDS JSON folder (results dir + cmds/<id>)"),
               ("dd_valid", "dataset_description valid"), ("dsd_valid", "dataset_structure_description valid"),
               ("dataset_description", "dataset_description.json"),
               ("dataset_structure_description", "dataset_structure_description.json")]
    table(wb.create_sheet("CMDS_JSON"), cj_cols, cj_rows, freeze="B2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return {"records": len(rows), "weblinks": len(wl_rows), "record_classes": len(rc_rows),
            "status": dict(status), "path": str(out_path)}


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
