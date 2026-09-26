"""AI-READI / CMDS metadata for one surveyed Zenodo record.

Two files per record, the same pair the rest of EyeACT emits:

* ``dataset_description.json`` (schema.aireadi.org v0.1.0, identical rules in
  CMDS v0.1.1): citation and provenance from the Zenodo record, mostly from
  its DataCite 4.x serialisation, which maps almost field for field.
* ``dataset_structure_description.json`` (CMDS structural schema): the
  classifier's modality composition as dataType/modality directories with
  ontology terms (the only place CMDS can hold modality information).

Both schemas use ``additionalProperties: false`` everywhere, so classifier
numbers (confidence, sample size) go into ``directoryDescription`` text and
the Excel workbook, never into extra keys.

Required fields Zenodo cannot supply get explicit placeholders and are listed
per record (``placeholder_fields``), derived values in ``derived_fields``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources

from .constants import CLASS_LABELS
from .dicom_map import strip_html

DD_SCHEMA_URL = "https://schema.aireadi.org/v0.1.0/dataset_description.json"
# The structure schema's $id is v0.1.1 but its `schema` const is still v0.1.0.
DSD_SCHEMA_URL = "https://schema.aireadi.org/v0.1.0/dataset_structure_description.json"
CMDS_NAME = "Clinical Multimodal Data Structure (CMDS) v0.1.1"
CMDS_URL = "https://cmds.aireadi.org"

_NCIT = ("NCI Thesaurus (NCIT)", "https://ncim.nci.nih.gov/",
         "https://ncit.nci.nih.gov/ncitbrowser/pages/concept_details.jsf?dictionary=NCI%20Thesaurus&code={}")
_MESH = ("Medical Subject Headings (MeSH)", "https://meshb.nlm.nih.gov/",
         "https://meshb.nlm.nih.gov/record/ui?ui={}")

# classifier class -> (CMDS dataType dir, modality dir, directory description)
CLASS_DIRS = {
    "CFP": ("retinal_photography", "cfp", "Color fundus photographs"),
    "IR": ("retinal_photography", "ir", "Infrared reflectance (near-infrared SLO) images"),
    "PSC": ("retinal_photography", "uwf_pseudocolor", "Ultra-widefield pseudocolor images"),
    "FAF": ("retinal_photography", "faf", "Fundus autofluorescence images"),
    "OCT": ("retinal_oct", "structural_oct", "OCT B-scans"),
    "OCTA": ("retinal_octa", "enface", "OCT angiography en face images"),
}
DATATYPE_INFO = {
    "retinal_photography": (
        "This directory contains retinal photography data, which are 2D images. They are also "
        "referred to as fundus photography.",
        [("Eye Fundus Photography", [("C147467", _NCIT)])],
    ),
    "retinal_oct": (
        "This directory contains data collected using optical coherence tomography (OCT), an imaging "
        "method using lasers that is used for mapping subsurface structure.",
        [("Optical Coherence Tomography", [("C20828", _NCIT)]),
         ("Tomography, Optical Coherence", [("D041623", _MESH)])],
    ),
    "retinal_octa": (
        "This directory contains optical coherence tomography angiography (OCTA) data, en face images "
        "of retinal and choroidal blood flow derived from repeated OCT scans.",
        [("Optical Coherence Tomography Angiography", [])],
    ),
}
# Subject terms added to dataset_description for each present class.
CLASS_SUBJECTS = {
    "CFP": ("Eye Fundus Photography", "C147467", _NCIT),
    "OCT": ("Tomography, Optical Coherence", "D041623", _MESH),
    "OCTA": ("Optical Coherence Tomography Angiography", None, None),
    "FAF": ("Fundus autofluorescence", None, None),
    "IR": ("Infrared reflectance imaging", None, None),
    "PSC": ("Ultra-widefield imaging", None, None),
}

ACCESS_MAP = {
    "open": "PublicOnScreenAccessAndDownload",
    "restricted": "CaseByCaseDownload",
    "closed": "NonPublicAccessNoDetails",
    "embargoed": "Other",
}
LICENSE_NAMES = {
    "cc-by-4.0": ("Creative Commons Attribution 4.0 International", "https://creativecommons.org/licenses/by/4.0/legalcode"),
    "cc-by": ("Creative Commons Attribution", None),
    "cc-zero": ("Creative Commons Zero v1.0 Universal", "https://creativecommons.org/publicdomain/zero/1.0/legalcode"),
    "cc0-1.0": ("Creative Commons Zero v1.0 Universal", "https://creativecommons.org/publicdomain/zero/1.0/legalcode"),
    "cc-by-sa-4.0": ("Creative Commons Attribution Share Alike 4.0 International", "https://creativecommons.org/licenses/by-sa/4.0/legalcode"),
    "cc-by-nc-4.0": ("Creative Commons Attribution Non Commercial 4.0 International", "https://creativecommons.org/licenses/by-nc/4.0/legalcode"),
    "cc-by-nc-sa-4.0": ("Creative Commons Attribution Non Commercial Share Alike 4.0 International", "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"),
    "cc-by-nc-nd-4.0": ("Creative Commons Attribution Non Commercial No Derivatives 4.0 International", "https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode"),
    "cc-by-nd-4.0": ("Creative Commons Attribution No Derivatives 4.0 International", "https://creativecommons.org/licenses/by-nd/4.0/legalcode"),
    "mit": ("MIT License", "https://opensource.org/licenses/MIT"),
    "apache-2.0": ("Apache License 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
    "odc-by-1.0": ("Open Data Commons Attribution License v1.0", "https://opendatacommons.org/licenses/by/1-0/"),
}
# Zenodo license ids are lowercase; SPDX identifiers are case-sensitive.
# Only ids listed here are labelled SPDX (canonical spelling); anything else
# keeps its Zenodo value under the scheme "Zenodo license id".
SPDX_IDS = {s.lower(): s for s in (
    "CC-BY-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0", "CC-BY-4.0",
    "CC-BY-SA-2.0", "CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "CC-BY-NC-2.0", "CC-BY-NC-2.5", "CC-BY-NC-3.0", "CC-BY-NC-4.0",
    "CC-BY-NC-SA-2.0", "CC-BY-NC-SA-2.5", "CC-BY-NC-SA-3.0", "CC-BY-NC-SA-4.0",
    "CC-BY-NC-ND-2.0", "CC-BY-NC-ND-2.5", "CC-BY-NC-ND-3.0", "CC-BY-NC-ND-4.0",
    "CC-BY-ND-2.0", "CC-BY-ND-2.5", "CC-BY-ND-3.0", "CC-BY-ND-4.0",
    "CC0-1.0", "CC-PDDC", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-3.0-only", "LGPL-3.0-or-later", "AGPL-3.0-only", "AGPL-3.0-or-later",
    "MPL-2.0", "ODbL-1.0", "ODC-By-1.0", "PDDL-1.0", "Unlicense", "EUPL-1.2",
    "Artistic-2.0", "ISC", "Zlib", "OFL-1.1", "etalab-2.0",
)}
SPDX_IDS.update({"cc-zero": "CC0-1.0", "cc0": "CC0-1.0", "odc-by": "ODC-By-1.0", "odbl": "ODbL-1.0",
                 "gpl-3.0": "GPL-3.0-only", "gpl-2.0": "GPL-2.0-only", "lgpl-3.0": "LGPL-3.0-only",
                 "agpl-3.0": "AGPL-3.0-only"})

# Organisation identifier scheme -> scheme URI (unknown schemes get none).
ORG_SCHEME_URIS = {"ROR": "https://ror.org", "ISNI": "https://isni.org", "GRID": "https://www.grid.ac",
                   "Crossref Funder ID": "https://www.crossref.org/services/funder-registry/"}


def rights_identifier(value: str) -> dict:
    """rightsIdentifier object: canonical SPDX id when known, else the raw id."""
    v = str(value).strip()
    spdx = SPDX_IDS.get(v.lower())
    if spdx:
        return {"rightsIdentifierValue": spdx, "rightsIdentifierScheme": "SPDX",
                "schemeURI": "https://spdx.org/licenses/"}
    return {"rightsIdentifierValue": v, "rightsIdentifierScheme": "Zenodo license id"}


def org_scheme(value: str, declared: str | None) -> tuple[str, str | None]:
    """(scheme, schemeURI) for an organisation identifier."""
    s = (declared or "").strip()
    canon = {k.lower(): k for k in ORG_SCHEME_URIS}.get(s.lower())
    if not canon and not s:
        low = str(value).lower()
        canon = ("ROR" if "ror.org/" in low else "ISNI" if "isni" in low else
                 "GRID" if "grid." in low else None)
    scheme = canon or s or "Other"
    return scheme, ORG_SCHEME_URIS.get(scheme)


LANG3 = {"eng": "en", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es", "por": "pt",
         "ita": "it", "zho": "zh", "chi": "zh", "jpn": "ja", "kor": "ko", "rus": "ru", "nld": "nl",
         "dut": "nl", "pol": "pl", "tur": "tr", "ara": "ar", "hin": "hi", "swe": "sv", "fin": "fi"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".tif": "image/tiff",
        ".tiff": "image/tiff", ".bmp": "image/bmp", ".gif": "image/gif", ".dcm": "image/DICOM",
        ".dicom": "image/DICOM", ".zip": "application/zip", ".tar": "application/x-tar",
        ".tar.gz": "application/gzip", ".tgz": "application/gzip", ".gz": "application/gzip",
        ".7z": "application/x-7z-compressed", ".rar": "application/vnd.rar",
        ".nii": "application/x-nifti", ".nii.gz": "application/x-nifti", ".nrrd": "application/x-nrrd",
        ".csv": "text/csv", ".json": "application/json", ".pdf": "application/pdf",
        ".txt": "text/plain", ".h5": "application/x-hdf5", ".mat": "application/x-matlab-data",
        ".webp": "image/webp", ".jp2": "image/jp2", ".ppm": "image/x-portable-pixmap"}

# datasetDeIdentLevel and datasetConsent are required, but Zenodo does not
# report them and the schema removed ECRIN's "NotKnown" option.
# * De-identification (project decision, applied to every record as a
#   default because the depositor does not report the methods): direct
#   identifiers are taken as removed:
#   DeIdentificationApplied ("some de-identification measures have been
#   applied; details in the boolean fields and/or separate documents") with
#   deIdentDirect true. The specific methods (HIPAA rules, rebased dates,
#   narrative text removed, k-anonymity) are not reported by the depositor, so those
#   flags are false, which here means "not reported", not "confirmed
#   absent"; deIdentDetails says so. NoDeIdentification would claim it was
#   confirmed that nothing was done, and ...PrimaryOutcomesReAssessed claims
#   a re-analysis. The enum and the booleans are the same for every record;
#   only the deIdentDetails text (and the provenance) depends on the record:
#   "Publicly released on Zenodo as image files" only for open records with
#   image files; for restricted, embargoed and closed records, that the
#   depositor restricts access and de-identification is not reported; else a
#   statement of the access right and that no image file was read
#   (deident_level).
# * Consent: the least assertive value the schema accepts,
#   ConsentSpecifiedNotElsewhereCategorised ("a descriptive statement
#   regarding consent is available, but it does not fit into categories
#   1 - 5"), whose statement is the consentsDetails text saying that the
#   source reports nothing. Every other option names a consent level
#   (NoExplicitConsent, NoRestriction, GeneralResearchUse, ...). All
#   restriction booleans are false, which here means "none reported", not
#   "none exist".
NOT_REPORTED = "Not reported by source"
DEIDENT_PUBLIC_RELEASE = {
    "deIdentType": "DeIdentificationApplied",
    "deIdentDirect": True, "deIdentHIPAA": False, "deIdentDates": False,
    "deIdentNonarr": False, "deIdentKAnon": False,
    "deIdentDetails": "Publicly released on Zenodo as image files, so direct identifiers are taken as removed "
                      "(deIdentDirect true). The specific de-identification methods are not reported by the "
                      "depositor: the false flags (deIdentHIPAA, deIdentDates, deIdentNonarr, deIdentKAnon) mean "
                      "not reported, not confirmed absent. Set by EyeACT harvesting; check the depositor's "
                      "documentation.",
}
DEIDENT_PROVENANCE = "derived (public release on Zenodo as image files; methods not reported by the depositor)"


# Records whose files the depositor keeps from the public (Zenodo access_right).
NON_PUBLIC_ACCESS = {"restricted": "Access restricted by the depositor",
                     "embargoed": "Embargoed by the depositor",
                     "closed": "Access closed by the depositor"}


def deident_level(access_right: str, has_images: bool, embargo_until: str | None = None) -> tuple[dict, str]:
    """(datasetDeIdentLevel, provenance) for a record. The enum and the five
    booleans never change; the free text says what is known: a public
    release as image files only for open records with image files; for
    restricted, embargoed and closed records that the depositor restricts
    access and that de-identification is not reported (``embargo_until``,
    when known, names the embargo end); else the access right and that no
    image file was read, with the flags applied as a project default."""
    if access_right == "open" and has_images:
        return dict(DEIDENT_PUBLIC_RELEASE), DEIDENT_PROVENANCE
    if access_right in NON_PUBLIC_ACCESS:
        until = f" until {embargo_until}" if access_right == "embargoed" and embargo_until else ""
        details = (f"{NON_PUBLIC_ACCESS[access_right]} on Zenodo{until} (access_right: {access_right}; files "
                   "not publicly released, no image files read), so de-identification is not reported: nothing "
                   "is known about how these files were de-identified. The enum and flags (deIdentDirect true; "
                   "deIdentHIPAA, deIdentDates, deIdentNonarr, deIdentKAnon false) are only the project default "
                   "that this required field needs, not a statement about these files. Set by EyeACT "
                   "harvesting; ask the depositor when requesting access.")
        return ({**DEIDENT_PUBLIC_RELEASE, "deIdentDetails": details},
                f"derived (Zenodo deposit, access_right {access_right}; access restricted by the depositor, "
                "de-identification not reported)")
    why = ("files not publicly released, no image files read" if access_right != "open"
           else "no image files read")
    details = (f"Deposited on Zenodo (access_right: {access_right or 'unknown'}; {why}). Direct identifiers "
               "are taken as removed (deIdentDirect true) as a project default, because the depositor does not "
               "report the de-identification methods: the false flags (deIdentHIPAA, deIdentDates, "
               "deIdentNonarr, deIdentKAnon) mean not reported, not confirmed absent. Set by EyeACT "
               "harvesting; check the depositor's documentation.")
    return ({**DEIDENT_PUBLIC_RELEASE, "deIdentDetails": details},
            f"derived (Zenodo deposit, access_right {access_right or 'unknown'}; {why}; methods not reported)")
PLACEHOLDER_CONSENT = {
    "consentType": "ConsentSpecifiedNotElsewhereCategorised",
    "consentNoncommercial": False, "consentGeogRestrict": False, "consentResearchType": False,
    "consentGeneticOnly": False, "consentNoMethods": False,
    "consentsDetails": f"{NOT_REPORTED} (Zenodo). Consent terms are unknown: the schema has no 'not known' "
                       "option, so this statement is the least assertive allowed value, and the false booleans "
                       "mean that no restriction was reported, not that none exists. Placeholder set by EyeACT "
                       "harvesting; check the depositor's documentation.",
}


@lru_cache(maxsize=4)
def load_schema(name: str) -> dict:
    """Bundled schema: 'dataset_description' or 'dataset_structure_description'."""
    txt = resources.files(__package__).joinpath(f"resources/schemas/{name}.schema.json").read_text(encoding="utf-8")
    return json.loads(txt)


def _allowed(node: dict) -> frozenset:
    vals = node.get("enum") or [x.get("const") for x in node.get("oneOf", []) if isinstance(x, dict)]
    return frozenset(v for v in vals if v)


@lru_cache(maxsize=32)
def enum_values(schema_name: str, definition: str) -> frozenset:
    """Allowed values of an enum (oneOf const or enum) in the schema.

    ``definition`` is a name under ``definitions`` or a slash path from the
    schema root (e.g. ``properties/date/items/properties/dateType``) for enums
    declared inline. Raises KeyError when the name does not resolve to a
    non-empty enum, so a wrong name can never silently turn every value into
    the default.
    """
    schema = load_schema(schema_name)
    if "/" in definition:
        node = schema
        for part in definition.split("/"):
            node = node[part]
    else:
        node = schema["definitions"][definition]
    vals = _allowed(node)
    if not vals:
        raise KeyError(f"{schema_name}: {definition} is not an enum")
    return vals


DATE_TYPE = "properties/date/items/properties/dateType"


def _enum(value, schema_name, definition, default="Other"):
    return value if value in enum_values(schema_name, definition) else default


def _dedupe(items: list) -> list:
    out, seen = [], set()
    for it in items:
        key = json.dumps(it, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _term(value, code_scheme):
    code, scheme = code_scheme
    name, scheme_uri, value_uri = scheme
    return {"relatedTermClassificationCode": code, "relatedTermScheme": name,
            "relatedTermSchemeURI": scheme_uri, "relatedTermValueURI": value_uri.format(code)}


# ---------------------------------------------------------------------------
# dataset_description
# ---------------------------------------------------------------------------
def build_dataset_description(scrape: dict, legacy: dict | None, datacite: dict | None,
                              summary: dict) -> tuple[dict, dict]:
    """Return (document, provenance) where provenance maps field -> source.

    summary: {present_classes, image_ext_counts, total_bytes, n_files,
              files_ext_counts, n_images}
    """
    legacy = legacy or {}
    lmeta = legacy.get("metadata") or {}
    dc = datacite or {}
    prov: dict[str, str] = {}
    rid = str(scrape.get("source_id"))
    landing = scrape.get("url") or f"https://zenodo.org/records/{rid}"
    dd = "dataset_description"

    doc: dict = {"schema": DD_SCHEMA_URL}
    prov["schema"] = "constant"

    doi = dc.get("doi") or legacy.get("doi") or lmeta.get("doi") or scrape.get("doi")
    if doi:
        doc["identifier"] = {"identifierValue": str(doi), "identifierType": "DOI"}
        prov["identifier"] = "zenodo"
    else:
        doc["identifier"] = {"identifierValue": landing, "identifierType": "URL"}
        prov["identifier"] = "derived (landing URL)"

    titles = []
    for t in dc.get("titles") or []:
        if t.get("title"):
            item = {"titleValue": t["title"]}
            if t.get("titleType"):
                item["titleType"] = _enum(t["titleType"], dd, "titleType")
            titles.append(item)
    if not titles:
        titles = [{"titleValue": lmeta.get("title") or scrape.get("title") or f"Zenodo record {rid}"}]
    doc["title"] = _dedupe(titles)
    prov["title"] = "zenodo_datacite" if dc.get("titles") else "zenodo_legacy"

    version = dc.get("version") or lmeta.get("version")
    if version:
        doc["version"] = str(version)
        prov["version"] = "zenodo"
    else:
        rel = ((lmeta.get("relations") or {}).get("version") or [{}])[0]
        idx = rel.get("index")
        doc["version"] = str(int(idx) + 1) if isinstance(idx, int) else "1"
        prov["version"] = "derived (Zenodo version index + 1)" if isinstance(idx, int) else "placeholder"

    alt = [{"alternateIdentifierValue": landing, "alternateIdentifierType": "URL"}]
    for a in dc.get("alternateIdentifiers") or []:
        v = a.get("alternateIdentifier")
        if v:
            alt.append({"alternateIdentifierValue": v,
                        "alternateIdentifierType": _enum(a.get("alternateIdentifierType"), dd, "identifierType")})
    doc["alternateIdentifier"] = _dedupe(alt)
    prov["alternateIdentifier"] = "zenodo_datacite + landing URL"

    creators = [_person(c, "creator") for c in dc.get("creators") or []]
    creators = [c for c in creators if c]
    if creators:
        prov["creator"] = "zenodo_datacite"
    else:
        creators = [c for c in (_legacy_person(c) for c in lmeta.get("creators") or []) if c]
        prov["creator"] = "zenodo_legacy" if creators else "placeholder"
    if not creators:
        creators = [{"creatorName": "Zenodo depositor (not reported)", "nameType": "Organizational"}]
    doc["creator"] = _dedupe(creators)

    contributors = []
    for c in dc.get("contributors") or []:
        p = _person(c, "contributor")
        if p:
            p = {"contributorType": _enum(c.get("contributorType"), dd, "contributorType"), **p}
            contributors.append(p)
    if contributors:
        doc["contributor"] = _dedupe(contributors)
        prov["contributor"] = "zenodo_datacite"

    year = str(dc.get("publicationYear") or "")[:4]
    if not re.fullmatch(r"\d{4}", year):
        year = str(lmeta.get("publication_date") or legacy.get("created") or "")[:4]
    if not re.fullmatch(r"\d{4}", year):
        year = str(datetime.now(timezone.utc).year)
        prov["publicationYear"] = "placeholder (current year)"
    else:
        prov["publicationYear"] = "zenodo"
    doc["publicationYear"] = year

    dates = [{"dateValue": d["date"], "dateType": _enum(d.get("dateType"), dd, DATE_TYPE)}
             for d in dc.get("dates") or [] if d.get("date")]
    if not dates and lmeta.get("publication_date"):
        dates = [{"dateValue": lmeta["publication_date"], "dateType": "Issued"}]
    if dates:
        doc["date"] = _dedupe(dates)
        prov["date"] = "zenodo"

    present = summary.get("present_classes") or []
    if present:
        rt = "Eye Imaging Dataset: " + ", ".join(CLASS_LABELS[c] for c in present)
        prov["resourceType"] = "classifier (resourceTypeValue) + constant Dataset"
    else:
        rt = "Dataset"
        prov["resourceType"] = "constant (no eye modality detected)"
    doc["resourceType"] = {"resourceTypeValue": rt, "resourceTypeGeneral": "Dataset"}

    access_right = (lmeta.get("access_right") or scrape.get("access_type") or "open").lower()
    has_images = bool(summary.get("n_images") or summary.get("image_ext_counts") or present)
    doc["datasetDeIdentLevel"], prov["datasetDeIdentLevel"] = deident_level(
        access_right, has_images, lmeta.get("embargo_date"))
    doc["datasetConsent"] = dict(PLACEHOLDER_CONSENT)
    prov["datasetConsent"] = "placeholder (not reported by source)"

    descs = []
    for d in dc.get("descriptions") or []:
        v = strip_html(d.get("description") or "")
        if v:
            dt = d.get("descriptionType")
            descs.append({"descriptionValue": v, "descriptionType":
                          dt if dt in ("Abstract", "Methods", "TechnicalInfo", "Other") else "Other"})
    if not descs and (lmeta.get("description") or scrape.get("description")):
        descs = [{"descriptionValue": strip_html(lmeta.get("description") or scrape.get("description")),
                  "descriptionType": "Abstract"}]
    descs = [d for d in descs if d["descriptionValue"]]
    if descs:
        doc["description"] = _dedupe(descs)
        prov["description"] = "zenodo"

    lang = dc.get("language") or lmeta.get("language")
    if lang:
        lang = LANG3.get(str(lang).lower(), str(lang).lower())
        if len(lang) >= 2:
            doc["language"] = lang
            prov["language"] = "zenodo"

    rels = []
    for r in dc.get("relatedIdentifiers") or []:
        v = r.get("relatedIdentifier")
        if not v:
            continue
        item = {"relatedIdentifierValue": v,
                "relatedIdentifierType": _enum(r.get("relatedIdentifierType"), dd, "identifierType"),
                "relationType": _enum(r.get("relationType"), dd, "relationType", default="References")}
        if r.get("resourceTypeGeneral"):
            item["resourceTypeGeneral"] = _enum(r["resourceTypeGeneral"], dd, "resourceItemType")
        rels.append(item)
    if rels:
        doc["relatedIdentifier"] = _dedupe(rels)
        prov["relatedIdentifier"] = "zenodo_datacite"

    # DataCite subjects (with their vocabulary codes when given), then the
    # legacy keywords, then the classifier's modality terms. One entry per
    # subject text (case-insensitive); a coded entry replaces an uncoded one.
    candidates = [_datacite_subject(s) for s in dc.get("subjects") or [] if isinstance(s, dict)]
    candidates += [{"subjectValue": k.strip()} for k in (lmeta.get("keywords") or scrape.get("keywords") or [])
                   if isinstance(k, str) and k.strip()]
    for cls in present:
        name, code, scheme = CLASS_SUBJECTS[cls]
        item = {"subjectValue": name}
        if code:
            item["subjectIdentifier"] = {"classificationCode": code, "subjectScheme": scheme[0],
                                         "schemeURI": scheme[1], "valueURI": scheme[2].format(code)}
        candidates.append(item)
    subjects, where = [], {}
    for item in candidates:
        if not item:
            continue
        key = item["subjectValue"].casefold()
        if key not in where:
            where[key] = len(subjects)
            subjects.append(item)
        elif "subjectIdentifier" in item and "subjectIdentifier" not in subjects[where[key]]:
            subjects[where[key]] = item
    if subjects:
        doc["subject"] = _dedupe(subjects)
        prov["subject"] = ("zenodo subjects/keywords + classifier modality terms" if present
                           else "zenodo subjects/keywords")

    org_name, org_id, org_declared = _managing_org(dc, lmeta)
    doc["managingOrganization"] = {"name": org_name}
    if org_id:
        scheme, uri = org_scheme(org_id, org_declared)
        ident = {"managingOrganizationIdentifierValue": org_id, "managingOrganizationScheme": scheme}
        if uri:
            ident["schemeURI"] = uri
        doc["managingOrganization"]["managingOrganizationIdentifier"] = ident
    prov["managingOrganization"] = "derived (first creator affiliation)" if org_name != "Zenodo" \
        else "derived (publisher fallback)"

    doc["accessType"] = ACCESS_MAP.get(access_right, "Other")
    prov["accessType"] = "zenodo (access_right mapped)"
    text = {
        "open": "Open access; files downloadable from Zenodo without login.",
        "restricted": "Restricted access; files are available on request through Zenodo.",
        "closed": "Closed access; files are not available from Zenodo.",
        "embargoed": "Embargoed on Zenodo; files become available after the embargo date.",
    }.get(access_right, f"Zenodo access_right: {access_right}.")
    # accessDetails.url is omitted on purpose: its schema pattern rejects any
    # URL containing the letter "s" (upstream bug), including every Zenodo URL.
    # urlLastChecked ("the date the url last responded with a 200") is
    # omitted with it: no url is given, so nothing was checked.
    doc["accessDetails"] = {"description": f"{text} Landing page: {landing}"}
    prov["accessDetails"] = "derived (access_right + landing URL)"

    rights = []
    for r in dc.get("rightsList") or []:
        if not r.get("rights"):
            continue  # e.g. bare info:eu-repo/semantics/openAccess
        item = {"rightsName": r["rights"]}
        if r.get("rightsUri"):
            item["rightsURI"] = r["rightsUri"]
        if r.get("rightsIdentifier"):
            item["rightsIdentifier"] = rights_identifier(r["rightsIdentifier"])
        rights.append(item)
    if rights:
        prov["rights"] = "zenodo_datacite"
    else:
        lic = (lmeta.get("license") or {}).get("id") if isinstance(lmeta.get("license"), dict) \
            else (lmeta.get("license") or scrape.get("license"))
        if lic and lic not in ("notspecified",):
            name, uri = LICENSE_NAMES.get(str(lic).lower(), (str(lic), None))
            item = {"rightsName": name, "rightsIdentifier": rights_identifier(str(lic))}
            if uri:
                item["rightsURI"] = uri
            rights = [item]
            prov["rights"] = "zenodo_legacy (license id mapped)"
        else:
            rights = [{"rightsName": "Not specified"}]
            prov["rights"] = "placeholder (no license on record)"
    doc["rights"] = _dedupe(rights)

    pub = (dc.get("publisher") or {})
    pub_name = pub.get("name") if isinstance(pub, dict) else (pub or None)
    doc["publisher"] = {"publisherName": pub_name or "Zenodo"}
    prov["publisher"] = "zenodo"

    funding = []
    for f in dc.get("fundingReferences") or []:
        if not f.get("funderName"):
            continue
        item = {"funderName": f["funderName"]}
        if f.get("funderIdentifier"):
            ftype = f.get("funderIdentifierType")
            item["funderIdentifier"] = {
                "funderIdentifierValue": f["funderIdentifier"],
                "funderIdentifierType": ftype if ftype in ("Crossref Funder ID", "GRID", "ISNI", "ROR") else "Other"}
        if f.get("awardNumber"):
            item["awardNumber"] = {"awardNumberValue": str(f["awardNumber"])}
            if f.get("awardUri"):
                item["awardNumber"]["awardURI"] = f["awardUri"]
        if f.get("awardTitle"):
            item["awardTitle"] = f["awardTitle"]
        funding.append(item)
    if funding:
        doc["fundingReference"] = _dedupe(funding)
        prov["fundingReference"] = "zenodo_datacite"

    size = []
    if summary.get("total_bytes"):
        size.append(_human_bytes(summary["total_bytes"]))
    if summary.get("n_files") is not None:
        size.append(f"{summary['n_files']} files")
    if summary.get("n_images"):
        size.append(f"{summary['n_images']} image files")
    if size:
        doc["size"] = _dedupe(size)
        prov["size"] = "zenodo file list + survey inventory"

    fmts = sorted({MIME[e] for e in list((summary.get("files_ext_counts") or {}).keys())
                   + list((summary.get("image_ext_counts") or {}).keys()) if e in MIME})
    if fmts:
        doc["format"] = fmts
        prov["format"] = "file extensions (Zenodo list + archive members)"
    return doc, prov


def _datacite_subject(s: dict) -> dict | None:
    """One DataCite subject -> {subjectValue[, subjectIdentifier]}.

    subjectIdentifier needs classificationCode and subjectScheme. The code is
    DataCite's classificationCode, else the last segment of valueUri (a MeSH
    or other vocabulary URI such as .../mesh/D012160). A subject without a
    scheme stays plain text.
    """
    value = str(s.get("subject") or "").strip()
    if not value:
        return None
    item: dict = {"subjectValue": value}
    scheme = str(s.get("subjectScheme") or "").strip()
    value_uri = str(s.get("valueUri") or s.get("valueURI") or "").strip()
    code = str(s.get("classificationCode") or "").strip()
    if not code and value_uri:
        code = re.split(r"[/#]", value_uri.rstrip("/#"))[-1].strip()
    if scheme and code:
        ident = {"classificationCode": code, "subjectScheme": scheme}
        scheme_uri = str(s.get("schemeUri") or s.get("schemeURI") or "").strip()
        if scheme_uri:
            ident["schemeURI"] = scheme_uri
        if value_uri:
            ident["valueURI"] = value_uri
        item["subjectIdentifier"] = ident
    return item


def _person(c: dict, role: str) -> dict | None:
    name = (c.get("name") or "").strip()
    if not name:
        return None
    key = "creatorName" if role == "creator" else "contributorName"
    nt = c.get("nameType")
    if nt not in ("Personal", "Organizational"):
        nt = "Personal" if ("," in name or c.get("givenName")) else "Organizational"
    out: dict = {key: name, "nameType": nt}
    ids = []
    for ni in c.get("nameIdentifiers") or []:
        v, scheme = ni.get("nameIdentifier"), ni.get("nameIdentifierScheme")
        if not v or not scheme:
            continue
        item = {"nameIdentifierValue": v, "nameIdentifierScheme": scheme}
        if scheme.upper() == "ORCID":
            item["nameIdentifierScheme"] = "ORCID"
            item["schemeURI"] = "https://orcid.org/"
            if not v.startswith("http"):
                item["nameIdentifierValue"] = f"https://orcid.org/{v}"
        ids.append(item)
    if ids:
        out["nameIdentifier"] = _dedupe(ids)
    affs = []
    for a in c.get("affiliation") or []:
        if isinstance(a, str):
            a = {"name": a}
        if not a.get("name"):
            continue
        item = {"affiliationName": a["name"]}
        if a.get("affiliationIdentifier"):
            scheme, uri = org_scheme(a["affiliationIdentifier"], a.get("affiliationIdentifierScheme"))
            item["affiliationIdentifier"] = {
                "affiliationIdentifierValue": a["affiliationIdentifier"],
                "affiliationIdentifierScheme": scheme}
            if uri:
                item["affiliationIdentifier"]["schemeURI"] = uri
        affs.append(item)
    if affs:
        out["affiliation"] = _dedupe(affs)
    return out


def _legacy_person(c: dict) -> dict | None:
    name = (c.get("name") or "").strip()
    if not name:
        return None
    out: dict = {"creatorName": name, "nameType": "Personal" if "," in name else "Organizational"}
    if c.get("orcid"):
        out["nameIdentifier"] = [{"nameIdentifierValue": f"https://orcid.org/{c['orcid']}",
                                  "nameIdentifierScheme": "ORCID", "schemeURI": "https://orcid.org/"}]
    if c.get("affiliation"):
        out["affiliation"] = [{"affiliationName": c["affiliation"]}]
    return out


def _managing_org(dc: dict, lmeta: dict) -> tuple[str, str | None, str | None]:
    """(name, identifier, declared identifier scheme) of the first affiliation."""
    for c in dc.get("creators") or []:
        for a in c.get("affiliation") or []:
            if isinstance(a, dict) and a.get("name"):
                return a["name"], a.get("affiliationIdentifier"), a.get("affiliationIdentifierScheme")
            if isinstance(a, str) and a:
                return a, None, None
    for c in lmeta.get("creators") or []:
        if c.get("affiliation"):
            return c["affiliation"], None, None
    return "Zenodo", None, None


def _human_bytes(n: int) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


# ---------------------------------------------------------------------------
# dataset_structure_description
# ---------------------------------------------------------------------------
def build_structure_description(classes: dict, present: list[str], summary: dict) -> dict:
    """CMDS directoryList from the classifier composition.

    classes: {cls: {"count": sampled confident count, "est_files": int,
                     "mean_conf": float}}
    summary: {n_images, n_classified, dicom_classes: [classes whose sampled
              images included real DICOM files], formats: [str]}
    NEG and UNCERTAIN never become directories (CMDS has no place for them);
    their counts live in the workbook.
    """
    by_dtype: dict[str, list[str]] = {}
    for cls in present:
        by_dtype.setdefault(CLASS_DIRS[cls][0], []).append(cls)

    n_img, n_cls = summary.get("n_images", 0), summary.get("n_classified", 0)
    sampled = n_cls < n_img
    # Review flags (runner.modality_review): the modality may be a classifier
    # false positive (non-eye images in an eye class); say so in the text.
    review = (f" REVIEW: this modality assignment may be a false positive; flags: {summary['review_note']}."
              if summary.get("review_note") else "")
    dirs = []
    for dtype in ("retinal_photography", "retinal_oct", "retinal_octa"):
        if dtype not in by_dtype:
            continue
        desc, terms = DATATYPE_INFO[dtype]
        standards = [{
            "standardName": CMDS_NAME,
            "standardDescription": "Standard for consistently structuring and describing clinical research datasets",
            "standardUse": "This directory and its sub-directories are named and organized following the "
                           "specification from this standard.",
            "standardRelatedIdentifier": [{"relatedIdentifierValue": CMDS_URL, "relatedIdentifierType": "URL",
                                           "relationType": "IsDescribedBy"}],
        }]
        # DICOM is named only for a directory whose own classes had sampled
        # DICOM files (a non-eye CT series elsewhere in the record does not
        # make the fundus photographs DICOM).
        if set(by_dtype[dtype]) & set(summary.get("dicom_classes") or ()):
            standards.append({
                "standardName": "Digital Imaging and Communications in Medicine (DICOM)",
                "standardDescription": "Standard for the digital storage and transmission of medical images and "
                                       "related information.",
                "standardUse": "Some data files within this directory are DICOM files as published by the depositor.",
                "standardIdentifier": [{"identifierValue": "https://doi.org/10.25504/FAIRsharing.b7z8by",
                                        "identifierType": "DOI"}],
                "standardRelatedIdentifier": [{"relatedIdentifierValue": "http://medical.nema.org/",
                                               "relatedIdentifierType": "URL", "relationType": "IsDescribedBy"}],
            })
        related_terms = []
        for value, codes in terms:
            t = {"relatedTermValue": value}
            if codes:
                t["relatedTermIdentifier"] = [_term(value, c) for c in codes]
            related_terms.append(t)
        subdirs = []
        total = 0
        for cls in by_dtype[dtype]:
            info = classes[cls]
            est = int(info["est_files"])
            total += est
            basis = (f"estimated from {info['count']} of {n_cls} sampled images ({n_img} total)"
                     if sampled else f"{info['count']} of {n_img} images classified")
            subdirs.append({
                "directoryName": CLASS_DIRS[cls][1],
                "directoryType": "modality",
                "directoryDescription": (
                    f"{CLASS_DIRS[cls][2]}; modality assigned by the EyeACT image classifier "
                    f"(predicted class {cls}, {basis}, mean confidence {info['mean_conf']:.2f}). "
                    f"File formats as published: {', '.join(summary.get('formats') or []) or 'unknown'}."
                    + review),
                "numberOfFiles": est,
            })
        dirs.append({
            "directoryName": dtype, "directoryType": "dataType", "directoryDescription": desc + review,
            "relatedTerm": related_terms, "relatedStandard": standards,
            "numberOfFiles": total, "directoryList": subdirs,
        })
    return {
        "schema": DSD_SCHEMA_URL,
        "directoryList": dirs,
        "metadataFileList": [{
            "metadataFileName": "dataset_description.json",
            "metadataFileDescription": "Dataset-level metadata harvested from the Zenodo record and mapped to "
                                       "the AI-READI dataset_description schema.",
        }],
    }


def flatten_dd(dd: dict, max_description: int = 2000) -> dict:
    """Workbook columns (dd_*) for the dataset_description fields that the
    summary columns do not carry. The JSON files remain the full record."""
    def ident(i: dict) -> str:
        return f"{i.get('relationType', '')} {i.get('relatedIdentifierType', '')}:{i.get('relatedIdentifierValue', '')}"

    desc = " | ".join(f"[{d.get('descriptionType')}] {d.get('descriptionValue')}" for d in dd.get("description") or [])
    if len(desc) > max_description:
        desc = desc[:max_description] + " [truncated; full text in dataset_description.json]"
    funding = []
    for f in dd.get("fundingReference") or []:
        parts = [f.get("funderName", "")]
        if f.get("awardNumber"):
            parts.append(f["awardNumber"].get("awardNumberValue", ""))
        if f.get("awardTitle"):
            parts.append(f["awardTitle"])
        funding.append(" / ".join(p for p in parts if p))
    subjects = []
    for s in dd.get("subject") or []:
        sid = s.get("subjectIdentifier") or {}
        code = f" ({sid.get('subjectScheme', '').split(' (')[0]} {sid.get('classificationCode')})" if sid else ""
        subjects.append(f"{s.get('subjectValue')}{code}")
    return {
        "dd_description": desc,
        "dd_subjects": "; ".join(subjects),
        "dd_relatedIdentifiers": "; ".join(ident(i) for i in dd.get("relatedIdentifier") or []),
        "dd_funding": "; ".join(funding),
        "dd_dates": "; ".join(f"{d.get('dateType')}: {d.get('dateValue')}" for d in dd.get("date") or []),
        "dd_language": dd.get("language", ""),
        "dd_format": "; ".join(dd.get("format") or []),
        "dd_size": "; ".join(dd.get("size") or []),
        "dd_contributors": "; ".join(c.get("contributorName", "") for c in dd.get("contributor") or [])[:1000],
        "dd_publisher": (dd.get("publisher") or {}).get("publisherName", ""),
        "dd_identifier": (dd.get("identifier") or {}).get("identifierValue", ""),
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate(doc: dict, schema_name: str) -> list[str]:
    """Validation errors as short strings ([] when valid, or a note when
    jsonschema is not installed)."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema not installed; not validated"]
    schema = load_schema(schema_name)
    v = jsonschema.Draft7Validator(schema)
    errs = []
    for e in sorted(v.iter_errors(doc), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in e.path) or "(root)"
        errs.append(f"{loc}: {e.message}"[:300])
    return errs
