"""DICOM-aligned attributes for classified records.

Rules come from ``resources/dicom_mapping.json`` (checked against DICOM PS3 2026d):
per-class Modality / SOP Class / Image Type / anatomy and device codes,
pixel-header derivations, a laterality regex with guards, and manufacturer
hints from text and proprietary extensions.

These are DICOM-aligned attributes, not DICOM conformance: a JPEG is never a
DICOM instance. Every record says how its values were obtained in
``dicom_mapping_status``:

* read_from_header: Modality, SOP Class and Image Type come from the headers
  of sampled DICOM files of the dominant class (majority value); Body Part,
  Anatomic Region, Acquisition Device Type and Ophthalmic Image Type are also
  taken from those headers when present (``dicom_header_attributes_used``);
* mixed: DICOM files were sampled, but none of the dominant class, so those
  values still come from the class mapping (header values stay in the
  ``dicom_header_*`` columns);
* inferred_from_classifier: no DICOM sampled, values from the class mapping;
* not_applicable: no classified images.

Laterality prefers the DICOM header (ImageLaterality, then Laterality) and
falls back to file and directory names; ``laterality_source`` counts both.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from functools import lru_cache
from importlib import resources
from pathlib import PurePosixPath

from .constants import EYE_CLASSES, VENDOR_OCT_EXTS

_LAT_MAP = {"od": "R", "re": "R", "right": "R", "r": "R",
            "os": "L", "le": "L", "left": "L", "l": "L",
            "ou": "B", "both": "B", "bilateral": "B"}
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_OPTOS_WEAK = {"california", "daytona", "silverstone", "monaco", "200tx", "uwf",
               "ultra-widefield", "ultrawidefield", "ultra-wide-field"}


@lru_cache(maxsize=1)
def mapping() -> dict:
    """The bundled DICOM mapping JSON."""
    txt = resources.files(__package__).joinpath("resources/dicom_mapping.json").read_text(encoding="utf-8")
    return json.loads(txt)


# ---------------------------------------------------------------------------
# laterality
# ---------------------------------------------------------------------------
def laterality(path: str) -> tuple[str, str, bool]:
    """(R|L|B|U, source basename|dirname|none, conflict) for one file path.

    Basename first, then parent directories nearest-first; the first component
    with a hit wins. Bare r/l only count when the component has another
    delimited field ("12_L.png", "pt3-R-faf.tif"); tokens glued to digits
    ("L1", "R2") never match because tokens split on non-alphanumerics only.
    "re"/"le" are accepted in basenames only (too many English words in dirs).
    """
    parts = [p for p in re.split(r"[\\/]|!/", path) if p]
    if not parts:
        return "U", "none", False
    base = parts[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    comps = [(stem, "basename")] + [(d, "dirname") for d in reversed(parts[:-1])]
    for comp, source in comps:
        toks = [t for t in _TOKEN_SPLIT.split(comp.lower()) if t]
        found = set()
        for t in toks:
            if t not in _LAT_MAP:
                continue
            if t in ("r", "l") and len(toks) < 2:
                continue
            if t in ("re", "le") and source != "basename":
                continue
            found.add(_LAT_MAP[t])
        if not found:
            continue
        sides = found - {"B"}
        if len(sides) > 1:
            return "U", source, True
        return (sides.pop() if sides else "B"), source, False
    return "U", "none", False


# ---------------------------------------------------------------------------
# manufacturer hints
# ---------------------------------------------------------------------------
def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def manufacturer_hints(text: str, names: list[str]) -> list[dict]:
    """Manufacturer mentions in record text and file names, strongest first.

    Returns [{manufacturer, model, matched, source}] with source 'text',
    'filename' or 'extension'. Proprietary extensions (.e2e, .fds, ...) are
    the strongest non-header evidence.
    """
    m = mapping()["manufacturer_hints_from_text"]
    out: list[dict] = []
    seen = set()
    ext_counts = Counter(PurePosixPath(n).suffix.lower() for n in names)
    for ext, maker in m["proprietary_extension_hints"].items():
        if ext in VENDOR_OCT_EXTS and ext_counts.get(ext):
            out.append({"manufacturer": maker.split(" (")[0], "model": "",
                        "matched": f"{ext} x{ext_counts[ext]}", "source": "extension"})
            seen.add(maker.split(" (")[0])
    joined_names = " ".join(names[:5000]).lower()
    low_text = (text or "").lower()
    for maker, spec in m["patterns"].items():
        rx = re.compile(spec["regex"], re.I)
        for src_name, hay in (("text", low_text), ("filename", joined_names)):
            hit = rx.search(hay)
            if not hit:
                continue
            # Optos model names are ordinary words ("University of California",
            # "Monaco") and UWF is not Optos-specific: require the brand too.
            if (maker == "Optos" and hit.group(0).lower().replace(" ", "-") in _OPTOS_WEAK
                    and not re.search(r"optos|optomap", hay)):
                continue
            model = ""
            for key, label in (spec.get("models") or {}).items():
                if key in hay:
                    model = label
                    break
            if maker not in seen:
                out.append({"manufacturer": maker, "model": model,
                            "matched": hit.group(0)[:40], "source": src_name})
                seen.add(maker)
            break
    return out


# ---------------------------------------------------------------------------
# per-class mapping
# ---------------------------------------------------------------------------
def _code(c) -> str:
    if not c:
        return ""
    if isinstance(c, list):
        return "; ".join(_code(x) for x in c)
    return f"{c.get('CodingSchemeDesignator', '')} {c.get('CodeValue', '')} {c.get('CodeMeaning', '')}".strip()


# OCT anatomy overrides. File and directory paths may use the looser forms;
# record prose (title, description, keywords) must use explicit phrases,
# because words like "angle" or "cornea" occur in ordinary descriptions of
# posterior-segment OCT sets ("scan angle", "through the cornea").
_ANTERIOR_PATH = re.compile(
    r"anterior[ _-]?(?:segment|chamber)|cornea|(?:^|[^a-z])as-?oct(?:[^a-z]|$)|iridocorneal")
_ANTERIOR_TEXT = re.compile(
    r"anterior[ -]segment|\bas-?oct\b|anterior chamber|\bcorneal oct\b|iridocorneal angle")
_ONH_PATH = re.compile(r"(?:^|[^a-z])onh(?:[^a-z]|$)|optic[ _-]?(?:disc|disk|nerve)|rnfl|peripapillary")
_ONH_TEXT = re.compile(
    r"\bperipapillary\b|\bcircumpapillary\b|optic nerve head (?:oct|scans?|cubes?|volumes?)"
    r"|\bonh[ -]oct\b|optic dis[ck] (?:oct|scans?|cubes?|volumes?)")


def class_attributes(cls: str, bits: int | None = None, maker: str = "",
                     path_text: str = "", blue_zero_frac: float = 0.0,
                     record_text: str = "") -> dict:
    """DICOM-aligned attributes for one classifier class (see mapping JSON).

    ``path_text``: file paths of this class's images. ``record_text``: record
    title, description and keywords (only strict phrases are used from it).
    """
    spec = mapping()["classes"].get(cls)
    if not spec:
        return {}
    sop = spec["SOPClassUID_0008_0016"]
    sop_name = spec["SOPClassName"]
    # 9..16 stored bits fit BitsAllocated 16 (Ophthalmic Photography 16 Bit).
    # Deeper or float samples fit no OP SOP class; they keep the 8-bit class
    # and are flagged by aggregate() (dicom_pixel_depth_note).
    if bits and 8 < bits <= 16 and spec.get("SOPClassUID_if_16bit"):
        sop, sop_name = spec["SOPClassUID_if_16bit"], spec["SOPClassName_if_16bit"]
    device = spec.get("AcquisitionDeviceTypeCodeSequence_0022_0015")
    if cls == "FAF" and device and device.get("alternative") and re.search(
            r"topcon|canon|kowa|visucam", maker, re.I):
        device = device["alternative"]
    filt = spec.get("LightPathFilterTypeStackCodeSequence_0022_0017")
    if cls == "FAF" and re.search(r"optos", maker, re.I):
        filt = {"CodingSchemeDesignator": "SCT", "CodeValue": "445465004", "CodeMeaning": "Green optical filter"}
    anat = spec.get("AnatomicRegionSequence_0008_2218")
    low_paths = path_text.lower()
    low_text = (record_text or "").lower()
    if cls == "OCT" and anat:
        if _ANTERIOR_PATH.search(low_paths) or _ANTERIOR_TEXT.search(low_text):
            anat = {"CodingSchemeDesignator": "SCT", "CodeValue": "31636006", "CodeMeaning": "Anterior chamber of eye"}
        elif _ONH_PATH.search(low_paths) or _ONH_TEXT.search(low_text):
            anat = {"CodingSchemeDesignator": "SCT", "CodeValue": "81016008", "CodeMeaning": "Optic nerve head"}
    oph_type = ""
    if cls == "OCTA":
        oph_type = _octa_slab(spec["OphthalmicImageTypeCodeSequence_0022_1615"], low_paths, low_text)
    channels = spec.get("ChannelDescriptionCodeSequence_0022_001A")
    samples_used = None
    if cls == "PSC":
        samples_used = 2 if blue_zero_frac >= 0.5 else None
        if samples_used is None:
            channels = None
    return {
        "Modality": spec["Modality_0008_0060"],
        "SOPClassUID": sop,
        "SOPClassName": sop_name,
        "ImageType": "\\".join(spec.get("ImageType_0008_0008") or []),
        "BodyPartExamined": spec.get("BodyPartExamined_0018_0015") or "",
        "AnatomicRegion": _code(anat),
        "AcquisitionDeviceType": _code(device),
        "LightPathFilter": _code(filt),
        "ChannelDescription": _code(channels),
        "SamplesPerPixelUsed": samples_used,
        "OphthalmicImageType": oph_type,
        "IlluminationType": _code(spec.get("IlluminationTypeCodeSequence_0022_0016")),
    }


# OCTA slab phrases allowed in record prose (title, description, keywords).
# Loose slab keywords ("deep", "cc", "disc", "choroid") are only read from
# file paths: in prose they match "deep learning", "CC BY 4.0", ...
_SLAB_TEXT = {
    "128265": re.compile(r"\bsuperficial (?:capillary|vascular|retinal) (?:plexus|complex|layer|slab)"),
    "128269": re.compile(r"\bdeep (?:capillary|vascular|retinal) (?:plexus|complex|layer|slab)"),
    "128271": re.compile(r"\bouter retina(?:l)? (?:slab|flow|layer)|\bavascular (?:outer retina|slab)\b"),
    "128273": re.compile(r"\bchoriocapillaris\b"),
    "128275": re.compile(r"\bchoroid(?:al)? (?:slab|vasculature|vessel layer|flow)\b"),
    "128263": re.compile(r"\bradial peripapillary capillar"),
}


def _octa_slab(o: dict, low_paths: str, low_text: str) -> str:
    """Ophthalmic Image Type (0022,1615) for OCTA en face images.

    Paths may use the loose slab keywords of the mapping JSON; record text
    only the strict phrases of _SLAB_TEXT. The paths are tried first; when
    they (or the text) name more than one slab, the record holds several
    slabs and the generic default (128259) is kept.
    """
    default = f"DCM {o['CodeValue']} {o['CodeMeaning']}"
    overrides = o.get("slab_keyword_overrides", {})
    meaning = {alt["CodeValue"]: alt["CodeMeaning"] for alt in overrides.values()}
    hits = {alt["CodeValue"] for pattern, alt in overrides.items()
            if re.search(rf"(?:^|[^a-z])(?:{pattern})(?:[^a-z]|$)", low_paths)}
    if not hits:
        hits = {code for code, rx in _SLAB_TEXT.items() if rx.search(low_text)}
    if len(hits) == 1:
        code = hits.pop()
        return f"DCM {code} {meaning.get(code, '')}".strip()
    return default


# ---------------------------------------------------------------------------
# record-level aggregation
# ---------------------------------------------------------------------------
def _range(vals: list) -> str:
    vals = [v for v in vals if v]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _top(counter: Counter, n: int = 3) -> str:
    return "; ".join(f"{k} ({v})" for k, v in counter.most_common(n) if k not in (None, ""))


def aggregate(images: list[dict], all_paths: list[str], present_classes: list[str],
              dominant: str | None, text: str, other_names: list[str]) -> dict:
    """Roll per-image facts up to one record row.

    images: [{cls, facts, path}] for every classified image (sample).
    all_paths: display paths of every image entry (laterality runs on all).
    """
    facts = [im["facts"] for im in images]
    # Laterality: DICOM ImageLaterality (0020,0062), then Laterality
    # (0020,0060), for sampled DICOM files; file/dir names for everything else.
    header_lat: dict[str, str] = {}
    for im in images:
        d = (im.get("facts") or {}).get("dicom") or {}
        for tag in ("ImageLaterality", "Laterality"):
            v = str(d.get(tag) or "").strip().upper()
            if v in ("R", "L", "B"):
                if im.get("path"):
                    header_lat[im["path"]] = v
                break
    lat = Counter()
    lat_src = Counter()
    lat_conflicts = 0
    for p in all_paths:
        if p in header_lat:
            side, src, conflict = header_lat[p], "dicom_header", False
        else:
            side, src, conflict = laterality(p)
        lat[side] += 1
        if side != "U":
            lat_src[src] += 1
        lat_conflicts += int(conflict)

    hints = manufacturer_hints(text, [PurePosixPath(p).name for p in all_paths[:20000]] + other_names)
    exif_makers = Counter(f.get("make") for f in facts)
    dicom_facts = [f["dicom"] for f in facts if f.get("dicom")]
    header_maker = Counter(d.get("Manufacturer") for d in dicom_facts)
    maker = ""
    maker_source = ""
    for counter, src in ((header_maker, "dicom_header"), (exif_makers, "exif")):
        top = [k for k, _ in counter.most_common() if k]
        if top:
            maker, maker_source = top[0], src
            break
    if not maker and hints:
        maker, maker_source = hints[0]["manufacturer"], f"hint_{hints[0]['source']}"
    # Maker evidence from the IR images only (header, then EXIF), for the IR
    # device columns: a record-level majority maker (Zeiss OCT DICOMs) says
    # nothing about the device of its IR images. An image counts as IR when
    # its thresholded label or its argmax class is IR, the same population
    # as the IR trigger in ir_device_fields (thresholded or argmax fraction).
    ir_facts = [im["facts"] for im in images
                if "IR" in (im.get("cls"), im.get("top")) and im.get("facts")]
    ir_maker = ir_maker_source = ir_model = ""
    for counter, src in ((Counter((f.get("dicom") or {}).get("Manufacturer") for f in ir_facts), "dicom_header"),
                         (Counter(f.get("make") for f in ir_facts), "exif")):
        top = [k for k, _ in counter.most_common() if k]
        if top:
            ir_maker, ir_maker_source = str(top[0]), src
            models = Counter(((f.get("dicom") or {}).get("ManufacturerModelName") or f.get("model"))
                             for f in ir_facts)
            ir_model = next((str(k) for k, _ in models.most_common() if k), "")
            break

    # Per-class DICOM attributes for every present class. Anatomy and OCTA
    # slab keywords are read from the paths of that class's own images, and
    # from strict phrases of the record text.
    per_class = {}
    for cls in present_classes:
        cls_images = [im for im in images if im["cls"] == cls]
        cls_facts = [im["facts"] for im in cls_images]
        cls_paths = " ".join(im.get("path") or "" for im in cls_images[:5000])
        bits = Counter(f.get("bits") for f in cls_facts).most_common(1)
        blue = sum(1 for f in cls_facts if f.get("blue_zero")) / max(1, len(cls_facts))
        per_class[cls] = class_attributes(cls, bits[0][0] if bits else None, maker, cls_paths, blue,
                                          record_text=text or "")

    # QC: real DICOM header Modality vs the classifier's mapped Modality.
    qc_agree = qc_total = 0
    for im in images:
        d = (im["facts"] or {}).get("dicom") or {}
        if d.get("Modality") and im["cls"] in mapping()["classes"]:
            qc_total += 1
            qc_agree += int(d["Modality"] == mapping()["classes"][im["cls"]]["Modality_0008_0060"])

    # Record-level Modality / SOP Class / Image Type: from the headers of the
    # dominant class's sampled DICOM files when there are any (majority
    # value per attribute), else from the class mapping. A DICOM file of
    # another class (a stray CT series next to fundus JPEGs) never makes the
    # record "read_from_header".
    dom = dict(per_class.get(dominant or "", {})) if dominant else {}
    dom_dicom = [im["facts"]["dicom"] for im in images
                 if dominant and im["cls"] == dominant and (im.get("facts") or {}).get("dicom")]
    from_header = False
    header_attrs: list[str] = []
    if dom_dicom:
        # (header keyword, record key, sets the read_from_header status)
        for key, attr, core in (("Modality", "Modality", True), ("SOPClassUID", "SOPClassUID", True),
                                ("ImageType", "ImageType", True),
                                ("BodyPartExamined", "BodyPartExamined", False),
                                ("AnatomicRegionSequence", "AnatomicRegion", False),
                                ("AcquisitionDeviceTypeCodeSequence", "AcquisitionDeviceType", False),
                                ("OphthalmicImageTypeCodeSequence", "OphthalmicImageType", False)):
            vals = Counter(_dicom_value(d.get(key)) for d in dom_dicom)
            vals.pop("", None)
            if vals:
                dom[attr] = vals.most_common(1)[0][0]
                header_attrs.append(attr)
                from_header = from_header or core
        if "SOPClassUID" in header_attrs:
            dom["SOPClassName"] = sop_class_name(dom["SOPClassUID"]) or dom.get("SOPClassName", "")
    if from_header:
        status = "read_from_header"
    elif dicom_facts:
        status = "mixed"   # DICOM sampled, but not the source of the record values
    elif images:
        status = "inferred_from_classifier"
    else:
        status = "not_applicable"

    lossy = Counter(f.get("lossy") for f in facts)
    spacing = Counter(tuple(sp) for sp in (f.get("spacing_mm") for f in facts)
                      if isinstance(sp, (list, tuple)) and len(sp) == 2)
    samples = Counter(_samples_per_pixel(f) for f in facts)
    allocated = Counter(_bits_allocated(f) for f in facts)
    high_bit = Counter((int(f["bits"]) - 1) if f.get("bits") else None for f in facts)
    too_deep = sum(1 for f in facts if f.get("float") or (f.get("bits") or 0) > 16)
    return {
        "dicom_mapping_status": status,
        "dicom_Modality": dom.get("Modality", ""),
        "dicom_SOPClassUID": dom.get("SOPClassUID", ""),
        "dicom_SOPClassName": dom.get("SOPClassName", ""),
        "dicom_ImageType": dom.get("ImageType", ""),
        "dicom_BodyPartExamined": dom.get("BodyPartExamined", ""),
        "dicom_AnatomicRegion_code": dom.get("AnatomicRegion", ""),
        "dicom_AcquisitionDeviceType_code": dom.get("AcquisitionDeviceType", ""),
        "dicom_LightPathFilter_code": dom.get("LightPathFilter", ""),
        "dicom_ChannelDescription_code": dom.get("ChannelDescription", ""),
        "dicom_OphthalmicImageType_code": dom.get("OphthalmicImageType", ""),
        "dicom_IlluminationType_code": dom.get("IlluminationType", ""),
        "dicom_header_attributes_used": ", ".join(header_attrs),
        "dicom_Modalities_present": "; ".join(sorted({v["Modality"] for v in per_class.values()})),
        "dicom_per_class": per_class,
        "dicom_ImageLaterality_dist": " ".join(f"{k}:{lat[k]}" for k in ("R", "L", "B", "U") if lat[k]),
        "laterality_R": lat["R"], "laterality_L": lat["L"], "laterality_B": lat["B"],
        "laterality_U": lat["U"], "laterality_conflicts": lat_conflicts,
        "laterality_source": "; ".join(f"{k}:{lat_src[k]}" for k in ("dicom_header", "basename", "dirname")
                                       if lat_src[k]),
        "dicom_Rows_range": _range([f.get("rows") for f in facts]),
        "dicom_Columns_range": _range([f.get("cols") for f in facts]),
        "dicom_PhotometricInterpretation_set": _top(Counter(f.get("photometric") for f in facts), 4),
        "dicom_SamplesPerPixel_set": _top(samples, 3),
        "dicom_BitsAllocated_set": _top(allocated, 4),
        "dicom_BitsStored_set": _top(Counter(f.get("bits") for f in facts), 4),
        "dicom_HighBit_set": _top(high_bit, 4),
        "dicom_pixel_depth_note": (f"{too_deep} sampled image(s) have float or >16-bit samples; no Ophthalmic "
                                   "Photography SOP class holds them without rescaling to 16 bit"
                                   if too_deep else ""),
        "dicom_NumberOfFrames_max": max([int(f.get("frames") or 1) for f in facts], default=0),
        "multiframe_files": sum(1 for f in facts if int(f.get("frames") or 1) > 1),
        "dicom_PixelSpacing_mm": "; ".join(f"{a}\\{b} ({n})" for (a, b), n in spacing.most_common(3)),
        "dicom_LossyImageCompression": _top(lossy, 2),
        "dicom_LossyImageCompressionMethod": _top(Counter(f.get("lossy_method") for f in facts), 3),
        "dicom_ImageComments": _top(Counter(_short(f.get("description"), 80) for f in facts), 3),
        "rgb_is_gray_frac": round(sum(1 for f in facts if f.get("rgb_is_gray")) / len(facts), 3) if facts else None,
        # Over the images the pixel check ran on (decoded images; undecodable
        # inventory-only entries carry no mask_like fact). Unweighted sample
        # share; the stratum-weighted share over classified images is frac_MASK.
        "mask_like_frac": (round(sum(1 for f in checked if f["mask_like"]) / len(checked), 3)
                           if (checked := [f for f in facts if "mask_like" in f]) else None),
        "dicom_Manufacturer": maker,
        "dicom_Manufacturer_source": maker_source,
        "ir_Manufacturer": ir_maker,
        "ir_Manufacturer_source": ir_maker_source,
        "ir_ManufacturerModelName": ir_model,
        "dicom_ManufacturerModelName": _top(Counter(f.get("model") for f in facts)),
        "dicom_SoftwareVersions": _top(Counter(f.get("software") for f in facts)),
        "dicom_AcquisitionDateTime_range": _range([f.get("acq_datetime") for f in facts]),
        "dicom_HorizontalFieldOfView": _top(Counter(_dicom_value(d.get("HorizontalFieldOfView"))
                                                    for d in dicom_facts), 3),
        "dicom_IlluminationWaveLength": _top(Counter(_dicom_value(d.get("IlluminationWaveLength"))
                                                     for d in dicom_facts), 3),
        "dicom_SeriesDescription": _top(Counter(_short(d.get("SeriesDescription"), 60)
                                                for d in dicom_facts), 3),
        "manufacturer_hint_text": "; ".join(f"{h['manufacturer']}{' ' + h['model'] if h['model'] else ''}"
                                            f" [{h['matched']}]" for h in hints),
        "manufacturer_hint_source": "; ".join(sorted({h["source"] for h in hints})),
        "n_real_dicom_sampled": len(dicom_facts),
        "dicom_header_Modality_set": _top(Counter(d.get("Modality") for d in dicom_facts), 4),
        "dicom_header_SOPClassUID_set": _top(Counter(d.get("SOPClassUID") for d in dicom_facts), 3),
        "dicom_qc_modality_agreement": f"{qc_agree}/{qc_total}" if qc_total else "",
    }


def _short(v, n: int) -> str:
    """One-line, truncated text value ('' when missing)."""
    if not v:
        return ""
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _samples_per_pixel(f: dict) -> int | None:
    """SamplesPerPixel (0028,0002) for the image as DICOM would encode it.

    DICOM fixes the sample count by PhotometricInterpretation (RGB/YBR 3,
    MONOCHROME/PALETTE 1) and has no alpha channel, so a TIFF/PNG header
    value that includes extra samples (RGBA = 4, gray+alpha = 2) is reduced
    to the DICOM count instead of reporting an impossible "RGB, 4 samples".
    """
    pi = str(f.get("photometric") or "").upper()
    if pi.startswith(("RGB", "YBR")):
        return 3
    if pi.startswith(("MONOCHROME", "PALETTE")):
        return 1
    if f.get("samples"):
        return int(f["samples"])
    return None


def _bits_allocated(f: dict) -> int | None:
    """BitsAllocated (0028,0100): the header value, else stored bits rounded up."""
    if f.get("bits_allocated"):
        return int(f["bits_allocated"])
    b = f.get("bits")
    if not b:
        return None
    b = int(b)
    return 1 if b == 1 else 8 if b <= 8 else 16 if b <= 16 else 32 if b <= 32 else 64


def _dicom_value(v) -> str:
    """Header value as a DICOM string (multi-valued ImageType joined by '\\')."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "\\".join(str(x) for x in v)
    return str(v).strip()


def sop_class_name(uid: str) -> str:
    """Name of a SOP Class UID: the mapping JSON first, then pydicom's table."""
    for spec in mapping()["classes"].values():
        if spec.get("SOPClassUID_0008_0016") == uid:
            return spec.get("SOPClassName", "")
        if spec.get("SOPClassUID_if_16bit") == uid:
            return spec.get("SOPClassName_if_16bit", "")
    try:
        from pydicom.uid import UID
        name = UID(uid).name
        return "" if name == uid else name
    except Exception:  # noqa: BLE001 - pydicom missing or odd UID
        return ""


def eye_classes_present(counts: dict, classified: int, min_fraction: float) -> list[str]:
    """Eye classes (never NEG/UNCERTAIN) with at least ``min_fraction`` of the sample."""
    if not classified:
        return []
    return [c for c in EYE_CLASSES
            if counts.get(c, 0) >= 1 and counts.get(c, 0) / classified >= min_fraction]


IR_SUBSTANTIAL_FRACTION = 0.2
_HEIDELBERG_RX = re.compile(r"heidelberg|spectralis|\bhra ?2?\b|\bhrt\b|\.e2e\b|\.sdb\b", re.I)


def ir_device_fields(row: dict) -> dict:
    """Device evidence for records where IR is dominant or substantial.

    IR is substantial when it is the dominant class (thresholded or argmax;
    not the dominant *eye* class, which a 3% IR share can win in a record
    that is mostly NEG) or makes up at least 20% of the non-mask classified
    images, weighted like frac_* (``frac_IR_non_mask`` or
    ``argmax_frac_IR_non_mask``). ``ir_device_hint`` then lists the manufacturer
    evidence found: the DICOM header or EXIF maker of the IR-classified
    images themselves (strong, labelled "(IR images, dicom_header|exif)"),
    then record-level text, file name and extension hints (weak, labelled
    "(record-level ...)"), or says that none was found.
    ``ir_non_spectralis_candidate`` is True when the evidence names a maker
    that is not Heidelberg (Spectralis, HRA, HRT, .e2e, .sdb), so IR from
    other devices stands out. When the IR images carry a maker (strong
    evidence), that maker alone decides; the record-level hints decide only
    when there is no strong evidence. Records with no evidence at all are
    not candidates (the device is unknown, and most IR in the training data
    is Spectralis).

    The fields follow the record's eye decision: a record with no eye class
    present (``present_eye_classes`` empty, status no_eye_images) gets no IR
    device fields, even when its argmax share of IR is substantial (mostly
    UNCERTAIN images whose argmax is IR), so a no-eye record never shows up
    as a non-Spectralis IR candidate. Rows without that column (older rows)
    are judged on the IR shares alone.
    """
    if "present_eye_classes" in row and not row.get("present_eye_classes"):
        return {"ir_device_hint": "", "ir_non_spectralis_candidate": False}
    n = row.get("n_classified") or 0
    # Shares over the non-mask images (masks carry no modality); rows written
    # before those columns existed fall back to the all-image shares.
    if "frac_IR_non_mask" in row or "argmax_frac_IR_non_mask" in row:
        frac = max(row.get("frac_IR_non_mask") or 0, row.get("argmax_frac_IR_non_mask") or 0)
    else:
        frac = max(row.get("frac_IR") or 0, row.get("argmax_frac_IR") or 0)
    dominant = "IR" in (row.get("dominant_class"), row.get("argmax_dominant_class"))
    if not n or not (dominant or frac >= IR_SUBSTANTIAL_FRACTION):
        return {"ir_device_hint": "", "ir_non_spectralis_candidate": False}
    evidence = []
    strong = ""
    maker, source = row.get("ir_Manufacturer") or "", row.get("ir_Manufacturer_source") or ""
    if maker and source in ("dicom_header", "exif"):
        model = row.get("ir_ManufacturerModelName") or ""
        strong = f"{maker}{' ' + model if model else ''}"
        evidence.append(f"{strong} (IR images, {source})")
    if row.get("manufacturer_hint_text"):
        evidence.append(f"{row['manufacturer_hint_text']} "
                        f"(record-level {row.get('manufacturer_hint_source') or 'hint'})")
    if not evidence:
        return {"ir_device_hint": "no manufacturer hint found", "ir_non_spectralis_candidate": False}
    text = "; ".join(evidence)
    # Strong evidence (the IR images' own header or EXIF maker) decides on its
    # own: a Spectralis mention in the record text, often about its OCT, must
    # not hide a Topcon or Optos IR device. Record-level hints decide only
    # when the IR images carry no maker.
    decisive = strong or text
    return {"ir_device_hint": text[:1000], "ir_non_spectralis_candidate": not _HEIDELBERG_RX.search(decisive)}


# Review signals for eye-modality results that may be false positives.
REVIEW_SETFIT_MAX_P = 0.2       # SetFit p(eye imaging) below this: the metadata says not eye
REVIEW_MASK_LIKE_FRAC = 0.3     # this share of mask-like images (binary or label maps)
REVIEW_LOW_CONFIDENCE = 0.75    # mean top-1 confidence of the dominant eye class below this


def modality_review(row: dict) -> dict:
    """Flags on a record's eye-modality result that call for a human check.

    The image classifier knows only eye modalities and NEG, so non-eye
    grayscale images (microscopy, binary masks, X-ray or CT slices, tissue
    OCT) can land in an eye class, most often OCTA. The flags:

    * ``setfit_not_eye``: the metadata classifier gives p(eye imaging)
      under REVIEW_SETFIT_MAX_P (or labels the record NEGATIVE);
    * ``mask_like``: at least REVIEW_MASK_LIKE_FRAC of the classified images
      are mask-like (images.is_mask_like), by ``frac_MASK``
      (stratum-weighted, like the other frac_* columns);
    * ``low_confidence``: the dominant eye class has a mean top-1 confidence
      under REVIEW_LOW_CONFIDENCE.

    ``modality_low_trust`` is True when an eye modality is present and the
    metadata or the mask share contradicts it (setfit_not_eye or
    mask_like); low confidence alone is reported but does not set it.
    Nothing is removed: the counts and CMDS documents stay, and the flags
    are written into the CMDS directory descriptions as well."""
    present = [c for c in (row.get("present_eye_classes") or "").split(", ") if c]
    if not present:
        return {"modality_review_flags": "", "modality_low_trust": False}
    flags, strong = [], False
    p_eye = row.get("setfit_prob_eye_imaging")
    label = str(row.get("setfit_label") or "").upper()
    if (isinstance(p_eye, (int, float)) and p_eye < REVIEW_SETFIT_MAX_P) or \
            (p_eye is None and label == "NEGATIVE"):
        p_txt = f"{p_eye:.3f}" if isinstance(p_eye, (int, float)) else "n/a"
        flags.append(f"setfit_not_eye (SetFit p(eye) {p_txt}, label {label or 'n/a'})")
        strong = True
    # frac_MASK: stratum-weighted share of the classified images labelled
    # MASK, the same denominator as the other frac_* columns. Rows written
    # before MASK was a label fall back to mask_like_frac.
    mask = row.get("frac_MASK")
    if not isinstance(mask, (int, float)):
        mask = row.get("mask_like_frac")
    if isinstance(mask, (int, float)) and mask >= REVIEW_MASK_LIKE_FRAC:
        flags.append(f"mask_like ({mask:.0%} of classified images are masks or label maps)")
        strong = True
    dom = row.get("dominant_eye_class")
    conf = ((row.get("_class_detail") or {}).get(dom) or {}).get("mean_conf") if dom else None
    if isinstance(conf, (int, float)) and conf < REVIEW_LOW_CONFIDENCE:
        flags.append(f"low_confidence ({dom} mean top-1 confidence {conf:.2f})")
    return {"modality_review_flags": "; ".join(flags), "modality_low_trust": strong}
