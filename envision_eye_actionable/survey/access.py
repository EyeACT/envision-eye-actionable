"""Access enrichment of restricted and embargoed Zenodo records (``enrich-access``).

A restricted Zenodo record has public metadata but hidden files: the file
list is empty and the files are granted case by case by the owner, through
the "Request access" form on the landing page. The survey gives such records
the status ``restricted`` from the legacy record JSON and reads no file.
This side command adds what a person needs to decide whether to ask for
access, without touching the survey's results:

* the InvenioRDM record JSON (``Accept: application/vnd.inveniordm.v1+json``)
  of every target record, cached under ``<out-dir>/cache/invenio``: its
  ``access`` block (record, files, status, embargo active/until/reason), the
  parent's access settings (``allow_user_requests``, ``allow_guest_requests``,
  ``accept_conditions_text``: whether the owner accepts requests through
  Zenodo, as far as Zenodo exposes it to a non-owner), the license ids and
  the resource type, and ``links.access_request`` (the API endpoint behind
  the landing page form);
* the discovery SetFit label and probability from the scrape and a keyword
  flag (eye imaging terms in title, keywords, subjects and description);
* for records whose files are still not public, the two CMDS documents
  written again from the full metadata into ``<out-dir>/cmds/<id>/``, with
  the de-identification and access texts of a non-public record (access
  restricted by the depositor, de-identification not reported; whether the
  owner takes requests; the embargo end).

Targets: the scrape records with ``access_type`` restricted or embargoed,
and the records whose last results row has status restricted, embargoed or
closed or an ``access_right`` of restricted, embargoed or closed. The
survey's ``survey_results.jsonl`` is only read; rows go to
``<out-dir>/access_results.jsonl`` (one line per record, the last line per
id wins, resumable: records already done are skipped unless ``refresh``).
The workbook step merges them as a supplement (see excel.py): they set the
``access_*`` and CMDS columns of the survey rows and never the
classification columns, and they feed the Access_Requests sheet.

Every request goes through ZenodoClient, so with ``shared_state_dir`` this
job shares the 429 cooldown and the combined per-minute budget with a
survey running at the same time; give it a small ``rpm`` share. Legacy and
DataCite JSON are read from the survey's metadata cache when present
(``read_cache_dirs``, read only) and fetched into this job's own cache
otherwise.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import cmds as cmds_mod
from .dicom_map import strip_html
from .results import load_results
from .zenodo import API, RemoteError, ZenodoClient, load_scrape, load_token, redact

INVENIO_ACCEPT = "application/vnd.inveniordm.v1+json"
SUPPLEMENT_NAME = "access_results.jsonl"
SUMMARY_NAME = "access_summary.json"
TARGET_ACCESS = ("restricted", "embargoed", "closed")
TARGET_STATUSES = ("restricted", "embargoed", "closed")
LANDING = "https://zenodo.org/records/{}"
EYE_PROB_MIN = 0.5            # SetFit p(eye imaging) from which a record counts as eye-relevant

# Eye imaging terms for the keyword flag. Acronyms are matched case
# sensitively (OCT, OCTA, AMD, SLO, FAF, UWF), words case insensitively.
_EYE_WORDS = re.compile(
    r"\b(retina[ls]?|retinopath\w*|fundus|fundi|ophthalm\w*|optical coherence tomograph\w*|macula[r]?|"
    r"maculopath\w*|glaucoma\w*|choroid\w*|uveitis|keratoconus|cornea[l]?|autofluorescen\w*|"
    r"fluorescein angiograph\w*|indocyanine green|scanning laser ophthalmoscop\w*|optic (?:nerve|disc|disk)|"
    r"eyes?|ocular|intraocular|vitreo\w*|diabetic eye|age[- ]related macular)\b", re.I)
_EYE_ACRONYMS = re.compile(r"\b(OCTA?|AMD|nAMD|SLO|FAF|UWF|DME|RNFL)\b")

# License ids (Zenodo, lower case) by what the project may do with the
# files: train (reuse permitted), evaluate only (NC or ND), or neither
# (unknown or no license: do not train).
_TRAIN_OK = re.compile(r"^(cc0|cc-zero|cc-by(-sa)?(-\d\.\d)?|odc-by|pddl|odc-pddl|mit|mit-license|bsd|"
                       r"bsd-\d-clause|apache|apache-2\.0|public-domain|pdm|cc-pdm)", re.I)
_NC_ND = re.compile(r"(^|-)(nc|nd)(-|$)", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def keyword_terms(*texts) -> list[str]:
    """Distinct eye imaging terms found in the texts (lower case words,
    acronyms as written), in order of first appearance."""
    seen: dict[str, None] = {}
    for t in texts:
        if not t:
            continue
        t = t if isinstance(t, str) else " ".join(str(x) for x in t)
        for m in _EYE_WORDS.finditer(t):
            seen.setdefault(m.group(1).lower(), None)
        for m in _EYE_ACRONYMS.finditer(t):
            seen.setdefault(m.group(1), None)
    return list(seen)


def license_use(license_ids) -> str:
    """What the project may do with a record's files by its license ids:
    'train' (reuse permitted), 'evaluate_only_nc_nd' (NC or ND: internal
    evaluation only), 'no_license' (none given: do not train) or
    'check_license' (another license: read it before any use)."""
    ids = [str(x).strip().lower() for x in (license_ids or []) if x and str(x).strip()]
    ids = [i for i in ids if i not in ("notspecified", "none")]
    if not ids:
        return "no_license"
    if any(_NC_ND.search(i) for i in ids):
        return "evaluate_only_nc_nd"
    if all(_TRAIN_OK.match(i) for i in ids):
        return "train"
    return "check_license"


def select_targets(scrape: list[dict], rows: dict[str, dict], ids: list[str] | None = None) -> list[str]:
    """Record ids to enrich, scrape order first, then result rows not in the
    scrape: scrape access_type restricted or embargoed, or a results row
    with status restricted, embargoed or closed, or access_right restricted,
    embargoed or closed. With ``ids``, only those among the targets."""
    out: dict[str, None] = {}
    for rec in scrape:
        rid = str(rec["source_id"])
        row = rows.get(rid) or {}
        if ((rec.get("access_type") or "").lower() in TARGET_ACCESS
                or row.get("status") in TARGET_STATUSES
                or (row.get("access_right") or "").lower() in TARGET_ACCESS):
            out[rid] = None
    for rid, row in rows.items():
        if row.get("status") in TARGET_STATUSES or (row.get("access_right") or "").lower() in TARGET_ACCESS:
            out.setdefault(str(rid), None)
    if ids:
        want = {str(i) for i in ids}
        return [r for r in out if r in want]
    return list(out)


def parse_access(doc: dict | None) -> dict:
    """The access_* fields of an InvenioRDM record JSON. Fields Zenodo does
    not expose are None (never guessed)."""
    doc = doc or {}
    acc = doc.get("access") or {}
    emb = acc.get("embargo") or {}
    parent_acc = ((doc.get("parent") or {}).get("access") or {})
    settings = parent_acc.get("settings") if isinstance(parent_acc.get("settings"), dict) else None
    meta = doc.get("metadata") or {}
    rights = [r for r in meta.get("rights") or [] if isinstance(r, dict)]
    lic_ids = [r.get("id") for r in rights if r.get("id")]
    lic_titles = [((r.get("title") or {}).get("en") if isinstance(r.get("title"), dict) else r.get("title"))
                  for r in rights]
    rt = meta.get("resource_type") or {}
    files = doc.get("files") or {}
    files_access = acc.get("files")
    embargo_active = emb.get("active")
    rid = str(doc.get("id") or "")
    out = {
        "access_record": acc.get("record"),
        "access_files": files_access,
        "access_status": acc.get("status"),
        "access_embargo_active": embargo_active,
        "access_embargo_until": emb.get("until"),
        "access_embargo_reason": emb.get("reason"),
        "access_files_public_now": files_access == "public" and not embargo_active,
        "access_files_enabled": files.get("enabled") if isinstance(files, dict) else None,
        "access_request_url": LANDING.format(rid) if rid else None,
        "access_request_api": (doc.get("links") or {}).get("access_request"),
        "access_settings_exposed": settings is not None,
        "access_allow_user_requests": settings.get("allow_user_requests") if settings else None,
        "access_allow_guest_requests": settings.get("allow_guest_requests") if settings else None,
        "access_accept_conditions_text": strip_html(settings.get("accept_conditions_text") or "")[:2000]
        if settings else None,
        "access_license_ids": "; ".join(lic_ids),
        "access_license": "; ".join(t for t in lic_titles if t) or "; ".join(lic_ids),
        "access_license_use": license_use(lic_ids),
        "access_resource_type": rt.get("id"),
        "access_title": meta.get("title"),
        "access_publication_date": meta.get("publication_date"),
    }
    out["access_request_note"] = request_note(out)
    return out


def request_note(a: dict) -> str:
    """One line on how to get the files, from the parsed access fields."""
    if a.get("access_files_public_now"):
        return "files are public now; no request needed"
    parts = []
    if a.get("access_embargo_active"):
        parts.append(f"embargoed until {a.get('access_embargo_until') or 'a date not given'}")
    if a.get("access_allow_user_requests") is True:
        parts.append("the owner accepts access requests from logged-in Zenodo users (Request access on the "
                     "landing page)")
    elif a.get("access_allow_user_requests") is False:
        parts.append("the owner does not accept access requests through Zenodo; contact the depositor")
    else:
        parts.append("Zenodo does not expose whether the owner accepts requests; try Request access on the "
                     "landing page or contact the depositor")
    if a.get("access_allow_guest_requests") is True:
        parts.append("guest (no account) requests are accepted too")
    if a.get("access_accept_conditions_text"):
        parts.append("the owner sets conditions (see access_accept_conditions_text)")
    return "; ".join(parts)


def access_details_text(a: dict, landing: str) -> str:
    """accessDetails.description of a non-public record from the parsed
    InvenioRDM access fields."""
    kind = "Embargoed" if a.get("access_embargo_active") else "Restricted"
    head = (f"{kind} by the depositor on Zenodo"
            + (f" until {a['access_embargo_until']}" if a.get("access_embargo_active")
               and a.get("access_embargo_until") else "")
            + f" (record {a.get('access_record') or 'unknown'}, files {a.get('access_files') or 'unknown'}).")
    note = request_note(a)
    return f"{head} {note[:1].upper()}{note[1:]}. Landing page: {landing}"


def build_cmds(rec: dict, legacy: dict | None, datacite: dict | None, a: dict) -> tuple[dict, dict, dict]:
    """(dataset_description, provenance, dataset_structure_description) of a
    non-public record from its full metadata, with the access and
    de-identification texts of the InvenioRDM access fields."""
    rid = str(rec["source_id"])
    lmeta = dict((legacy or {}).get("metadata") or {})
    access_right = "embargoed" if a.get("access_embargo_active") else (
        lmeta.get("access_right") or rec.get("access_type") or "restricted")
    if access_right == "open":          # files not public now, whatever the legacy JSON said
        access_right = "restricted"
    lmeta["access_right"] = access_right
    leg = {**(legacy or {}), "metadata": lmeta}
    files = (legacy or {}).get("files") or []
    summary = {"present_classes": [], "n_files": len(files) if files else None,
               "total_bytes": sum(int(f.get("size") or 0) for f in files) or None}
    dd, prov = cmds_mod.build_dataset_description(rec, leg, datacite, summary)
    dd["datasetDeIdentLevel"], prov["datasetDeIdentLevel"] = cmds_mod.deident_level(
        access_right, False, a.get("access_embargo_until"))
    landing = rec.get("url") or LANDING.format(rid)
    dd["accessDetails"] = {"description": access_details_text(a, landing)}
    prov["accessDetails"] = "zenodo_inveniordm (access, parent access settings) + landing URL"
    dsd = cmds_mod.build_structure_description({}, [], {"n_images": 0, "n_classified": 0, "dicom_classes": [],
                                                        "formats": [], "review_note": ""})
    return dd, prov, dsd


def cmds_row_fields(dd: dict, prov: dict, dsd: dict) -> dict:
    """The CMDS columns of a row (as runner.Survey._write_cmds sets them)."""
    dd_err = cmds_mod.validate(dd, "dataset_description")
    dsd_err = cmds_mod.validate(dsd, "dataset_structure_description")
    out = dict(cmds_mod.flatten_dd(dd))
    out.update({
        "dd_valid": not dd_err, "dd_errors": "; ".join(dd_err[:5]),
        "dsd_valid": not dsd_err, "dsd_errors": "; ".join(dsd_err[:5]),
        "cmds_placeholder_fields": ", ".join(k for k, v in prov.items() if v.startswith("placeholder")),
        "cmds_derived_fields": ", ".join(k for k, v in prov.items() if v.startswith("derived")),
        "cmds_classifier_fields": "",
        "cmds_provenance": prov,
        "cmds_resourceTypeValue": dd["resourceType"]["resourceTypeValue"],
        "cmds_creators": "; ".join(c["creatorName"] for c in dd["creator"])[:500],
        "cmds_publicationYear": dd["publicationYear"],
        "cmds_version": dd["version"],
        "cmds_accessType": dd["accessType"],
        "cmds_rights": "; ".join(r["rightsName"] for r in dd["rights"]),
        "cmds_managingOrganization": dd["managingOrganization"]["name"],
        "cmds_directories": "",
        "cmds_note": "metadata-only CMDS written again by enrich-access from the full record metadata "
                     "(files not public)",
    })
    return out


def check_out_dir(out_dir: Path, survey_out_dir: Path | None):
    """Refuse an out dir that is, holds or lies inside the survey's out dir,
    or that holds a survey_results.jsonl: this job must never write where
    the survey writes."""
    o = Path(out_dir).resolve()
    if (o / "survey_results.jsonl").exists():
        raise ValueError(f"{out_dir} holds a survey_results.jsonl; give enrich-access its own --out-dir")
    if survey_out_dir is not None:
        s = Path(survey_out_dir).resolve()
        if o == s or s in o.parents or o in s.parents:
            raise ValueError(f"--out-dir {out_dir} overlaps the survey out dir {survey_out_dir}; "
                             "use a separate dir (e.g. results/<run>/access)")


class AccessEnricher:
    """Fetch, parse and write the access rows of the target records."""

    def __init__(self, scrape_path: Path, survey_out_dir: Path | None, out_dir: Path,
                 cache_dir: Path | None = None, read_cache_dirs: list[Path] | None = None,
                 rpm: int = 10, shared_state_dir: Path | None = None, shared_rpm: int = 120,
                 min_interval: float = 0.5, token_file: Path | None = None, offline: bool = False,
                 write_cmds: bool = True, refresh: bool = False, client: ZenodoClient | None = None):
        check_out_dir(out_dir, survey_out_dir)
        self.scrape_path = scrape_path
        self.survey_out_dir = survey_out_dir
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else self.out_dir / "cache"
        (self.cache_dir / "invenio").mkdir(parents=True, exist_ok=True)
        if read_cache_dirs is None:
            read_cache_dirs = [survey_out_dir / "cache"] if survey_out_dir is not None else []
        self.read_cache_dirs = [Path(d) for d in read_cache_dirs]
        self.write_cmds = write_cmds
        self.refresh = refresh
        self.offline = offline
        if client is None:
            token = None if offline else load_token(token_file)
            client = ZenodoClient(self.cache_dir, metadata_dir=None, min_interval=min_interval, offline=offline,
                                  max_per_minute=rpm, shared_state_dir=shared_state_dir,
                                  shared_per_minute=shared_rpm, token=token)
        self.client = client
        self.results_path = self.out_dir / SUPPLEMENT_NAME

    # -- metadata ------------------------------------------------------------
    def _read_only(self, kind: str, rid: str) -> dict | None:
        for d in self.read_cache_dirs + [self.cache_dir]:
            p = d / kind / f"{rid}.json"
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
        return None

    def legacy(self, rid: str) -> tuple[dict | None, str]:
        doc = self._read_only("legacy", rid)
        if doc is not None:
            return doc, "cache"
        return self.client.legacy(rid), "fetched"

    def datacite(self, rid: str) -> tuple[dict | None, str]:
        doc = self._read_only("datacite", rid)
        if doc is not None:
            return doc, "cache"
        return self.client.datacite(rid), "fetched"

    def invenio(self, rid: str) -> tuple[dict | None, str | None]:
        """(InvenioRDM record JSON, error). Cached; a 404 or 410 (deleted or
        unknown record) is an error, not retried."""
        p = self.cache_dir / "invenio" / f"{rid}.json"
        if p.is_file() and not self.refresh:
            try:
                return json.loads(p.read_text(encoding="utf-8")), None
            except (OSError, ValueError):
                pass
        if self.offline:
            return None, "offline"
        try:
            r = self.client.get(f"{API}/{rid}", headers={"Accept": INVENIO_ACCEPT}, timeout=(20, 60))
        except RemoteError as e:
            return None, f"inveniordm record JSON: {e}"
        try:
            doc = r.json()
        except ValueError:
            return None, "inveniordm record JSON: not JSON"
        finally:
            r.close()
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(p)
        return doc, None

    # -- rows ----------------------------------------------------------------
    def enrich_record(self, rec: dict, row: dict | None) -> dict:
        rid = str(rec["source_id"])
        out = {"record_id": rid, "supplement": "access", "access_checked_at": _now(),
               "access_survey_status": (row or {}).get("status"),
               "access_scrape_access_type": rec.get("access_type"),
               "access_setfit_label": rec.get("label"),
               "access_setfit_prob": rec.get("prob_eye_imaging"),
               "access_request_url": rec.get("url") or LANDING.format(rid)}
        doc, err = self.invenio(rid)
        out["access_fetch_ok"] = doc is not None
        if err:
            out["access_error"] = redact(err)[:300]
        if doc is not None:
            out.update(parse_access(doc))
            out["access_request_url"] = rec.get("url") or out.get("access_request_url") or LANDING.format(rid)
        meta = (doc or {}).get("metadata") or {}
        subjects = [s.get("subject") for s in meta.get("subjects") or [] if isinstance(s, dict)]
        terms = keyword_terms(rec.get("title"), meta.get("title"), rec.get("keywords") or [], subjects,
                              strip_html(meta.get("description") or rec.get("description") or ""))
        out["access_keyword_terms"] = "; ".join(terms)
        out["access_keyword_match"] = bool(terms)
        prob = rec.get("prob_eye_imaging")
        out["access_eye_relevant"] = bool(rec.get("label") == "EYE_IMAGING"
                                          or (isinstance(prob, (int, float)) and prob >= EYE_PROB_MIN) or terms)
        if not out.get("access_title"):
            out["access_title"] = rec.get("title")
        if not out.get("access_license_ids") and rec.get("license"):
            out["access_license_ids"] = rec["license"]
            out["access_license"] = rec["license"]
            out["access_license_use"] = license_use([rec["license"]])
            out["access_license_source"] = "scrape"
        elif out.get("access_license_ids") is not None:
            out["access_license_source"] = "zenodo_inveniordm"
        if doc is not None and not out.get("access_files_public_now") and self.write_cmds:
            legacy, lsrc = self.legacy(rid)
            datacite, dsrc = self.datacite(rid)
            out["access_metadata_sources"] = f"legacy {lsrc if legacy is not None else 'missing'}, " \
                                             f"datacite {dsrc if datacite is not None else 'missing'}"
            try:
                dd, prov, dsd = build_cmds(rec, legacy, datacite, out)
                folder = self.out_dir / "cmds" / rid
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "dataset_description.json").write_text(
                    redact(json.dumps(dd, indent=2, ensure_ascii=False)), encoding="utf-8")
                (folder / "dataset_structure_description.json").write_text(
                    redact(json.dumps(dsd, indent=2, ensure_ascii=False)), encoding="utf-8")
                out.update(cmds_row_fields(dd, prov, dsd))
                out["cmds_dir"] = folder.relative_to(self.out_dir).as_posix()
            except Exception as e:  # noqa: BLE001 - one record never stops the job
                out["access_cmds_error"] = redact(f"{type(e).__name__}: {e}")[:300]
        return out

    def done_ids(self) -> set[str]:
        """Ids with a successful row already (skipped unless refresh)."""
        if self.refresh:
            return set()
        return {rid for rid, r in load_results(self.results_path).items() if r.get("access_fetch_ok")}

    def run(self, ids: list[str] | None = None, limit: int | None = None, progress_every: int = 50) -> dict:
        scrape = load_scrape(self.scrape_path)
        by_id = {str(r["source_id"]): r for r in scrape}
        rows = load_results(self.survey_out_dir / "survey_results.jsonl") if self.survey_out_dir else {}
        targets = select_targets(scrape, rows, ids)
        done = self.done_ids()
        todo = [r for r in targets if r not in done]
        if limit is not None:
            todo = todo[:limit]
        stats = {"started_at": _now(), "targets": len(targets), "already_done": sum(r in done for r in targets),
                 "todo": len(todo), "written": 0, "fetch_failed": 0, "cmds_written": 0}
        self._write_summary({**stats, "state": "running"})
        t0 = time.time()
        with open(self.results_path, "a", encoding="utf-8") as fp:
            for i, rid in enumerate(todo, 1):
                rec = by_id.get(rid) or {"source_id": rid, "title": (rows.get(rid) or {}).get("title"),
                                         "url": LANDING.format(rid),
                                         "access_type": (rows.get(rid) or {}).get("access_type"),
                                         "license": (rows.get(rid) or {}).get("license")}
                out = self.enrich_record(rec, rows.get(rid))
                fp.write(redact(json.dumps(out, default=str, ensure_ascii=False)) + "\n")
                fp.flush()
                stats["written"] += 1
                stats["fetch_failed"] += not out["access_fetch_ok"]
                stats["cmds_written"] += bool(out.get("cmds_dir"))
                if i % progress_every == 0 or i == len(todo):
                    stats["elapsed_s"] = round(time.time() - t0, 1)
                    stats["zenodo_requests"] = self.client.n_requests
                    self._write_summary({**stats, "state": "running"})
                    print(f"[enrich-access] {i}/{len(todo)} records, {self.client.n_requests} requests, "
                          f"{stats['elapsed_s']:.0f}s", file=sys.stderr, flush=True)
        stats["elapsed_s"] = round(time.time() - t0, 1)
        stats["zenodo_requests"] = self.client.n_requests
        stats["finished_at"] = _now()
        stats.update(summarize(self.results_path))
        self._write_summary({**stats, "state": "done"})
        return stats

    def _write_summary(self, d: dict):
        p = self.out_dir / SUMMARY_NAME
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
        tmp.replace(p)


def summarize(results_path: Path) -> dict:
    """Counts over the access rows (last line per id)."""
    from collections import Counter
    rows = load_results(results_path)
    c = Counter()
    for r in rows.values():
        c["rows"] += 1
        c["fetch_ok"] += bool(r.get("access_fetch_ok"))
        c[f"files_{r.get('access_files')}"] += 1
        c["files_public_now"] += bool(r.get("access_files_public_now"))
        c["embargo_active"] += bool(r.get("access_embargo_active"))
        c[f"allow_user_requests_{r.get('access_allow_user_requests')}"] += 1
        c["eye_relevant"] += bool(r.get("access_eye_relevant"))
        c["keyword_match"] += bool(r.get("access_keyword_match"))
        c["eye_relevant_not_public"] += bool(r.get("access_eye_relevant") and not r.get("access_files_public_now"))
        c["cmds"] += bool(r.get("cmds_dir"))
        c[f"license_use_{r.get('access_license_use')}"] += 1
    return {"counts": dict(sorted(c.items()))}
