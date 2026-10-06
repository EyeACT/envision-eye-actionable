"""Figshare source for the survey: article metadata, file URLs, DataCite.

The survey pipeline is written around a Zenodo record: a "legacy" record
JSON (file list with md5 and links, access_right, title, creators, ...)
and a DataCite JSON. FigshareClient is a ZenodoClient whose legacy() and
datacite() return the same two documents for a Figshare article, so the
fetch, probe, remote zip, classify, CMDS and weblink code runs unchanged.

Record ids: Figshare article ids and Zenodo record ids are both integers
and can collide, so a Figshare record is ``figshare-<article id>``
everywhere in the survey (rows, spool, cmds/, keep dir). load_scrape adds
the prefix to records whose ``source`` is figshare.

Metadata, in order of preference:

* the article JSON (``GET /v2/articles/<id>``): from the pull's metadata
  dir (``--metadata-dir``, files ``<article id>.json``) when present, else
  fetched and cached in ``<cache>/figshare/``. It is converted to the
  legacy shape (to_legacy) and cached in ``<cache>/legacy/<record id>.json``.
* DataCite JSON for the article DOI
  (``https://api.datacite.org/application/vnd.datacite.datacite+json/<doi>``,
  the same document Zenodo serves with that Accept header). When DataCite
  has no record of the DOI, a minimal DataCite-shaped document is built from
  the article (``_source: figshare_article``) so CMDS still gets creators,
  titles, rights and dates.

Access: Figshare marks an article confidential, embargoed (whole article or
files), a metadata-only record, or download-disabled. Only open articles
list their files to the survey; the others get an empty file list and
access_right closed, embargoed or restricted, so the fetch role gives them
status restricted exactly as for Zenodo. A metadata-only record (data held
elsewhere) is open with no files: status no_files, and its links go to the
weblink catalogue. Link-only files (``is_link_only``, a URL instead of a
file) are never downloaded; their URLs go to the weblink catalogue too.

Files: download_url (``https://ndownloader.figshare.com/files/<id>``)
redirects to a signed S3 URL. Range requests survive the redirect (206 with
Content-Range), so remote zip central directories, archive probes and
member range reads work as for Zenodo. Figshare has no container API: zips
are listed and read by range requests only.

Rate limits: Figshare publishes no numeric limit and sends no rate limit
headers; it asks clients not to exceed about one request per second. The
defaults are 60 requests per minute and 1.0 s between requests, every
request counted (API, DataCite, downloads, each range read). 429 and 5xx
back off as for Zenodo (429 pauses every worker); a 403 from
api.figshare.com, which Figshare's abuse filter sends instead of 429, is
retried with backoff too. The 429 cooldown and the per-minute budget use
their own files in the shared state dir (figshare_not_before,
figshare_requests), apart from Zenodo's.

Token: FIGSHARE_ACCESS_TOKEN, else --token-file, else
~/.config/envision-survey/figshare_token when it exists. Requests to api.figshare.com carry ``Authorization: token ...``. It is
never sent to another host, never logged and registered for redaction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote, urlparse

from requests.auth import AuthBase

from .zenodo import FIGSHARE_PREFIX, RETRY_STATUSES, ZenodoClient, register_secret

logger = logging.getLogger(__name__)

API = "https://api.figshare.com/v2"
DATACITE_URL = "https://api.datacite.org/application/vnd.datacite.datacite+json/"
LANDING = "https://figshare.com/articles/{}"
TOKEN_ENV = "FIGSHARE_ACCESS_TOKEN"
# Read when neither FIGSHARE_ACCESS_TOKEN nor --token-file is given and the file exists.
DEFAULT_TOKEN_FILE = Path("~/.config/envision-survey/figshare_token")
MAX_PER_MINUTE = 60
MIN_INTERVAL = 1.0
API_HOST = "api.figshare.com"


def article_id(record_id: str) -> str:
    """Article id of a record id (``figshare-123`` or ``123``)."""
    rid = str(record_id)
    return rid[len(FIGSHARE_PREFIX):] if rid.startswith(FIGSHARE_PREFIX) else rid


def record_id(article: int | str) -> str:
    return f"{FIGSHARE_PREFIX}{article_id(str(article))}"


def load_token(token_file: Path | None = None, use_default: bool = True) -> str | None:
    """Figshare token from FIGSHARE_ACCESS_TOKEN, else the first line of
    ``token_file``, else (``use_default``) of DEFAULT_TOKEN_FILE when it
    exists; None when there is none. An explicit ``token_file`` that cannot
    be read is an error. Registered for redaction, never logged."""
    tok = os.environ.get(TOKEN_ENV, "").strip()
    path = Path(token_file).expanduser() if token_file is not None else None
    if path is None and use_default and DEFAULT_TOKEN_FILE.expanduser().is_file():
        path = DEFAULT_TOKEN_FILE.expanduser()
    if not tok and path is not None:
        try:
            tok = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except (OSError, IndexError):
            raise ValueError(f"cannot read the Figshare token file {path}") from None
    if tok:
        register_secret(tok)
        return tok
    return None


class FigshareToken(AuthBase):
    """``Authorization: token`` for api.figshare.com only (never on the
    redirect to S3 or to any other host). Its repr never shows the token."""

    def __init__(self, token: str):
        self._token = token

    def __call__(self, r):
        if (urlparse(r.url).hostname or "").lower() == API_HOST:
            r.headers["Authorization"] = f"token {self._token}"
        else:
            r.headers.pop("Authorization", None)
        return r

    def __repr__(self) -> str:
        return "FigshareToken(<token hidden>)"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# article -> legacy record JSON, DataCite JSON
# ---------------------------------------------------------------------------
def access_right(article: dict) -> str:
    """Zenodo-style access_right of an article: closed (confidential),
    embargoed, restricted (downloads disabled) or open. A metadata-only
    record is open: it has nothing to restrict, its data is elsewhere."""
    if article.get("is_confidential"):
        return "closed"
    if article.get("is_embargoed"):
        return "embargoed"
    if article.get("download_disabled"):
        return "restricted"
    return "open"


def _md5(f: dict) -> str | None:
    v = (f.get("computed_md5") or f.get("supplied_md5") or "").strip().lower()
    return v if re.fullmatch(r"[0-9a-f]{32}", v) else None


def file_entries(article: dict) -> tuple[list[dict], list[str]]:
    """(legacy file entries, link-only URLs) of an article. File names are
    unique keys: a repeated name gets its Figshare file id as a prefix."""
    files, links, seen = [], [], set()
    for f in article.get("files") or []:
        url = f.get("download_url")
        if f.get("is_link_only"):
            if url:
                links.append(url)
            continue
        name = str(f.get("name") or f.get("id") or "").replace("\\", "/").split("/")[-1].strip()
        if not name or not url:
            continue
        key = name if name not in seen else f"{f.get('id')}_{name}"
        seen.add(key)
        md5 = _md5(f)
        files.append({"key": key, "size": int(f.get("size") or 0), "checksum": f"md5:{md5}" if md5 else "",
                      "links": {"self": url}, "id": f.get("id"), "mimetype": f.get("mimetype")})
    return files, links


def _person(a: dict) -> dict | None:
    """Zenodo legacy creator: ``Family, Given`` when Figshare gives the
    parts (a comma marks a person in cmds._legacy_person)."""
    first, last = (a.get("first_name") or "").strip(), (a.get("last_name") or "").strip()
    name = f"{last}, {first}" if first and last else (a.get("full_name") or "").strip()
    if not name:
        return None
    out = {"name": name}
    if a.get("orcid_id"):
        out["orcid"] = a["orcid_id"]
    return out


LICENSE_IDS = {"cc by 4.0": "cc-by-4.0", "cc0": "cc-zero", "cc by-sa 4.0": "cc-by-sa-4.0",
               "cc by-nc 4.0": "cc-by-nc-4.0", "cc by-nc-sa 4.0": "cc-by-nc-sa-4.0",
               "cc by-nc-nd 4.0": "cc-by-nc-nd-4.0", "cc by-nd 4.0": "cc-by-nd-4.0", "mit": "mit",
               "apache 2.0": "apache-2.0", "odc-by 1.0": "odc-by-1.0"}


def to_legacy(article: dict) -> dict:
    """The article as a Zenodo legacy record JSON (the fields the survey
    reads), plus ``_figshare`` with the access facts."""
    access = access_right(article)
    files, link_only = file_entries(article)
    withheld = len(files) if access != "open" else 0
    if access != "open":
        files = []                     # listed perhaps, but not downloadable: never planned
    lic = article.get("license") or {}
    lic_name = lic.get("name") if isinstance(lic, dict) else (str(lic) if lic else None)
    lic_id = LICENSE_IDS.get(str(lic_name or "").strip().lower(), lic_name)
    weblinks = [{"url": u, "type": "figshare_link_only_file"} for u in link_only]
    for m in article.get("related_materials") or []:
        if isinstance(m, dict):
            u = m.get("link") or (f"https://doi.org/{m['identifier']}" if m.get("identifier_type") == "DOI"
                                  and m.get("identifier") else m.get("identifier"))
            if u and str(u).startswith(("http://", "https://")):
                weblinks.append({"url": u, "type": f"figshare_related_material ({m.get('relation') or ''})"})
    meta = {
        "title": article.get("title") or "",
        "description": article.get("description") or "",
        "creators": [p for p in (_person(a) for a in article.get("authors") or []) if p],
        "access_right": access,
        "keywords": list(article.get("tags") or article.get("keywords") or [])
        + [c.get("title") for c in article.get("categories") or [] if c.get("title")],
        "publication_date": str(article.get("published_date") or "")[:10] or None,
        "version": str(article.get("version")) if article.get("version") is not None else None,
        "doi": article.get("doi") or None,
        "resource_type": {"type": article.get("defined_type_name")},
    }
    if lic_id:
        meta["license"] = {"id": lic_id}
    if access == "embargoed" and article.get("embargo_date"):
        meta["embargo_date"] = str(article["embargo_date"])[:10]
    aid = article.get("id")
    return {
        "id": record_id(aid) if aid is not None else None,
        "doi": article.get("doi") or None,
        "created": article.get("created_date"),
        "metadata": {k: v for k, v in meta.items() if v not in (None, "")},
        "files": files,
        "links": {"html": article.get("url_public_html") or (LANDING.format(aid) if aid else None)},
        "_weblinks": weblinks,
        "_figshare": {
            "article_id": aid, "defined_type": article.get("defined_type"),
            "defined_type_name": article.get("defined_type_name"),
            "is_embargoed": bool(article.get("is_embargoed")), "embargo_type": article.get("embargo_type"),
            "embargo_date": article.get("embargo_date"), "embargo_reason": article.get("embargo_reason") or None,
            "is_confidential": bool(article.get("is_confidential")),
            "confidential_reason": article.get("confidential_reason") or None,
            "is_metadata_record": bool(article.get("is_metadata_record")),
            "metadata_reason": article.get("metadata_reason") or None,
            "download_disabled": bool(article.get("download_disabled")),
            "files_withheld": withheld, "link_only_files": len(link_only),
            "url_public_html": article.get("url_public_html"),
        },
    }


def datacite_from_article(article: dict) -> dict:
    """A minimal DataCite-shaped document from the article, used only when
    DataCite has no record of the DOI (``_source`` says so)."""
    creators = []
    for a in article.get("authors") or []:
        name = (a.get("full_name") or "").strip()
        if not name:
            continue
        c = {"name": name, "nameType": "Personal"}
        if a.get("first_name") and a.get("last_name"):
            c.update(givenName=a["first_name"], familyName=a["last_name"],
                     name=f"{a['last_name']}, {a['first_name']}")
        if a.get("orcid_id"):
            c["nameIdentifiers"] = [{"nameIdentifier": f"https://orcid.org/{a['orcid_id']}",
                                     "nameIdentifierScheme": "ORCID", "schemeUri": "https://orcid.org"}]
        creators.append(c)
    lic = article.get("license") or {}
    rights = [{"rights": lic.get("name"), **({"rightsUri": lic["url"]} if lic.get("url") else {})}] \
        if isinstance(lic, dict) and lic.get("name") else []
    pub = str(article.get("published_date") or "")
    doc = {
        "_source": "figshare_article",
        "doi": article.get("doi") or None,
        "titles": [{"title": article.get("title")}] if article.get("title") else [],
        "creators": creators,
        "publisher": {"name": "figshare"},
        "publicationYear": pub[:4] or None,
        "dates": [{"date": pub[:10], "dateType": "Issued"}] if pub else [],
        "rightsList": rights,
        "subjects": [{"subject": t} for t in article.get("tags") or []],
        "descriptions": ([{"description": article["description"], "descriptionType": "Abstract"}]
                         if article.get("description") else []),
        "version": str(article.get("version")) if article.get("version") is not None else None,
        "types": {"resourceTypeGeneral": "Dataset" if article.get("defined_type") == 3 else "Other",
                  "resourceType": article.get("defined_type_name")},
    }
    return {k: v for k, v in doc.items() if v not in (None, "", [])}


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
class FigshareClient(ZenodoClient):
    """Throttled, caching client for Figshare articles (see module doc)."""

    source = "figshare"
    has_container_api = False
    retry_statuses = RETRY_STATUSES
    cooldown_name = "figshare_not_before"
    requests_name = "figshare_requests"

    def __init__(self, cache_dir: Path, metadata_dir: Path | None = None, min_interval: float = MIN_INTERVAL,
                 offline: bool = False, max_per_minute: int | None = MAX_PER_MINUTE,
                 shared_state_dir: Path | None = None, shared_per_minute: int | None = MAX_PER_MINUTE,
                 token: str | None = None):
        super().__init__(cache_dir, metadata_dir=None, min_interval=min_interval, offline=offline,
                         max_per_minute=max_per_minute, shared_state_dir=shared_state_dir,
                         shared_per_minute=shared_per_minute, token=None)
        # metadata_dir of the base class holds legacy JSONs (read directly by
        # pipeline._cached_legacy and partition): Figshare article JSONs go
        # in article_dir instead, legacy JSONs in <cache>/legacy only.
        self.article_dir = Path(metadata_dir) if metadata_dir else None
        self.authenticated = bool(token)
        if token:
            register_secret(token)
            self.session.auth = FigshareToken(token)
        (cache_dir / "figshare").mkdir(parents=True, exist_ok=True)
        self._urls: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._urls_lock = threading.Lock()

    def _retryable(self, status: int, url: str) -> bool:
        if status == 403 and (urlparse(url).hostname or "").lower() == API_HOST:
            return True              # Figshare's abuse filter answers 403, not 429
        return status in self.retry_statuses

    # -- documents ----------------------------------------------------------
    def article(self, rid: str) -> dict | None:
        """Article JSON: pull metadata dir, then cache, then the API."""
        aid = article_id(rid)
        if not aid.isdigit():
            return None
        for p in ([self.article_dir / f"{aid}.json"] if self.article_dir else []) + \
                [self.cache_dir / "figshare" / f"{aid}.json"]:
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    logger.warning("Bad article file %s: %s", p, e)
        doc = self.get_json(f"{API}/articles/{aid}")
        if doc is not None:
            self._write(self.cache_dir / "figshare" / f"{aid}.json", doc)
        return doc

    def legacy(self, record_id: str) -> dict | None:
        p = self.cache_dir / "legacy" / f"{record_id}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        art = self.article(record_id)
        if art is None:
            return None
        doc = to_legacy(art)
        self._write(p, doc)
        return doc

    def datacite(self, record_id: str) -> dict | None:
        p = self.cache_dir / "datacite" / f"{record_id}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if self.offline:
            return None
        art = self.article(record_id)
        if art is None:
            return None
        doc = None
        if art.get("doi"):
            doc = self.get_json(DATACITE_URL + quote(str(art["doi"]), safe="/"),
                                accept="application/vnd.datacite.datacite+json")
            if doc is not None and not (doc.get("titles") or doc.get("creators")):
                doc = None
        if doc is None:
            doc = datacite_from_article(art)
        self._write(p, doc)
        return doc

    @staticmethod
    def _write(p: Path, doc: dict):
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(doc, fp)
        tmp.replace(p)

    # -- files ----------------------------------------------------------------
    def source_file_url(self, record_id: str, key: str) -> str | None:
        """download_url of the file with this key (from the legacy JSON)."""
        with self._urls_lock:
            urls = self._urls.get(record_id)
            if urls is not None:
                self._urls.move_to_end(record_id)
        if urls is None:
            leg = self.legacy(record_id) or {}
            urls = {f["key"]: (f.get("links") or {}).get("self") for f in leg.get("files") or [] if f.get("key")}
            with self._urls_lock:
                self._urls[record_id] = urls
                while len(self._urls) > 256:
                    self._urls.popitem(last=False)
        return urls.get(key)


def landing_url(record_id: str) -> str:
    return LANDING.format(article_id(record_id))
