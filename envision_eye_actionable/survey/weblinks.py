"""Catalogue external links for records whose data is not on Zenodo.

Used for records with no files, or none that yielded a usable image. Links
come from the scrape's ``external_links``, the discovery enrichment
(``_weblinks`` with its scraper type, ``_dataset_links``) and the DataCite
``relatedIdentifiers``. Each link is cleaned (the scraper glued prose onto
some URLs), de-duplicated, stripped of self-references, categorised by
domain and flagged ``dataset_likely``. An optional HEAD check records the
HTTP status (10 s timeout, polite global rate limit, 429 Retry-After cooldown
per host, definitive answers cached across runs).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .zenodo import USER_AGENT, retry_after_seconds

# DOI prefixes of data repositories.
DATA_DOI_PREFIXES = {
    "10.5281": "Zenodo", "10.5061": "Dryad", "10.6084": "figshare", "10.17632": "Mendeley Data",
    "10.21227": "IEEE DataPort", "10.6019": "EMPIAR", "10.7910": "Harvard Dataverse",
    "10.13026": "PhysioNet", "10.7937": "TCIA", "10.17605": "OSF", "10.34740": "Kaggle",
    "10.60775": "FAIRhub", "10.18150": "Figshare institutional", "10.4121": "4TU.ResearchData",
    "10.15468": "GBIF", "10.25740": "Stanford Digital Repository", "10.57967": "Hugging Face",
}
DATASET_DOMAINS = (
    "zenodo.org", "figshare.com", "datadryad.org", "data.mendeley.com", "kaggle.com", "physionet.org",
    "osf.io", "dataverse", "ieee-dataport.org", "openneuro.org", "grand-challenge.org", "lila.science",
    "ebi.ac.uk", "medicaldecathlon.com", "synapse.org", "cancerimagingarchive.net", "dnanex.us",
    "proteinatlas.org", "ukbiobank.ac.uk", "fairhub.io", "aireadi.org", "openml.org", "data.gov",
    "datacite.org", "b2share", "archive.ics.uci.edu", "paperswithcode.com/dataset", "tianchi.aliyun.com",
    "dataportal", "4tu.nl", "gin.g-node.org", "brainlife.io",
)
CODE_DOMAINS = ("github.com", "gitlab", "bitbucket.org", "pypi.org", "sourceforge.net",
                "colab.research.google.com", "hub.docker.com", "cran.r-project.org", "readthedocs")
PAPER_DOMAINS = ("doi.org", "arxiv.org", "biorxiv.org", "medrxiv.org", "pubmed", "ncbi.nlm.nih.gov",
                 "sciencedirect.com", "springer", "nature.com", "wiley.com", "mdpi.com", "frontiersin.org",
                 "ieeexplore.ieee.org", "dl.acm.org", "plos.org", "elsevier", "tandfonline.com",
                 "researchgate.net", "arvojournals.org", "bmj.com", "jamanetwork.com", "thelancet.com",
                 "cell.com", "science.org", "oup.com", "sagepub.com", "semanticscholar.org",
                 "openreview.net", "aclanthology.org", "cvf.com", "neurips.cc", "iop.org", "osapublishing",
                 "optica.org", "spiedigitallibrary.org", "hindawi.com", "karger.com", "lww.com")
CLOUD_DOMAINS = ("drive.google.com", "docs.google.com", "dropbox.com", "onedrive", "1drv.ms",
                 "sharepoint.com", "box.com", "mega.nz", "pan.baidu.com", "wetransfer", "amazonaws.com",
                 "storage.googleapis.com", "blob.core.windows.net", "icloud.com", "jianguoyun", "weiyun")
# Word glued onto a URL by the scraper ("...releases/The", "...index.rstThe").
_GLUED_WORD = re.compile(r"(?<=[a-z0-9/_\-])(The|This|These|That|For|In|See|And|Available|Please|More|Our|We|It|"
                         r"All|To|Data|Dataset|Code|Paper|Link|Download|Here|Note|From|With)$")
_DATA_PATH = re.compile(r"(?:^|[/_\-.?=])(data|dataset|datasets|download|downloads|files?)(?:$|[/_\-.?=])", re.I)
_DATA_EXT = re.compile(r"\.(zip|tar|tgz|gz|7z|rar|h5|nii|dcm|png|jpe?g|tiff?)$", re.I)


def clean_url(url: str) -> str:
    """Remove scraper junk: U+FFFD tails, unbalanced '(' tails, glued words,
    trailing punctuation."""
    u = (url or "").strip()
    u = u.split("�", 1)[0]
    if "(" in u and u.count("(") > u.count(")"):
        u = u[: u.rfind("(")]
    for _ in range(2):
        u = u.rstrip(".,;:)]}>'\"")
        u = _GLUED_WORD.sub("", u)
    return u.rstrip(".,;:)]}>'\"")


def _doi_of(url: str) -> str | None:
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", url)
    return m.group(1).rstrip(".") if m else None


def categorize(url: str) -> tuple[str, str]:
    """(category, detail) with category in dataset_repository, code,
    paper_doi, institutional, cloud_drive, other."""
    p = urlparse(url if "://" in url else f"https://{url}")
    host = (p.netloc or "").lower()
    full = (host + p.path).lower()
    doi = _doi_of(url) if ("doi.org" in host or url.startswith("10.")) else None
    if doi:
        prefix = doi.split("/", 1)[0]
        if prefix in DATA_DOI_PREFIXES:
            return "dataset_repository", f"DOI {DATA_DOI_PREFIXES[prefix]}"
        return "paper_doi", "DOI"
    if any(d in host for d in CLOUD_DOMAINS):
        return "cloud_drive", host
    if "huggingface.co" in host:
        return ("dataset_repository", host) if "/datasets/" in p.path else ("code", host)
    if any(d in full for d in DATASET_DOMAINS):
        return "dataset_repository", host
    if any(d in host for d in CODE_DOMAINS):
        return "code", host
    if any(d in host for d in PAPER_DOMAINS):
        return "paper_doi", host
    if re.search(r"\.(edu|gov|ac\.[a-z]{2}|edu\.[a-z]{2}|gov\.[a-z]{2})$", host) or \
            re.search(r"(^|\.)(uni-|univ|university|hospital|institut|klinik)", host):
        return "institutional", host
    if host.endswith("github.io"):
        return "institutional", host  # project pages; dataset_likely decides
    return "other", host


def dataset_likely(url: str, category: str, scraper_type: str | None) -> bool:
    if category in ("dataset_repository", "cloud_drive"):
        return True
    if scraper_type in ("archive_download", "direct_file", "potential_download"):
        return True
    path = urlparse(url).path
    if _DATA_EXT.search(path):
        return True
    if category == "paper_doi":
        return False
    return bool(_DATA_PATH.search(path))


def collect_links(rid: str, scrape: dict, legacy: dict | None, datacite: dict | None) -> list[dict]:
    """All candidate links for a record, cleaned and de-duplicated."""
    legacy = legacy or {}
    raw: list[tuple[str, str, str | None]] = []  # (url, origin, scraper_type)
    for u in scrape.get("external_links") or []:
        raw.append((u, "scrape.external_links", None))
    for w in legacy.get("_weblinks") or []:
        if isinstance(w, dict) and w.get("url"):
            raw.append((w["url"], "discovery._weblinks", w.get("type")))
        elif isinstance(w, str):
            raw.append((w, "discovery._weblinks", None))
    for d in legacy.get("_dataset_links") or []:
        if isinstance(d, dict):
            u = d.get("url") or d.get("identifier") or d.get("link")
            if u:
                raw.append((u, "discovery._dataset_links", d.get("type") or "data_platform"))
        elif isinstance(d, str):
            raw.append((d, "discovery._dataset_links", "data_platform"))
    for r in (datacite or {}).get("relatedIdentifiers") or []:
        v, t = r.get("relatedIdentifier"), r.get("relatedIdentifierType")
        if not v or r.get("relationType") in ("IsVersionOf", "HasVersion", "IsNewVersionOf",
                                               "IsPreviousVersionOf", "IsPartOf"):
            continue
        if t == "DOI":
            v = f"https://doi.org/{v}"
        elif t != "URL":
            continue
        raw.append((v, f"datacite.relatedIdentifiers ({r.get('relationType')})", None))

    self_ids = {rid, str(legacy.get("conceptrecid") or "")}
    out, seen = [], {}
    for url, origin, stype in raw:
        cleaned = clean_url(url)
        if not re.match(r"^(https?|ftp)://", cleaned):
            if cleaned.startswith("10."):
                cleaned = f"https://doi.org/{cleaned}"
            elif cleaned.startswith("www."):
                cleaned = f"https://{cleaned}"
            else:
                continue
        norm = cleaned.lower().rstrip("/").replace("http://", "https://")
        if _is_self(norm, self_ids):
            continue
        if norm in seen:
            rec = out[seen[norm]]
            if origin not in rec["origin"]:
                rec["origin"] += f"; {origin}"
            rec["scraper_type"] = rec["scraper_type"] or stype
            continue
        cat, detail = categorize(cleaned)
        seen[norm] = len(out)
        out.append({"url": cleaned, "url_raw": url if url != cleaned else "", "origin": origin,
                    "scraper_type": stype, "category": cat, "category_detail": detail})
    for rec in out:
        rec["dataset_likely"] = dataset_likely(rec["url"], rec["category"], rec["scraper_type"])
    return out


def _is_self(norm: str, ids: set[str]) -> bool:
    for i in ids:
        if not i:
            continue
        if re.search(rf"zenodo\.org/(records?|deposit)/{i}(?:$|[/?#])", norm):
            return True
        if norm.endswith(f"10.5281/zenodo.{i}"):
            return True
    return False


def _is_zenodo_url(url: str) -> bool:
    """Zenodo pages, and doi.org Zenodo DOIs (they redirect to Zenodo)."""
    p = urlparse(url)
    host = (p.netloc or "").lower()
    return host == "zenodo.org" or host.endswith(".zenodo.org") or (
        host.endswith("doi.org") and p.path.lower().startswith("/10.5281/"))


def _definitive(res: dict) -> bool:
    """Cache only answers that will not change on a retry: 2xx, 3xx and 4xx
    other than 408 (timeout) and 429 (rate limit). 5xx and transport errors
    are checked again on the next run."""
    s = res.get("http_status")
    return s is not None and 200 <= s < 500 and s not in (408, 429)


class LinkChecker:
    """HEAD (falling back to a streamed GET) with a global rate limit, a 429
    cooldown per host, and a JSON cache of definitive answers so reruns do
    not re-hit the same hosts.

    Zenodo links (and Zenodo DOIs) go through the survey's ZenodoClient
    throttle, so they share its rate limit and 429 cooldown with the
    metadata calls and downloads.
    """

    def __init__(self, cache_path: Path, min_interval: float = 1.0, timeout: float = 10.0,
                 zenodo=None, max_host_wait: float = 120.0):
        self.cache_path = cache_path
        self.min_interval = min_interval
        self.timeout = timeout
        self.zenodo = zenodo
        self.max_host_wait = max_host_wait
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._lock = threading.Lock()
        self._last = 0.0
        self._host_not_before: dict[str, float] = {}
        self.cache: dict = {}
        self._dirty = False
        self._saved_at = time.monotonic()
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                # Older caches also froze 429 / 5xx / timeouts: drop those.
                self.cache = {u: r for u, r in cached.items() if isinstance(r, dict) and _definitive(r)}
            except (OSError, ValueError, AttributeError):
                self.cache = {}

    def _wait_turn(self, url: str, host: str) -> bool:
        """Wait for the rate limit. False when the host is cooling down for
        longer than ``max_host_wait`` (the link is then not checked now)."""
        if self.zenodo is not None and _is_zenodo_url(url):
            self.zenodo.throttle()
            return True
        with self._lock:
            now = time.monotonic()
            cool = self._host_not_before.get(host, 0.0) - now
            if cool > self.max_host_wait:
                return False
            wait = max(self.min_interval - (now - self._last), cool)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        return True

    def _cooldown(self, url: str, host: str, r) -> float:
        secs = retry_after_seconds(r.headers.get("Retry-After"), 60.0, 429)
        if self.zenodo is not None and _is_zenodo_url(url):
            self.zenodo.backoff(secs)
        else:
            with self._lock:
                self._host_not_before[host] = max(self._host_not_before.get(host, 0.0),
                                                  time.monotonic() + secs)
        return secs

    def _account(self, r, first_counted: bool = True):
        """Count every zenodo.org request of a response (redirect hops
        included) into the shared Zenodo budget and pass each Zenodo answer
        to ZenodoClient.observe, so a low X-RateLimit-Remaining pauses the
        producer as well. The first request was already throttled when
        ``first_counted``."""
        if self.zenodo is None:
            return
        hops = list(getattr(r, "history", None) or []) + [r]
        for i, h in enumerate(hops):
            if not _is_zenodo_url(getattr(h, "url", "") or ""):
                continue
            if i > 0 or not first_counted:
                self.zenodo.throttle()          # late, but it takes the slot the hop used
            self.zenodo.observe(h)

    def _request(self, url: str):
        zen = self.zenodo is not None and _is_zenodo_url(url)
        r = self.session.head(url, allow_redirects=True, timeout=self.timeout)
        self._account(r, first_counted=zen)
        if r.status_code in (403, 405, 501) or r.status_code >= 500:
            if zen:
                self.zenodo.throttle()          # the fallback GET is a request of its own
            r = self.session.get(url, allow_redirects=True, timeout=self.timeout, stream=True)
            r.close()
            self._account(r, first_counted=zen)
        return r

    def check(self, url: str) -> dict:
        if url in self.cache:
            return self.cache[url]
        host = (urlparse(url).netloc or "").lower()
        res = {"http_status": None, "final_url": "", "content_type": "", "content_length": None,
               "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": ""}
        for attempt in range(2):          # one retry after a 429
            if not self._wait_turn(url, host):
                res["error"] = "host rate-limited (429); not checked"
                break
            try:
                r = self._request(url)
            except requests.RequestException as e:
                res.update(http_status=None, error=type(e).__name__)
                break
            res.update(http_status=r.status_code, final_url=r.url if r.url != url else "",
                       content_type=r.headers.get("Content-Type", "")[:80],
                       content_length=int(r.headers["Content-Length"]) if
                       r.headers.get("Content-Length", "").isdigit() else None,
                       error="")
            if r.status_code != 429:
                break
            secs = self._cooldown(url, host, r)
            res["error"] = f"rate limited (429, Retry-After {secs:.0f} s)"
            if secs > self.max_host_wait:
                break
        if _definitive(res):
            self.cache[url] = res
            self._dirty = True
        return res

    def save(self, min_interval_s: float = 0.0):
        """Write the cache when it changed, and (``min_interval_s``) at most
        that often: a 30,000-record run would otherwise rewrite a large
        cache after every record."""
        if not self._dirty or time.monotonic() - self._saved_at < min_interval_s:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache), encoding="utf-8")
        tmp.replace(self.cache_path)
        self._dirty = False
        self._saved_at = time.monotonic()
