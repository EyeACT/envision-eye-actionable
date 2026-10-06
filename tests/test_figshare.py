"""Offline tests of the Figshare source (survey/figshare.py, sources.py):
id namespacing, article to legacy conversion, access handling, the client
(cache, DataCite fallback, token scope, retries), range-only zip reading,
CMDS repository naming, and one end-to-end pipeline run (no network)."""

from __future__ import annotations

import hashlib
import json

import pytest
import requests
from test_pipeline import _cfg, _OnnxStub, _run_roles
from test_survey import _HttpResp, _png_bytes, _zip_bytes

SECRET = "figSECRET" + "z" * 30
DL = "https://ndownloader.figshare.com/files/{}"


def _article(aid, files: dict, **kw) -> dict:
    """A Figshare article JSON; ``files`` maps name -> bytes (file ids from 100)."""
    doc = {
        "id": aid, "title": f"Fundus set {aid}", "doi": f"10.6084/m9.figshare.{aid}.v1", "version": 1,
        "description": "<p>Colour fundus photographs.</p>", "defined_type": 3, "defined_type_name": "dataset",
        "url_public_html": f"https://figshare.com/articles/dataset/Fundus_set/{aid}",
        "published_date": "2024-05-01T10:00:00Z", "created_date": "2024-05-01T10:00:00Z",
        "authors": [{"full_name": "Jane Doe", "first_name": "Jane", "last_name": "Doe", "orcid_id": "0000-0002-1825-0097"}],
        "tags": ["fundus", "retina"], "categories": [{"title": "Ophthalmology"}],
        "license": {"value": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
        "is_embargoed": False, "embargo_type": "file", "embargo_date": None, "is_confidential": False,
        "is_metadata_record": False, "download_disabled": False, "related_materials": [],
        "files": [{"id": 100 + i, "name": name, "size": len(data), "is_link_only": False,
                   "download_url": DL.format(100 + i), "computed_md5": hashlib.md5(data).hexdigest(),
                   "supplied_md5": "", "mimetype": "application/octet-stream"}
                  for i, (name, data) in enumerate(files.items())],
    }
    doc.update(kw)
    return doc


class _FakeFigshare:
    """ndownloader file URLs (Range included), DataCite (404 by default) and
    the article API. Any Zenodo URL fails the test."""

    def __init__(self, blobs: dict, articles: dict | None = None, datacite: dict | None = None):
        self.blobs, self.articles, self.datacite = blobs, articles or {}, datacite or {}
        self.calls = []

    def get(self, url, headers=None, stream=False, timeout=None):
        headers = headers or {}
        self.calls.append((url, headers.get("Range")))
        assert "zenodo.org" not in url, url
        if url.startswith("https://ndownloader.figshare.com/files/"):
            data = self.blobs.get(int(url.rsplit("/", 1)[1]))
            if data is None:
                return _HttpResp(404)
            rng = headers.get("Range")
            if rng:
                a, _, b = rng.split("=")[1].partition("-")
                a, b = int(a), (int(b) if b else len(data) - 1)
                return _HttpResp(206, data[a:b + 1], {"Content-Range": f"bytes {a}-{b}/{len(data)}"})
            return _HttpResp(200, data)
        if url.startswith("https://api.datacite.org/"):
            doi = url.split("+json/", 1)[1]
            return _HttpResp(200, json.dumps(self.datacite[doi]).encode()) if doi in self.datacite else _HttpResp(404)
        if url.startswith("https://api.figshare.com/v2/articles/"):
            aid = url.rsplit("/", 1)[1]
            return _HttpResp(200, json.dumps(self.articles[aid]).encode()) if aid in self.articles else _HttpResp(404)
        return _HttpResp(404)


@pytest.fixture(autouse=True)
def _no_real_token(tmp_path, monkeypatch):
    """Never pick up a token from the machine running the tests."""
    from envision_eye_actionable.survey import figshare
    monkeypatch.delenv("FIGSHARE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(figshare, "DEFAULT_TOKEN_FILE", tmp_path / "no_such_dir" / "figshare_token")


def _fclient(tmp_path, session=None, article_dir=None, **kw):
    from envision_eye_actionable.survey.figshare import FigshareClient
    c = FigshareClient(tmp_path / "cache", article_dir, min_interval=0, max_per_minute=None,
                       shared_per_minute=None, **kw)
    if session is not None:
        c.session = session
    return c


# ---------------------------------------------------------------------------
# ids and scrape
# ---------------------------------------------------------------------------
def test_scrape_namespaces_figshare_ids_and_leaves_zenodo_alone(tmp_path):
    from envision_eye_actionable.survey import sources, spool, zenodo
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"source": "figshare", "source_id": "14798004"},
                             {"source": "figshare", "source_id": "figshare-5"}]), encoding="utf-8")
    recs = zenodo.load_scrape(p)
    assert [r["source_id"] for r in recs] == ["figshare-14798004", "figshare-5"]
    assert recs[0]["figshare_id"] == "14798004"
    assert sources.detect_source(p) == "figshare"
    z = tmp_path / "z.json"
    z.write_text(json.dumps([{"source": "zenodo", "source_id": "14798004"}, {"id": 7}]), encoding="utf-8")
    assert [r["source_id"] for r in zenodo.load_scrape(z)] == ["14798004", "7"]
    assert sources.detect_source(z) == "zenodo"
    m = tmp_path / "m.json"
    m.write_text(json.dumps([{"source": "figshare", "source_id": "1"}, {"source_id": "2"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        sources.detect_source(m)
    for ok in ("123", "figshare-123"):
        assert spool.valid_record_id(ok) and zenodo.is_record_id(ok)
    for bad in ("figshare-", "figshare-1a", "../1", "figshare-../1", "zenodo-1", "", "1" * 21):
        assert not spool.valid_record_id(bad), bad
    assert zenodo.record_source("figshare-9") == "figshare" and zenodo.record_source("9") == "zenodo"
    assert sources.landing_url("figshare-9") == "https://figshare.com/articles/9"


# ---------------------------------------------------------------------------
# article -> legacy
# ---------------------------------------------------------------------------
def test_open_article_becomes_a_legacy_record(tmp_path):
    from envision_eye_actionable.survey import figshare, zenodo
    a = _article(11, {"a.png": b"x" * 10, "b.zip": b"y" * 20})
    a["files"].append({"id": 300, "name": "a.png", "size": 5, "is_link_only": False,
                       "download_url": DL.format(300), "computed_md5": "", "supplied_md5": "ABCDEF" * 5 + "01"})
    a["files"].append({"id": 301, "name": "link", "size": 0, "is_link_only": True,
                       "download_url": "https://physionet.org/content/x/"})
    a["related_materials"] = [{"identifier": "10.1038/s41597-022-01388-1", "identifier_type": "DOI",
                               "relation": "IsSupplementTo", "link": None}]
    leg = figshare.to_legacy(a)
    files = zenodo.record_files(leg)
    assert [f["key"] for f in files] == ["a.png", "b.zip", "300_a.png"]          # repeated name gets the file id
    assert files[0]["url"] == DL.format(100) and files[0]["md5"] == hashlib.md5(b"x" * 10).hexdigest()
    assert files[2]["md5"] == ("abcdef" * 5 + "01")                             # supplied md5 used, lowercased
    assert leg["metadata"]["access_right"] == "open"
    assert leg["metadata"]["license"] == {"id": "cc-by-4.0"}
    assert leg["metadata"]["creators"] == [{"name": "Doe, Jane", "orcid": "0000-0002-1825-0097"}]
    assert "Ophthalmology" in leg["metadata"]["keywords"]
    urls = [w["url"] for w in leg["_weblinks"]]
    assert urls == ["https://physionet.org/content/x/", "https://doi.org/10.1038/s41597-022-01388-1"]
    assert leg["_figshare"]["link_only_files"] == 1 and leg["id"] == "figshare-11"


@pytest.mark.parametrize("flags,access", [
    ({"is_embargoed": True, "embargo_type": "article", "embargo_date": "2027-01-01T00:00:00Z"}, "embargoed"),
    ({"is_embargoed": True, "embargo_type": "file"}, "embargoed"),
    ({"is_confidential": True, "confidential_reason": "patient data"}, "closed"),
    ({"download_disabled": True}, "restricted"),
])
def test_non_public_articles_list_no_files(flags, access):
    from envision_eye_actionable.survey import figshare
    leg = figshare.to_legacy(_article(12, {"a.png": b"x"}, **flags))
    assert leg["metadata"]["access_right"] == access
    assert leg["files"] == [] and leg["_figshare"]["files_withheld"] == 1
    if access == "embargoed" and flags.get("embargo_date"):
        assert leg["metadata"]["embargo_date"] == "2027-01-01"


def test_metadata_record_is_open_without_files():
    from envision_eye_actionable.survey import figshare
    a = _article(13, {}, is_metadata_record=True, metadata_reason="Data held at PhysioNet")
    a["files"] = [{"id": 1, "name": "x", "is_link_only": True, "download_url": "https://physionet.org/x"}]
    leg = figshare.to_legacy(a)
    assert leg["metadata"]["access_right"] == "open" and leg["files"] == []
    assert leg["_weblinks"][0]["url"] == "https://physionet.org/x"


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
def test_client_reads_the_pull_dir_caches_legacy_and_resolves_file_urls(tmp_path):
    from envision_eye_actionable.survey import pipeline
    adir = tmp_path / "articles"
    adir.mkdir()
    (adir / "21.json").write_text(json.dumps(_article(21, {"a.zip": b"z" * 7})), encoding="utf-8")
    fake = _FakeFigshare({})
    c = _fclient(tmp_path, fake, article_dir=adir)
    assert c.metadata_dir is None and c.article_dir == adir     # article JSONs are never read as legacy JSONs
    leg = c.legacy("figshare-21")
    assert leg["files"][0]["key"] == "a.zip"
    assert (tmp_path / "cache" / "legacy" / "figshare-21.json").is_file()
    assert pipeline._cached_legacy(c, "figshare-21")["files"][0]["key"] == "a.zip"
    assert c.source_file_url("figshare-21", "a.zip") == DL.format(100)
    assert c.source_file_url("figshare-21", "nope.zip") is None
    assert fake.calls == []                                     # all from the pull dir
    # not in the pull dir: fetched from the API and cached under cache/figshare
    fake.articles["22"] = _article(22, {"b.png": b"p"})
    assert c.legacy("figshare-22")["files"][0]["key"] == "b.png"
    assert (tmp_path / "cache" / "figshare" / "22.json").is_file()
    assert c.legacy("figshare-23") is None


def test_datacite_from_the_api_else_built_from_the_article(tmp_path):
    adir = tmp_path / "articles"
    adir.mkdir()
    for aid in (31, 32):
        (adir / f"{aid}.json").write_text(json.dumps(_article(aid, {})), encoding="utf-8")
    dc = {"doi": "10.6084/m9.figshare.31.v1", "titles": [{"title": "From DataCite"}],
          "creators": [{"name": "Doe, Jane", "nameType": "Personal"}], "publisher": "figshare"}
    fake = _FakeFigshare({}, datacite={"10.6084/m9.figshare.31.v1": dc})
    c = _fclient(tmp_path, fake, article_dir=adir)
    assert c.datacite("figshare-31")["titles"][0]["title"] == "From DataCite"
    built = c.datacite("figshare-32")
    assert built["_source"] == "figshare_article" and built["titles"][0]["title"] == "Fundus set 32"
    assert built["creators"][0]["nameIdentifiers"][0]["nameIdentifier"].endswith("0000-0002-1825-0097")
    assert built["rightsList"][0]["rights"] == "CC BY 4.0"
    n = len(fake.calls)
    c.datacite("figshare-31")
    c.datacite("figshare-32")
    assert len(fake.calls) == n                                  # cached, both kinds


def test_token_goes_to_the_figshare_api_only_and_is_redacted(tmp_path, monkeypatch):
    from envision_eye_actionable.survey import figshare, zenodo
    monkeypatch.setenv("FIGSHARE_ACCESS_TOKEN", SECRET)
    tok = figshare.load_token(None)
    assert tok == SECRET and zenodo.redact(f"x {SECRET} y") == f"x {zenodo.REDACTED} y"
    auth = figshare.FigshareToken(tok)
    assert SECRET not in repr(auth)
    for url, sent in (("https://api.figshare.com/v2/articles/1", True),
                      ("https://ndownloader.figshare.com/files/1", False),
                      ("https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/1/a.zip", False),
                      ("https://api.datacite.org/dois/x", False)):
        req = requests.Request("GET", url, headers={"Authorization": "token stale"}).prepare()
        auth(req)
        assert (req.headers.get("Authorization") == f"token {SECRET}") is sent, url
        if not sent:
            assert "Authorization" not in req.headers
    monkeypatch.delenv("FIGSHARE_ACCESS_TOKEN")
    assert figshare.load_token(None) is None
    with pytest.raises(ValueError):
        figshare.load_token(tmp_path / "missing")
    default = tmp_path / "cfg" / "figshare_token"
    default.parent.mkdir()
    default.write_text("default-token-value\n", encoding="utf-8")
    monkeypatch.setattr(figshare, "DEFAULT_TOKEN_FILE", default)
    assert figshare.load_token(None) == "default-token-value"
    assert figshare.load_token(None, use_default=False) is None
    other = tmp_path / "t2"
    other.write_text("explicit-token-value\n", encoding="utf-8")
    assert figshare.load_token(other) == "explicit-token-value"
    monkeypatch.setenv("FIGSHARE_ACCESS_TOKEN", "env-token-value")
    assert figshare.load_token(other) == "env-token-value"


def test_403_from_the_api_is_retried_and_from_files_is_not(tmp_path, monkeypatch):
    from envision_eye_actionable.survey import zenodo
    monkeypatch.setattr(zenodo.time, "sleep", lambda s: None)

    class S:
        def __init__(self):
            self.n = 0

        def get(self, url, headers=None, stream=False, timeout=None):
            self.n += 1
            if "api.figshare.com" in url and self.n == 1:
                return _HttpResp(403)
            return _HttpResp(403) if "ndownloader" in url else _HttpResp(200, b"{}")

    s = S()
    c = _fclient(tmp_path, s)
    assert c.get("https://api.figshare.com/v2/articles/1").status_code == 200 and s.n == 2
    with pytest.raises(zenodo.RemoteError):
        c.get("https://ndownloader.figshare.com/files/1")
    assert s.n == 3


def test_zips_are_listed_and_read_by_range_without_a_container_api(tmp_path):
    import random

    from envision_eye_actionable.survey import remote
    png = _png_bytes(size=(40, 30))
    z = _zip_bytes({f"img/{i}.png": png for i in range(6)})
    adir = tmp_path / "articles"
    adir.mkdir()
    (adir / "41.json").write_text(json.dumps(_article(41, {"set.zip": z})), encoding="utf-8")
    fake = _FakeFigshare({100: z})
    c = _fclient(tmp_path, fake, article_dir=adir)
    assert remote.file_url("figshare-41", "set.zip", c) == DL.format(100)
    assert remote.file_url("41", "set.zip").startswith("https://zenodo.org/api/records/41/")
    lst = remote.list_zip(c, "figshare-41", "set.zip", len(z))
    assert lst.source == "range" and lst.error is None and len(lst.images) == 6
    res = remote.sample_remote_zips(c, "figshare-41", [{"key": "set.zip", "size": len(z)}], tmp_path / "t", 3,
                                    random.Random(1))
    assert len(res.fetched) == 3
    assert all(u.startswith("https://ndownloader.figshare.com/files/100") for u, _ in fake.calls)
    assert all(r for _, r in fake.calls)                         # every request a range read


# ---------------------------------------------------------------------------
# CMDS and weblinks
# ---------------------------------------------------------------------------
def test_cmds_name_figshare_and_keep_zenodo_documents_unchanged():
    from envision_eye_actionable.survey import cmds, figshare
    a = _article(51, {}, is_embargoed=True, embargo_date="2027-02-01T00:00:00Z")
    leg = figshare.to_legacy(a)
    rec = {"source_id": "figshare-51", "source": "figshare", "url": a["url_public_html"]}
    summary = {"present_classes": [], "n_files": 0}
    dd, prov = cmds.build_dataset_description(rec, leg, figshare.datacite_from_article(a), summary)
    text = json.dumps(prov)
    assert "zenodo" not in text.lower() and "figshare_article (no DataCite record)" in text
    assert "Figshare" in dd["datasetDeIdentLevel"]["deIdentDetails"]
    assert "Zenodo" not in json.dumps(dd)
    assert dd["identifier"]["identifierValue"] == "10.6084/m9.figshare.51.v1"
    dsd = cmds.build_structure_description({}, [], {"n_images": 0, "n_classified": 0, "repository": "Figshare"})
    assert "Figshare record" in dsd["metadataFileList"][0]["metadataFileDescription"]
    # a Zenodo record: the same texts as before
    zrec = {"source_id": "77", "url": "https://zenodo.org/records/77"}
    zdd, zprov = cmds.build_dataset_description(zrec, {"metadata": {"access_right": "embargoed", "title": "t"}},
                                                None, summary)
    assert zprov["title"] == "zenodo_legacy" and "on Zenodo" in zdd["datasetDeIdentLevel"]["deIdentDetails"]


def test_weblinks_drop_links_to_the_article_itself():
    from envision_eye_actionable.survey import weblinks
    leg = {"_weblinks": [{"url": "https://figshare.com/articles/dataset/Fundus_set/61"},
                         {"url": "https://doi.org/10.6084/m9.figshare.61.v2"},
                         {"url": "https://springernature.figshare.com/articles/dataset/x/61/1"},
                         {"url": "https://figshare.com/articles/dataset/other/610"},
                         {"url": "https://physionet.org/content/x/"}]}
    links = weblinks.collect_links("figshare-61", {}, leg, None)
    assert [x["url"] for x in links] == ["https://figshare.com/articles/dataset/other/610",
                                         "https://physionet.org/content/x/"]


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_pipeline_surveys_a_figshare_scrape(tmp_path, monkeypatch):
    pytest.importorskip("jsonschema")
    from envision_eye_actionable.survey import zenodo
    monkeypatch.setattr(zenodo.time, "sleep", lambda s: None)
    bright = _png_bytes(size=(64, 48), color=(200, 120, 60))
    z = _zip_bytes({f"fundus/{i}.png": bright for i in range(5)})
    arts = {
        71: _article(71, {"fundus.zip": z, "extra.png": bright}),
        72: _article(72, {"a.png": bright}, is_embargoed=True, embargo_date="2027-01-01T00:00:00Z"),
        73: _article(73, {"a.png": bright}, is_confidential=True),
        74: _article(74, {}, is_metadata_record=True),
        75: _article(75, {"table.csv": b"a,b\n1,2\n"}),
    }
    meta = tmp_path / "articles"
    meta.mkdir()
    blobs = {}
    for aid, a in arts.items():
        (meta / f"{aid}.json").write_text(json.dumps(a), encoding="utf-8")
    # file ids restart at 100 per article: give each article its own ids
    for aid, a in arts.items():
        for f in a["files"]:
            f["id"] = aid * 1000 + f["id"]
            f["download_url"] = DL.format(f["id"])
        (meta / f"{aid}.json").write_text(json.dumps(a), encoding="utf-8")
    blobs[71100], blobs[71101] = z, bright
    blobs[75100] = b"a,b\n1,2\n"
    scrape = [{"source": "figshare", "source_id": str(aid), "title": a["title"], "url": a["url_public_html"]}
              for aid, a in arts.items()]
    (tmp_path / "scrape.json").write_text(json.dumps(scrape), encoding="utf-8")
    cfg = _cfg(tmp_path, meta, triage_cap=0, source="figshare")
    session = _FakeFigshare(blobs)
    out, prod, proc = _run_roles(cfg, session, monkeypatch, onnx=_OnnxStub)
    assert out == {"fetch": 0, "process": 0}
    assert prod.zenodo.source == "figshare" and not prod.zenodo.has_container_api
    rows = {}
    for line in (cfg.out_dir / "survey_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["record_id"]] = r
    assert set(rows) == {f"figshare-{a}" for a in arts}
    assert all(r["source"] == "figshare" for r in rows.values())
    ok = rows["figshare-71"]
    assert ok["status"] == "ok", ok.get("error")
    assert ok["n_classified"] == 6 and "remote_zip" in ok["sampling_mode"]
    assert rows["figshare-72"]["status"] == "restricted" and rows["figshare-72"]["access_right"] == "embargoed"
    assert rows["figshare-73"]["status"] == "restricted" and rows["figshare-73"]["access_right"] == "closed"
    assert rows["figshare-74"]["status"] == "no_files"
    assert rows["figshare-75"]["status"] == "no_image_files"
    dd = json.loads((cfg.out_dir / "cmds" / "figshare-71" / "dataset_description.json").read_text())
    assert dd["identifier"]["identifierValue"] == "10.6084/m9.figshare.71.v1"
    assert "zenodo" not in json.dumps(dd).lower()
    assert (cfg.out_dir / "cmds" / "figshare-72" / "dataset_description.json").is_file()
    assert not any("zenodo.org" in u for u, _ in session.calls)
    assert (cfg.state_dir / "figshare_requests").is_file() or cfg.zenodo_shared_per_minute is None
