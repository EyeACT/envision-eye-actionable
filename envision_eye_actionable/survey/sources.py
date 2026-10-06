"""Which repository a survey run reads (Zenodo or Figshare) and its client.

One run reads one source. ``--source auto`` (the default) takes it from
the scrape: Figshare when every record has ``source: figshare``, Zenodo
when none has, an error when the scrape mixes them (run each source into
its own out dir; the workbook merges them with --results-dir).
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCES = ("zenodo", "figshare")

# Request pacing per source: (requests per minute, seconds between requests).
DEFAULT_RATES = {"zenodo": (120, 0.5), "figshare": (60, 1.0)}
DEFAULT_METADATA_DIRS = {"zenodo": Path("./data/metadata/zenodo"),
                         "figshare": Path("./data/metadata/figshare_full")}


def detect_source(scrape_path: Path) -> str:
    """zenodo or figshare from the records of a scrape (see module doc)."""
    with open(scrape_path, encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        data = data.get("results") or data.get("records") or []
    kinds = {"figshare" if str(r.get("source") or "").lower() == "figshare"
             or str(r.get("source_id") or "").startswith("figshare-") else "zenodo" for r in data}
    if len(kinds) > 1:
        raise ValueError(f"{scrape_path} mixes Zenodo and Figshare records: survey each source on its own "
                         "(--source with a scrape of that source only)")
    return kinds.pop() if kinds else "zenodo"


def make_client(cfg, *, shared_state_dir=None, offline: bool | None = None, with_token: bool = True):
    """The metadata and file client of ``cfg.source`` (ZenodoClient or
    FigshareClient), with the run's pacing, cache dir and token."""
    from .zenodo import ZenodoClient
    from .zenodo import load_token as zenodo_token
    cache = cfg.metadata_cache_dir or (cfg.out_dir / "cache")
    kw = dict(min_interval=cfg.zenodo_interval, offline=cfg.offline if offline is None else offline,
              max_per_minute=cfg.zenodo_per_minute, shared_state_dir=shared_state_dir,
              shared_per_minute=cfg.zenodo_shared_per_minute)
    if getattr(cfg, "source", "zenodo") == "figshare":
        from .figshare import FigshareClient
        from .figshare import load_token as figshare_token
        return FigshareClient(cache, cfg.metadata_dir, token=figshare_token(cfg.token_file) if with_token else None,
                              **kw)
    return ZenodoClient(cache, cfg.metadata_dir, token=zenodo_token(cfg.token_file) if with_token else None, **kw)


def landing_url(record_id: str) -> str:
    """Landing page of a record id when the scrape gives none."""
    from .zenodo import record_source
    if record_source(record_id) == "figshare":
        from .figshare import landing_url as fl
        return fl(record_id)
    return f"https://zenodo.org/records/{record_id}"
