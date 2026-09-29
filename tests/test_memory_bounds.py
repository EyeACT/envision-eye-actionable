"""Memory bounds of the survey roles (OOM kills of the process role on the
last records of the full pull: an 8 million member tar, and a fetch role
that had grown to 8 GB)."""

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _tar(path: Path, members: dict[str, bytes], mode: str = "w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_iter_tar_stream_keeps_no_member_list_and_extracts_the_current_member(tmp_path):
    from envision_eye_actionable.survey.archives import iter_tar_stream, open_tar_stream
    members = {f"d/{i:04d}.bin": bytes([i % 256]) * (i % 7 + 1) for i in range(600)}
    path = _tar(tmp_path / "a.tar.gz", members)
    seen = {}
    with open_tar_stream(path) as tf:
        for m in iter_tar_stream(tf):
            assert len(tf.members) <= 1          # 'for m in tf' holds all 600
            seen[m.name] = tf.extractfile(m).read()
    assert seen == members
    # the unfixed loop grows the list with the member count (the OOM)
    with open_tar_stream(path) as tf:
        for _m in tf:
            pass
        assert len(tf.members) == 600


def test_walker_caps_entries_but_counts_every_image(tmp_path):
    from envision_eye_actionable.survey.archives import RecordWalker
    members = {f"img/{i:03d}.png": b"\x89PNG\r\n\x1a\n" + bytes(8) for i in range(25)}
    members["vol/scan.mhd"] = b"ObjectType = Image\n"
    members["vol/scan.raw"] = bytes(16)
    members.update({f"notes/{i}.txt": b"x" for i in range(30)})
    path = _tar(tmp_path / "r.tar.gz", members)
    w = RecordWalker(scratch=tmp_path / "s", max_entries=10)
    w.add_file(path, "r.tar.gz")
    assert len(w.entries) == 10
    assert w.n_entries_over_cap == 16 and w.n_images_seen == 26
    assert w.containers[0].n_images == 26
    assert sum(w.image_ext_counts.values()) == 26
    # uncapped: the volume header still finds its data member although only
    # companion-like names are kept from the listing
    w2 = RecordWalker(scratch=tmp_path / "s2")
    w2.add_file(path, "r.tar.gz")
    hdr = [e for e in w2.entries if e.member == "vol/scan.mhd"]
    assert hdr and hdr[0].companion == "vol/scan.raw"
    assert w2.n_entries_over_cap == 0 and w2.n_images_seen == len(w2.entries) == 26


def test_over_recycle_limit():
    from envision_eye_actionable.survey import pipeline as pl
    cfg = SimpleNamespace(role_recycle_gb=4.0)
    assert pl.over_recycle_limit(cfg, measure=lambda: int(3.9e9)) is None
    assert pl.over_recycle_limit(cfg, measure=lambda: int(4.2e9)) == int(4.2e9)
    assert pl.over_recycle_limit(cfg, measure=lambda: None) is None
    assert pl.over_recycle_limit(SimpleNamespace(role_recycle_gb=0), measure=lambda: 10 ** 12) is None
    got = pl.role_memory_bytes()
    if sys.platform.startswith("linux"):
        assert isinstance(got, int) and got > 0
    else:
        assert got is None


def test_recycled_roles_restart_cleanly_without_using_up_restarts(tmp_path, monkeypatch):
    """fetch exits with EXIT_RECYCLE three times (max_role_restarts is 1):
    each is a clean restart, never a crash, and the pipeline ends with 0."""
    from envision_eye_actionable.survey import pipeline
    from test_pipeline import _cfg, _write_records
    meta = _write_records(tmp_path, {"3301": {"a.pdf": b"x"}})
    cfg = _cfg(tmp_path, meta, offline=True, max_role_restarts=1, restart_backoff_s=0.1, poll_s=0.1)
    script = tmp_path / "roles.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(Path(__file__).parent.parent)!r})\n"
        "from pathlib import Path\n"
        "from envision_eye_actionable.survey import pipeline as pl\n"
        f"state = Path({str(cfg.state_dir)!r})\n"
        "role = sys.argv[1]\n"
        "if role == 'monitor':\n"
        "    sys.exit(0)\n"
        "lock = pl.RoleLock(state, role)\n"
        "if role == 'fetch':\n"
        "    n_file = state / 'fetch_starts'\n"
        "    n = int(n_file.read_text()) if n_file.exists() else 0\n"
        "    n_file.write_text(str(n + 1))\n"
        "    time.sleep(0.5)\n"
        "    sys.exit(pl.EXIT_RECYCLE if n < 3 else 0)\n"
        "gone = 0\n"
        "while gone < 3:\n"
        "    gone = 0 if pl.role_alive(state, 'fetch') else gone + 1\n"
        "    time.sleep(0.1)\n"
        "sys.exit(0)\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    code = pipeline.run_pipeline(cfg, [], 0.5, base_cmd=[sys.executable, str(script)])
    ev = [json.loads(x) for x in (cfg.state_dir / "pipeline_events.jsonl").read_text().splitlines()]
    assert not [e for e in ev if e["event"] in ("role_crashed", "role_given_up")], ev
    status = json.loads((cfg.state_dir / "pipeline_status.json").read_text())
    assert status["clean_restarts"].get("fetch") == 3 and status["restarts"].get("fetch", 0) == 0
    assert code == 0


@pytest.mark.parametrize("role", ["process"])
def test_processor_exits_with_recycle_code_when_over_the_limit(tmp_path, monkeypatch, role):
    """The consumer checks its memory after each record and leaves with
    EXIT_RECYCLE (the record is finished first)."""
    from envision_eye_actionable.survey import pipeline as pl
    calls = []

    class P(pl.Processor):
        def __init__(self):                      # no survey, spool or model
            self.cfg = SimpleNamespace(ids=None, poll_s=0.01, role_recycle_gb=1.0)
            self._stop = False
            self.status = SimpleNamespace(update=lambda **k: None, count=lambda *a: None)
            self.spool = SimpleNamespace(ready_records=lambda: ["1", "2"])

        def process_one(self, rid):
            calls.append(rid)
            return "row"

        def survey_event(self, ev):
            calls.append(ev["event"])

    monkeypatch.setattr(pl, "role_memory_bytes", lambda: int(2e9))
    assert P()._run() == pl.EXIT_RECYCLE
    assert calls == ["1", "process_stop"]
