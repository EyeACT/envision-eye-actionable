"""Backfill of eye-positive records refused for room (--keep-max-gb reached
or the disk guard): listed most eye images first within a byte budget, and
fetched again by the producer in that order."""

import json

from test_pipeline import _FakeZenodo, _Multi, _cfg, _png_bytes, _write_records


def _row(rid, n_oct, gb, reason="--keep-max-gb 200 reached (keep dir 200.0 GB, record 1.00 GB)", **kw):
    r = {"record_id": rid, "status": "ok", "n_OCT": n_oct, "n_classified": n_oct, "n_eye_images": n_oct,
         "kept": False, "kept_reason": reason, "spool_bytes": int(gb * 1e9)}
    r.update(kw)
    return r


def test_refused_for_room():
    from envision_eye_actionable.survey.keep import refused_for_room
    assert refused_for_room(_row("1", 3, 1.0))
    assert refused_for_room(_row("1", 3, 1.0, reason="disk: keeping would leave less than ..."))
    assert not refused_for_room(_row("1", 3, 1.0, reason="no eye image"))
    assert not refused_for_room({"kept": True, "kept_reason": ""})
    assert not refused_for_room({"status": "ok"}) and not refused_for_room(None)


def test_backfill_refused_most_eye_images_first_within_budget(tmp_path, capsys):
    from envision_eye_actionable.survey import cli
    from envision_eye_actionable.survey.keep import backfill_ids
    out = tmp_path / "out"
    out.mkdir()
    rows = [_row("4401", 10, 5.0), _row("4402", 900, 60.0), _row("4403", 300, 30.0),
            _row("4404", 50, 1.0, reason="disk: keeping would leave less than --disk-floor-gb 80"),
            _row("4405", 0, 2.0, reason="no eye image"),
            _row("4406", 200, 3.0),
            {"record_id": "4406", "status": "ok", "n_OCT": 2, "kept": True},          # kept since: not listed
            _row("4407", 400, 0.0),                                                  # nothing fetched
            _row("4408", 700, 4.0),
            {"record_id": "4408", "status": "started"},                              # unfinished line after it
            {"record_id": "4409", "status": "ok", "n_IR": 2, "n_classified": 2}]     # before retention
    (out / "survey_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ids, s = backfill_ids(out / "survey_results.jsonl", include_refused=True)
    assert ids == ["4409", "4402", "4408", "4403", "4404", "4401"]
    ids, s = backfill_ids(out / "survey_results.jsonl", include_refused=True, budget_bytes=40e9)
    # 4402 (60 GB) does not fit; the next ones do while the budget lasts
    assert ids == ["4409", "4408", "4403", "4404", "4401"]
    assert s["left_out"] == 1 and s["left_out_ids"][0]["record_id"] == "4402"
    assert s["selected_bytes_gb"] == 40.0 and s["selected_eye_images"] == 700 + 300 + 50 + 10
    assert s["left_out_eye_images"] == 900
    # default: pre-retention rows only, as before
    assert backfill_ids(out / "survey_results.jsonl")[0] == ["4409"]
    f = tmp_path / "ids.txt"
    assert cli.main(["keep-backfill-ids", "--out-dir", str(out), "--output", str(f), "--include-refused",
                     "--budget-gb", "40"]) == 0
    assert f.read_text().split() == ["4409", "4408", "4403", "4404", "4401"]
    assert json.loads(capsys.readouterr().out)["left_out"] == 1


def test_producer_refetches_rows_refused_for_room_in_file_order(tmp_path):
    from envision_eye_actionable.survey import pipeline
    png = _png_bytes(size=(64, 48))
    rids = ["4501", "4502", "4503", "4504"]
    records = {r: {f"{r}.png": png} for r in rids}
    meta = _write_records(tmp_path, records)
    cfg = _cfg(tmp_path, meta, consumer_grace_s=0.0, fetch_workers=1, order="scrape")
    cfg.out_dir.mkdir(parents=True)
    rows = [_row("4501", 5, 1.0), _row("4502", 50, 1.0),
            {"record_id": "4503", "status": "ok", "n_OCT": 3, "kept": True},
            _row("4504", 0, 1.0, reason="no eye image")]
    (cfg.out_dir / "survey_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg.refetch_ids = ["4502", "4503", "4504", "4501"]
    prod = pipeline.Producer(cfg)
    prod.zenodo.session = _Multi({rid: _FakeZenodo(rid, f) for rid, f in records.items()})
    assert prod.run() == 0
    ev = [json.loads(x) for x in (cfg.state_dir / "fetch_events.jsonl").read_text().splitlines()]
    assert [e["record"] for e in ev if e["event"] == "fetched"] == ["4502", "4501"]
    assert next(e for e in ev if e["event"] == "fetch_start")["n_refetch_keep"] == 2
