"""Test isolation: never pick up a real Zenodo token from the environment or
the default token file of the machine the tests run on."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_zenodo_token(tmp_path_factory, monkeypatch):
    from envision_eye_actionable.survey import zenodo
    monkeypatch.delenv(zenodo.TOKEN_ENV, raising=False)
    monkeypatch.setattr(zenodo, "DEFAULT_TOKEN_FILE", tmp_path_factory.mktemp("home") / "no_token_file")
