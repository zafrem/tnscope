import shutil
from importlib import resources

import pytest
import yaml


@pytest.fixture
def custom_data(tmp_path):
    """Return a function that copies the bundled data and appends lexicon entries.

    ``make(category, lang, entries)`` appends ``entries`` (list of dicts) to
    ``lexicon/<category>/<lang>.yml`` of a fresh copy and returns the data dir.
    """
    dest = tmp_path / "data"
    shutil.copytree(str(resources.files("tnscope") / "_data"), dest)

    def make(category: str, lang: str, entries: list[dict], replace: bool = False):
        path = dest / "lexicon" / category / f"{lang}.yml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
        doc = doc or {"lang": lang, "entries": []}
        doc["entries"] = list(entries) if replace else list(doc.get("entries") or []) + list(entries)
        path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
        return dest

    make.root = dest
    return make


@pytest.fixture(autouse=True)
def no_installed_addons(monkeypatch):
    """Keep the core tests independent of whichever add-ons are pip-installed.

    ``Engine()`` normally discovers ``tnscope.stage2`` / ``tnscope.stage3`` plugins;
    here it finds none, so a test gets an add-on only by passing one explicitly
    (``Engine(stage2=[...])``). Discovery itself is tested by calling
    ``tnscope.stages.discover_stage2`` directly.
    """
    monkeypatch.setattr("tnscope.engine.discover_stage2", lambda warnings: [])
    monkeypatch.setattr("tnscope.engine.discover_stage3", lambda warnings: [])
