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
