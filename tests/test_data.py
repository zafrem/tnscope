import shutil
from importlib import resources

import pytest
import yaml

from tnscope.data import DataVersionError, load_data


def _copy_bundled(tmp_path):
    dest = tmp_path / "data"
    shutil.copytree(str(resources.files("tnscope") / "_data"), dest)
    return dest


def test_loads_bundled_data():
    d = load_data()
    assert set(d.languages) == {"en", "ko", "ja", "zh-hans", "zh-hant"}
    e = next(x for x in d.lexicon["en"] if x.id == "hate_en_0005")
    assert e.rank == 5
    assert e.category == "hate"
    assert d.scoring["floor"] == {"rank": 5, "min_score": 60}
    assert {x.category for v in d.lexicon.values() for x in v} == {"hate", "defamation"}


def test_rejects_schema_version_2(tmp_path):
    dest = _copy_bundled(tmp_path)
    meta = dest / "reference" / "meta.yml"
    m = yaml.safe_load(meta.read_text())
    m["schema_version"] = 2
    meta.write_text(yaml.safe_dump(m))
    with pytest.raises(DataVersionError):
        load_data(dest)


def test_custom_data_dir(tmp_path):
    dest = _copy_bundled(tmp_path)
    d = load_data(dest)
    assert "en" in d.lexicon
