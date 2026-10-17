import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vendor_data.py"


def _layout(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / "vendor_data.py")
    data = tmp_path / "src" / "tnscope" / "_data"
    data.mkdir(parents=True)
    (data / "sentinel").write_text("keep")
    return tmp_path / "scripts" / "vendor_data.py", data


def test_invalid_source_keeps_existing_data(tmp_path):
    script, data = _layout(tmp_path)
    bad = tmp_path / "not-a-data-repo"
    (bad / "lexicon" / "hate").mkdir(parents=True)  # partial source
    r = subprocess.run([sys.executable, str(script), str(bad)], capture_output=True, text=True)
    assert r.returncode == 2
    assert (data / "sentinel").read_text() == "keep"
    assert "missing" in r.stderr
