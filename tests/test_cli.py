import io
import json
import subprocess
import sys

import pytest

from tnscope.cli import main


def _stdin(data: bytes):
    return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8")


def _actions(data):
    return [d["action"] for s in data["sentences"] for d in s["detections"]]


def test_json_scan(capsys):
    assert main(["scan", "Go back, wetback.", "--lang", "en", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["masked_text"] == "Go back, *******."
    assert _actions(data) == ["block"]


def test_human_output(capsys):
    assert main(["scan", "Go back, wetback.", "--lang", "en"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[rank 5 score 90] Go back, *******."
    assert out[-1] == "Go back, *******."


def test_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _stdin(b"Go back, wetback."))
    assert main(["scan", "--lang", "en", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["masked_text"] == "Go back, *******."


def test_file(tmp_path, capsys):
    p = tmp_path / "in.txt"
    p.write_text("Go back, wetback.", encoding="utf-8")
    assert main(["scan", "--lang", "en", "--json", "--file", str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["masked_text"] == "Go back, *******."


def test_text_beats_file(tmp_path, capsys):
    p = tmp_path / "in.txt"
    p.write_text("Go back, wetback.", encoding="utf-8")
    assert main(["scan", "hello there", "--lang", "en", "--json", "--file", str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["masked_text"] == "hello there"


def test_missing_file_is_usage_error(tmp_path, capsys):
    p = tmp_path / "nope.txt"
    assert main(["scan", "--lang", "en", "--file", str(p)]) == 2
    assert capsys.readouterr().err.strip()


def test_block_threshold_changes_action(capsys):
    main(["scan", "Go back, wetback.", "--lang", "en", "--json", "--block", "95"])
    assert _actions(json.loads(capsys.readouterr().out)) == ["review"]


def test_unsupported_lang(capsys):
    secret = "SECRETINPUT"
    assert main(["scan", secret, "--lang", "xx"]) == 2
    err = capsys.readouterr().err
    assert err.strip() and secret not in err


def test_empty_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _stdin(b""))
    assert main(["scan", "--lang", "en", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sentences"] == [] and data["masked_text"] == ""


def test_mask_char_too_long(capsys):
    assert main(["scan", "Go back, wetback.", "--lang", "en", "--mask-char", "ab"]) == 2
    assert capsys.readouterr().err.strip()


def test_mask_char_custom(capsys):
    main(["scan", "Go back, wetback.", "--lang", "en", "--json", "--mask-char", "#"])
    assert json.loads(capsys.readouterr().out)["masked_text"] == "Go back, #######."


def test_lang_required(capsys):
    assert main(["scan", "hi"]) == 2


def test_python_dash_m():
    r = subprocess.run(
        [sys.executable, "-m", "tnscope", "scan", "Go back, wetback.", "--lang", "en", "--json"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["masked_text"] == "Go back, *******."


def test_invalid_utf8_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _stdin(b"abc \xff\xfe"))
    assert main(["scan", "--lang", "en"]) == 2
    err = capsys.readouterr().err
    assert err.strip() and "Traceback" not in err and "abc" not in err
    assert len(err.strip().splitlines()) == 1


def test_invalid_utf8_file(tmp_path, capsys):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"abc \xff\xfe")
    assert main(["scan", "--lang", "en", "--file", str(p)]) == 2
    err = capsys.readouterr().err
    assert err.strip() and "Traceback" not in err and "abc" not in err


def test_non_ascii_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _stdin("wetback 한국어".encode()))
    assert main(["scan", "--lang", "en", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["masked_text"] == "******* 한국어"


@pytest.mark.parametrize("opt", ["--mask-min", "--flag", "--review", "--block"])
@pytest.mark.parametrize("val", ["nan", "inf", "-inf"])
def test_non_finite_numeric_options(opt, val, capsys):
    secret = "SECRET wetback"
    assert main(["scan", secret, "--lang", "en", opt, val]) == 2
    err = capsys.readouterr().err
    assert err.strip() and "SECRET" not in err and "Traceback" not in err
    errors = [ln for ln in err.splitlines() if ln.startswith("tnscope: error:")]
    assert len(errors) == 1 and err.rstrip().endswith(errors[0])


def test_inconsistent_thresholds_exit_2(capsys):
    secret = "SECRET wetback"
    assert main(["scan", secret, "--lang", "en", "--flag", "70"]) == 2
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "flag <= review <= block" in err and "SECRET" not in err


def test_lower_flag_moves_mask_default(capsys):
    main(["scan", "Never say wetback.", "--lang", "en", "--json",
          "--block", "100", "--review", "90", "--flag", "75"])
    data = json.loads(capsys.readouterr().out)
    assert _actions(data) == ["pass"]
    assert data["masked_text"] == "Never say wetback."
    main(["scan", "Never say wetback.", "--lang", "en", "--json", "--flag", "20"])
    assert json.loads(capsys.readouterr().out)["masked_text"] == "Never say *******."


def test_max_stage_flag_rejects_bad_value(capsys):
    assert main(["scan", "x", "--lang", "en", "--max-stage", "4"]) == 2


def test_plugins_json_and_text(capsys):
    assert main(["plugins", "--json"]) == 0
    items = json.loads(capsys.readouterr().out)
    names = [it["name"] for it in items]
    assert {"en", "ko", "ja", "zh-hans", "zh-hant", "judge"}.issubset(names)

    assert main(["plugins"]) == 0
    text_out = capsys.readouterr().out
    assert "tnscope-ko" in text_out and "tnscope-judge" in text_out


def test_install_dry_run(capsys):
    assert main(["install", "ko", "en", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == "tnscope-ko tnscope-en"

    assert main(["install", "all", "--dry-run"]) == 0
    out = capsys.readouterr().out.strip().split()
    assert out == [
        "tnscope-en",
        "tnscope-ko",
        "tnscope-ja",
        "tnscope-zh-hans",
        "tnscope-zh-hant",
        "tnscope-judge",
    ]


def test_install_runs_pip_and_resets_engine(monkeypatch, capsys):
    from tnscope import stages

    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr(stages.subprocess, "run", fake_run)
    assert main(["install", "ko", "--upgrade"]) == 0
    assert len(calls) == 1
    cmd, check = calls[0]
    assert check is True
    assert cmd == [sys.executable, "-m", "pip", "install", "--upgrade", "tnscope-ko"]
    assert "installed: tnscope-ko" in capsys.readouterr().out


def test_install_unknown_plugin_exits_2(capsys):
    assert main(["install", "fr"]) == 2
    err = capsys.readouterr().err
    assert "unknown plugin" in err and "choose from:" in err


def test_install_pip_failure_exits_1(monkeypatch, capsys):
    from tnscope import stages

    def fail_run(cmd, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(stages.subprocess, "run", fail_run)
    assert main(["install", "ko"]) == 1
    assert "pip install failed" in capsys.readouterr().err
