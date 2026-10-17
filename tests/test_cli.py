import io
import json
import subprocess
import sys

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
