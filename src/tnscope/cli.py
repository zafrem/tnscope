"""Command-line interface: ``tnscope scan``, ``tnscope plugins``, ``tnscope install``."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

from pydantic import ValidationError

from tnscope.engine import default_engine
from tnscope.models import Thresholds
from tnscope.stages import install_plugins, list_plugins


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse messages never echo positional input text
        self.print_usage(sys.stderr)
        self.exit(2, f"tnscope: error: {message}\n")


def _finite(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError("must be a finite number")
    return x


def _build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="tnscope")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
    s = sub.add_parser("scan", help="scan text for harmful content")
    s.add_argument("text", nargs="?", default=None, metavar="TEXT")
    s.add_argument("--lang", required=True)
    s.add_argument("--file", default=None)
    s.add_argument("--json", action="store_true")
    s.add_argument("--mask-min", type=_finite, default=None,
                   help="mask detections with score >= this (default: the flag threshold)")
    s.add_argument("--mask-char", default="*")
    s.add_argument("--block", type=_finite, default=None)
    s.add_argument("--review", type=_finite, default=None)
    s.add_argument("--flag", type=_finite, default=None)
    s.add_argument("--verify-definite", action="store_true",
                   help="also send definite lexicon hits to stage 2 add-ons")
    s.add_argument("--max-stage", type=int, choices=(1, 2, 3), default=2,
                   help="1 lexicon only, 2 add NLP add-ons (default), 3 also LLM judge (opt-in)")

    pl = sub.add_parser("plugins", help="list stage 2/3 add-on plugins and their status")
    pl.add_argument("--json", action="store_true")

    inst = sub.add_parser(
        "install",
        help="install language add-on packages (en, ko, ja, zh-hans, zh-hant, judge, or all)",
    )
    inst.add_argument(
        "targets",
        nargs="+",
        metavar="PLUGIN",
        help="language code(s), 'judge', or 'all'",
    )
    inst.add_argument("--upgrade", "-U", action="store_true", help="upgrade already-installed add-ons")
    inst.add_argument("--dry-run", action="store_true", help="print packages to install without running pip")
    return p


def _fmt(x: float) -> str:
    return f"{x:g}"


def _run_plugins(args) -> int:
    items = list_plugins()
    if args.json:
        print(json.dumps(items))
        return 0
    for item in items:
        ver = f" ({item['version']})" if item["version"] else ""
        langs = ",".join(item["langs"])
        print(f"{item['name']:<8}  stage {item['stage']}  {item['package']}{ver:<10}  [{item['status']}]  langs={langs}")
    return 0


def _run_install(args) -> int:
    try:
        pkgs = install_plugins(args.targets, upgrade=args.upgrade, dry_run=args.dry_run)
    except ValueError as e:
        print(f"tnscope: error: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"tnscope: error: pip install failed (exit code {e.returncode})", file=sys.stderr)
        return 1
    if args.dry_run:
        print(" ".join(pkgs))
    else:
        print(f"installed: {' '.join(pkgs)}")
    return 0


def _run(args) -> int:
    if args.command == "plugins":
        return _run_plugins(args)
    if args.command == "install":
        return _run_install(args)
    if len(args.mask_char) != 1:
        print("tnscope: error: --mask-char must be a single character", file=sys.stderr)
        return 2
    if args.text is not None:
        text = args.text
    else:
        try:
            if args.file is not None:
                with open(args.file, "rb") as f:
                    raw = f.read()
            else:
                raw = sys.stdin.buffer.read()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            what = "file" if args.file is not None else "stdin"
            print(f"tnscope: error: cannot read {what} ({type(e).__name__})", file=sys.stderr)
            return 2

    overrides = {
        k: v for k, v in (("block", args.block), ("review", args.review), ("flag", args.flag))
        if v is not None
    }
    try:
        engine = default_engine()
        thresholds = None
        if overrides:
            base = engine.default_thresholds.model_dump()
            try:
                thresholds = Thresholds(**{**base, **overrides})
            except ValidationError:
                print(
                    "tnscope: error: invalid thresholds: need flag <= review <= block",
                    file=sys.stderr,
                )
                return 2
        result = engine.scan(
            text,
            args.lang,
            thresholds=thresholds,
            mask_min_score=args.mask_min,
            mask_char=args.mask_char,
            max_stage=args.max_stage,
            verify_definite=args.verify_definite,
        )
    except ValueError as e:
        print(f"tnscope: error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(result.model_dump_json())
        return 0
    for s in result.sentences:
        a, b = s.span
        print(f"[rank {s.rank} score {_fmt(s.top_score)}] {result.masked_text[a:b]}")
    print(result.masked_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    return _run(args)
