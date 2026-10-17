"""Command-line interface: ``tnscope scan``."""

from __future__ import annotations

import argparse
import sys

from tnscope.engine import scan
from tnscope.models import Thresholds


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse messages never echo positional input text
        self.print_usage(sys.stderr)
        self.exit(2, f"tnscope: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="tnscope")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
    s = sub.add_parser("scan", help="scan text for harmful content")
    s.add_argument("text", nargs="?", default=None, metavar="TEXT")
    s.add_argument("--lang", required=True)
    s.add_argument("--file", default=None)
    s.add_argument("--json", action="store_true")
    s.add_argument("--mask-min", type=float, default=35.0)
    s.add_argument("--mask-char", default="*")
    s.add_argument("--block", type=float, default=None)
    s.add_argument("--review", type=float, default=None)
    s.add_argument("--flag", type=float, default=None)
    return p


def _fmt(x: float) -> str:
    return f"{x:g}"


def _run(args) -> int:
    if len(args.mask_char) != 1:
        print("tnscope: error: --mask-char must be a single character", file=sys.stderr)
        return 2
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"tnscope: error: cannot read file ({type(e).__name__})", file=sys.stderr)
            return 2
    else:
        text = sys.stdin.read()

    overrides = {
        k: v for k, v in (("block", args.block), ("review", args.review), ("flag", args.flag))
        if v is not None
    }
    try:
        result = scan(
            text,
            args.lang,
            thresholds=Thresholds(**overrides),
            mask_min_score=args.mask_min,
            mask_char=args.mask_char,
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
