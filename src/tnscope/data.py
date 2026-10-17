"""Load the vendored lexicon / normalize / context / scoring data."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

SUPPORTED_SCHEMA_VERSION = 1
CATEGORIES = ("hate", "defamation")


class DataVersionError(Exception):
    """Raised when the data snapshot has an unsupported schema_version."""


@dataclass
class Entry:
    id: str
    concept: int
    term: str
    type: str
    rank: int
    ambiguity: float
    requires_context: bool
    flags: tuple[str, ...]
    hit_keywords: tuple[str, ...]
    always_run: bool
    category: str


@dataclass
class Data:
    data_version: str
    languages: tuple[str, ...]
    lexicon: dict[str, list[Entry]]
    normalize_rules: dict[str, list[dict]]
    context_rules: dict[str, list[dict]]
    scoring: dict


def _yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(data_dir: Path | None = None) -> Data:
    root = Path(str(resources.files("tnscope") / "_data")) if data_dir is None else Path(data_dir)
    meta = _yaml(root / "reference" / "meta.yml")
    version = meta.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION or isinstance(version, bool):
        raise DataVersionError(
            f"unsupported schema_version {version!r}; expected {SUPPORTED_SCHEMA_VERSION}"
        )

    lexicon: dict[str, list[Entry]] = {}
    for category in CATEGORIES:
        for path in sorted((root / "lexicon" / category).glob("*.yml")):
            lang = path.stem
            for e in _yaml(path).get("entries") or []:
                lexicon.setdefault(lang, []).append(
                    Entry(
                        id=e["id"],
                        concept=e["concept"],
                        term=e["term"],
                        type=e["type"],
                        rank=e["rank"],
                        ambiguity=float(e.get("ambiguity", 0.0)),
                        requires_context=bool(e.get("requires_context", False)),
                        flags=tuple(e.get("flags") or ()),
                        hit_keywords=tuple(k for k in (e.get("hit_keywords") or ()) if k),
                        always_run=bool(e.get("always_run", False)),
                        category=category,
                    )
                )

    def rules(sub: str) -> dict[str, list[dict]]:
        return {
            p.stem: list(_yaml(p).get("rules") or [])
            for p in sorted((root / sub).glob("*.yml"))
        }

    return Data(
        data_version=str(meta["data_version"]),
        languages=tuple(meta["languages"]),
        lexicon=lexicon,
        normalize_rules=rules("normalize"),
        context_rules=rules("context"),
        scoring=_yaml(root / "scoring" / "scoring.yml"),
    )
