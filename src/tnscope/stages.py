"""Stage 2 (NLP) and stage 3 (LLM judge) plugin contract.

The wire formats mirror ``tns-context-engine/schemas/{nlp,llm}_*.schema.json``.
Add-on packages register a scorer object under the ``tnscope.stage2`` or
``tnscope.stage3`` entry-point group; the core never imports one by name.

Request ``text`` is the sentence under review and request ``spans`` are offsets
into that text (original characters, code points, half-open), not into the
whole input.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib.metadata import PackageNotFoundError, entry_points, version as pkg_version
import subprocess
import sys
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = 1
STAGE2_GROUP = "tnscope.stage2"
STAGE3_GROUP = "tnscope.stage3"
ANY_LANG = "*"
# NlpSignals.intent values that mean "the term is not being used abusively".
NON_ABUSIVE_INTENTS = frozenset({"report", "media", "discussion"})

ADDON_PACKAGES: dict[str, str] = {
    "en": "tnscope-en",
    "ko": "tnscope-ko",
    "ja": "tnscope-ja",
    "zh-hans": "tnscope-zh-hans",
    "zh-hant": "tnscope-zh-hant",
    "judge": "tnscope-judge",
}


class SpanRef(BaseModel):
    span: tuple[int, int]
    concept: int


class NlpRequest(BaseModel):
    text: str
    lang: str
    spans: list[SpanRef]


class NlpSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    negation: bool | None = None
    quotation: bool | None = None
    target_person: bool | None = None
    intent: str | None = Field(default=None, min_length=1)
    sentiment: float | None = Field(default=None, ge=-1, le=1)


class NlpResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=100)
    signals: NlpSignals

    @property
    def non_abusive_frame(self) -> bool:
        """True when the signals say the term is quoted, negated, reported or discussed."""
        s = self.signals
        return bool(s.quotation or s.negation or s.intent in NON_ABUSIVE_INTENTS)


class LlmRequest(BaseModel):
    text: str
    lang: str
    spans: list[SpanRef]
    prompt_template_id: str


class LlmVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: dict[str, float] = Field(min_length=1)
    label: str
    rationale: str | None = None
    model_id: str = Field(min_length=1)
    prompt_version: int = Field(ge=1)


@runtime_checkable
class Stage2Scorer(Protocol):
    api_version: int
    langs: frozenset[str]

    def score(self, request: NlpRequest) -> NlpResult | dict: ...


@runtime_checkable
class Stage3Scorer(Protocol):
    api_version: int
    langs: frozenset[str]

    def judge(self, request: LlmRequest) -> LlmVerdict | dict: ...


def covers(scorer, lang: str) -> bool:
    return lang in scorer.langs or ANY_LANG in scorer.langs


def _discover(group: str, method: str, warnings: list[str]) -> list:
    found = []
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
            scorer = obj() if isinstance(obj, type) else obj
            if not callable(getattr(scorer, method, None)):
                raise TypeError("missing " + method)
            if getattr(scorer, "api_version", None) != API_VERSION:
                warnings.append(f"plugin_incompatible group={group} name={ep.name}")
                continue
            found.append(scorer)
        except Exception as exc:  # a broken add-on must not break the core
            warnings.append(f"plugin_load_failed group={group} name={ep.name} error={type(exc).__name__}")
    return found


def discover_stage2(warnings: list[str]) -> list:
    return _discover(STAGE2_GROUP, "score", warnings)


def discover_stage3(warnings: list[str]) -> list:
    return _discover(STAGE3_GROUP, "judge", warnings)


def _pkg_ver(package: str) -> str | None:
    try:
        return pkg_version(package)
    except PackageNotFoundError:
        return None


def list_plugins() -> list[dict]:
    """Return status records for known official add-ons and any other installed plugins."""
    ep_by_group: dict[tuple[str, str], object] = {}
    for group in (STAGE2_GROUP, STAGE3_GROUP):
        for ep in entry_points(group=group):
            ep_by_group[(group, ep.name)] = ep

    items: list[dict] = []
    seen_eps: set[tuple[str, str]] = set()
    for name, package in ADDON_PACKAGES.items():
        stage = 3 if name == "judge" else 2
        group = STAGE3_GROUP if stage == 3 else STAGE2_GROUP
        method = "judge" if stage == 3 else "score"
        ep = ep_by_group.get((group, name))
        seen_eps.add((group, name))
        ver = _pkg_ver(package)
        if ep is None:
            items.append({
                "name": name,
                "package": package,
                "stage": stage,
                "langs": ["*"] if stage == 3 else [name],
                "installed": False,
                "version": ver,
                "status": "missing",
            })
            continue
        status = "ok"
        langs = ["*"] if stage == 3 else [name]
        try:
            obj = ep.load()
            scorer = obj() if isinstance(obj, type) else obj
            if not callable(getattr(scorer, method, None)):
                status = "broken"
            elif getattr(scorer, "api_version", None) != API_VERSION:
                status = "incompatible"
            else:
                raw_langs = getattr(scorer, "langs", ())
                if raw_langs:
                    langs = sorted(raw_langs)
        except Exception:
            status = "broken"
        items.append({
            "name": name,
            "package": package,
            "stage": stage,
            "langs": langs,
            "installed": status == "ok",
            "version": ver,
            "status": status,
        })

    # Include any third-party entry-point plugins not in ADDON_PACKAGES
    for (group, name), ep in sorted(ep_by_group.items()):
        if (group, name) in seen_eps:
            continue
        stage = 3 if group == STAGE3_GROUP else 2
        method = "judge" if stage == 3 else "score"
        dist = getattr(ep, "dist", None)
        package = getattr(dist, "name", None) or name
        ver = getattr(dist, "version", None)
        status = "ok"
        langs: list[str] = []
        try:
            obj = ep.load()
            scorer = obj() if isinstance(obj, type) else obj
            if not callable(getattr(scorer, method, None)):
                status = "broken"
            elif getattr(scorer, "api_version", None) != API_VERSION:
                status = "incompatible"
            else:
                langs = sorted(getattr(scorer, "langs", ()))
        except Exception:
            status = "broken"
        items.append({
            "name": name,
            "package": package,
            "stage": stage,
            "langs": langs,
            "installed": status == "ok",
            "version": ver,
            "status": status,
        })
    return items


def resolve_addon_packages(targets: Iterable[str]) -> list[str]:
    """Resolve plugin target names (e.g. ``['ko', 'en']`` or ``['all']``) to package names."""
    pkgs: list[str] = []
    valid = sorted([*ADDON_PACKAGES.keys(), "all"])
    for raw in targets:
        t = raw.strip().lower()
        if not t:
            continue
        if t == "all":
            for pkg in ADDON_PACKAGES.values():
                if pkg not in pkgs:
                    pkgs.append(pkg)
        elif t in ADDON_PACKAGES:
            pkg = ADDON_PACKAGES[t]
            if pkg not in pkgs:
                pkgs.append(pkg)
        else:
            raise ValueError(f"unknown plugin {raw!r}; choose from: {', '.join(valid)}")
    if not pkgs:
        raise ValueError(f"no plugin specified; choose from: {', '.join(valid)}")
    return pkgs


def install_plugins(
    targets: Iterable[str],
    *,
    upgrade: bool = False,
    dry_run: bool = False,
    pip_args: Sequence[str] = (),
) -> list[str]:
    """Install one or more official tnscope add-on packages via ``pip`` and reset the default engine."""
    pkgs = resolve_addon_packages(targets)
    if dry_run:
        return pkgs
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(pip_args)
    cmd.extend(pkgs)
    subprocess.run(cmd, check=True)
    from tnscope.engine import reset_default_engine

    reset_default_engine()
    return pkgs
