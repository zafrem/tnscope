# tnscope

Trust and safety text scanner. Given text and a language, it finds hate,
defamation and related harmful spans, scores them, and returns a masked copy.
The caller decides what to do with the result.

> **Warning: draft data.** All scores come from DRAFT, uncalibrated lexicon data
> that still awaits native-speaker review and policy-owner review. Do not treat
> scores or actions as production-quality decisions yet.

## Scope of v1

- Lexicon-only: every detection has `stage_reached == 1`.
- No HTTP API. It is a Python library and a CLI.
- No PII masking.
- No language auto-detection: `lang` is required.

## Install

```bash
pip install tnscope        # Python >= 3.11
```

For development: `pip install -e ".[dev]"`.

## Library

```python
import tnscope

result = tnscope.scan("Go back, wetback.", "en")
print(result.masked_text)   # Go back, *******.
for s in result.sentences:
    for d in s.detections:
        print(d.concept, d.category, d.score, d.label, d.action)
# 2001 hate 90.0 definite block
```

`tnscope.scan(text, lang, **kwargs)` uses a shared default `Engine` over the
bundled data and forwards its keyword arguments to
`Engine.scan(text, lang, *, thresholds=None, mask_min_score=None, mask_char="*")`.
It returns a `ScanResult` with `sentences` (span, rank, top_score, detections),
`masked_text` and `data_version`. Use `tnscope.Engine(data_dir=...)` to scan
with another data snapshot; its `scoring.yml` then supplies the default
thresholds (`Engine.default_thresholds`).

Each detection also carries `signals`:
`{"ambiguous": bool, "requires_context": bool}`. `ambiguous` is true when a
matched entry of the concept has `ambiguity >=` the data's
`router.ambiguous_min`; `requires_context` is true when a matched entry is
marked as needing context to be harmful. These are hints for the caller (for
example, to route to human review); they do not change scores or actions.
Detections with the same span and concept are reported once.

Offsets are Unicode code points of the original text. A `ScanResult` built
from text that contains a lone surrogate (e.g. `"\ud800"`) is returned
normally, but `ScanResult.model_dump_json()` then fails because such text
cannot be encoded as UTF-8 JSON; clean the input first if you serialize.

## CLI

```bash
tnscope scan "Go back, wetback." --lang en
tnscope scan "Go back, wetback." --lang en --json
tnscope scan --lang ko --file input.txt
echo "text" | tnscope scan --lang ja
```

Options: `--lang` (required), `--file`, `--json`, `--mask-min`, `--mask-char`,
`--block`, `--review`, `--flag`. Input is read from the TEXT argument, `--file`
or stdin (UTF-8). Numeric options must be finite numbers, and the thresholds
must satisfy `flag <= review <= block`; otherwise the exit code is 2.

## Thresholds and actions

Scores are 0-100. Default thresholds (read from the data's `scoring.yml`):
block >= 85, review >= 60, flag >= 35; defamation needs >= 90 to block. If you
set a stricter `block` than 90, it applies to defamation too; a lower `block`
never lowers the defamation requirement. Each detection carries a `label`
(e.g. `definite`) and an `action` (`block`, `review`, `flag`, `pass`).

The `action` is a stage-1 lexicon suggestion, not a moderation decision.
**tnscope only reports; the caller decides** what to do with each detection.
Override with `Thresholds(...)` or the CLI flags. `Thresholds(...)` must
satisfy `flag <= review <= block` and rejects NaN and infinities; fields you
do not pass take the built-in defaults above (not a custom data dir's values;
start from `engine.default_thresholds.model_dump()` for that).

## Masking

Spans of detections with score >= `mask_min_score` are replaced character by
character with `mask_char` (default `*`), so offsets in `masked_text` line up
with the original text. `mask_min_score` defaults to the effective `flag`
threshold, so lowering `flag` (or `--flag`) also masks more; pass
`mask_min_score` (or `--mask-min`) to set it independently.

## Languages

`en`, `ko`, `ja`, `zh-hans`, `zh-hant`.

Only the lexicon of the `lang` you pass is applied. Mixed-language text will
miss terms from the other languages.

Known normalization gaps (from the data spec, open items): the data seeds only
`strip`, `map`, `strip_between`, `nfkc` and `collapse_repeats` rules. Still to
add per language:

- en: fuller homoglyph tables (only a partial Cyrillic-to-Latin map exists
  today).
- ko: jamo composition/decomposition and choseong handling (needs algorithmic
  rule types; ko deliberately avoids NFKC because it rewrites compatibility
  jamo). Because there is no NFKC step for `ko`, decomposed Korean (NFD, as
  produced for example by macOS file names) is not detected.
- ja: hiragana/katakana folding and kanji variants.
- zh-hans / zh-hant: homophone and variant-character tables.

Obfuscated spellings relying on these will be missed.

## Stage 2 / 3 add-ons

`tnscope` itself has no models. Optional add-on packages (for example
`tnscope-ko`) register a scorer under the `tnscope.stage2` entry-point group,
and an LLM judge under `tnscope.stage3`. Hits the lexicon cannot decide on its
own (score below 85, or an ambiguous / `requires_context` hit) are sent to the
add-on for that language, and the scores are fused per `scoring.yml`.

- `max_stage=1` (CLI `--max-stage 1`) disables add-ons; `2` is the default;
  `3` also allows the LLM judge, which may send text to a model, so it is
  opt-in.
- A scorer has `api_version = 1`, `langs` (a set of language codes, or `{"*"}`)
  and `score(NlpRequest)` / `judge(LlmRequest)`. See `tnscope.stages`.
- Request `text` is the sentence and `spans` are offsets into that sentence.
- A broken add-on never raises into your scan: the earlier score stands and a
  code such as `stage2_failed lang=ko error=ValueError` appears in
  `ScanResult.warnings` (never input text).
