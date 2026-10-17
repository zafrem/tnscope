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

`scan(text, lang, *, thresholds=None, mask_min_score=35.0, mask_char="*")`
returns a `ScanResult` with `sentences` (span, rank, top_score, detections),
`masked_text` and `data_version`.

## CLI

```bash
tnscope scan "Go back, wetback." --lang en
tnscope scan "Go back, wetback." --lang en --json
tnscope scan --lang ko --file input.txt
echo "text" | tnscope scan --lang ja
```

Options: `--lang` (required), `--file`, `--json`, `--mask-min`, `--mask-char`,
`--block`, `--review`, `--flag`. Input is read from the TEXT argument, `--file`
or stdin (UTF-8).

## Thresholds and actions

Scores are 0-100. Default thresholds: block >= 85, review >= 60, flag >= 35.
Category overrides for block: illegal >= 70, defamation >= 90. Each detection
carries a `label` (e.g. `definite`) and an `action` (`block`, `review`,
`flag`, `pass`). **tnscope only reports; the caller decides** what to do with each
action. Override with `Thresholds(...)` or the CLI flags.

## Masking

Spans of detections with score >= `mask_min_score` (default 35) are replaced
character by character with `mask_char` (default `*`), so offsets in
`masked_text` line up with the original text.

## Languages

`en`, `ko`, `ja`, `zh-hans`, `zh-hant`.

Known normalization gaps (from the data spec, open items): the data seeds only
`strip`, `map`, `strip_between`, `nfkc` and `collapse_repeats` rules. Still to
add per language:

- en: homoglyph tables.
- ko: jamo composition/decomposition and choseong handling (needs algorithmic
  rule types; ko deliberately avoids NFKC because it rewrites compatibility
  jamo).
- ja: hiragana/katakana folding and kanji variants.
- zh-hans / zh-hant: homophone and variant-character tables.

Obfuscated spellings relying on these will be missed.
