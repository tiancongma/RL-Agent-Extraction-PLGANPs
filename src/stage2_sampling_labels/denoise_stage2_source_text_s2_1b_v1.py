#!/usr/bin/env python3
"""S2-1b high-confidence source denoise projection.

This helper is intentionally narrow: it deletes only deterministic,
high-confidence source boilerplate/noise before Stage2 evidence construction.
It does not discover formulations, select important tables, construct row
universes, or complete source anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

RULE_CONFIDENCE = "high"
SUMMARY_FIELDS = [
    "paper_key",
    "input_text_path",
    "output_text_path",
    "raw_char_count",
    "denoised_char_count",
    "removed_char_count",
    "raw_line_count",
    "denoised_line_count",
    "removed_line_count",
    "rule_id",
    "rule_class",
    "rule_confidence",
    "source_locator",
    "removed_text_preview",
    "preservation_exception",
]


@dataclass(frozen=True)
class RemovedEvent:
    rule_id: str
    rule_class: str
    rule_confidence: str
    source_locator: str
    removed_text_preview: str
    preservation_exception: str = ""


@dataclass(frozen=True)
class DenoiseResult:
    paper_key: str
    input_text_path: str
    denoised_text: str
    raw_char_count: int
    denoised_char_count: int
    removed_char_count: int
    raw_line_count: int
    denoised_line_count: int
    removed_line_count: int
    removed_events: list[RemovedEvent]


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    rule_class: str
    pattern: re.Pattern[str]

    def matches(self, line: str) -> bool:
        return bool(self.pattern.match(line.strip()))


RULES: tuple[_Rule, ...] = (
    _Rule(
        "S2_1B_PUBLISHER_CHROME_V1",
        "publisher_chrome",
        re.compile(
            r"^(?:Journals\s*&\s*Books|Help|Search|View\s+PDF|Download\s+full\s+issue|Outline|"
            r"Article\s+outline|Show\s+more|Show\s+less|Get\s+rights\s+and\s+content|"
            r"ScienceDirect|Elsevier|SpringerLink|Wiley\s+Online\s+Library)\s*$",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "S2_1B_DOWNLOAD_MARKER_V1",
        "download_marker",
        re.compile(
            r"^(?:Downloaded\s+from\b|This\s+article\s+was\s+downloaded\s+by\b|"
            r"Accessed\s+from\b).*$",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "S2_1B_PAGE_HEADER_FOOTER_V1",
        "page_header_footer",
        re.compile(
            r"^(?:Page\s+\d+\s+of\s+\d+|\d+\s*/\s*\d+|"
            r"doi:\s*10\.\S+|https?://doi\.org/10\.\S+|"
            r"[A-Z][A-Za-z&\- ]+\s+\d+\s*\(\d{4}\)\s*\d+\s*[-–]\s*\d+)\s*$",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "S2_1B_AUTHOR_PAGE_RUNNING_LINE_V1",
        "author_page_running_line",
        re.compile(
            r"^[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?\s+(?:/\s+)?(?:Page\s+)?\d+\s+(?:of\s+\d+)?$",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "S2_1B_COPYRIGHT_LICENSE_V1",
        "copyright_or_license_boilerplate",
        re.compile(
            r"^(?:Copyright\b|©|\(c\)|All\s+rights\s+reserved\b|Creative\s+Commons\b|"
            r"This\s+is\s+an\s+open\s+access\s+article\b|Licensed\s+under\b).*$",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "S2_1B_RELATED_ARTICLES_V1",
        "article_recommendation_or_related_articles",
        re.compile(
            r"^(?:Recommended\s+articles?|Related\s+articles?|Related\s+article\s+metadata\b|"
            r"Cited\s+by\s+\d+|View\s+related\s+articles?).*$",
            re.IGNORECASE,
        ),
    ),
)

REFERENCE_HEADING_RE = re.compile(r"^(?:References|Bibliography|Literature\s+cited)\s*$", re.IGNORECASE)
ISOLATED_REFERENCE_RE = re.compile(
    r"^(?:\[?\d+\]?\.?\s+|[A-Z][A-Za-z'\-]+\s+[A-Z](?:\.|,)).*"
    r"(?:\b(?:19|20)\d{2}\b|\bdoi:\s*10\.|\b\d+\s*[:;]\s*\d+(?:\s*[-–]\s*\d+)?)"
    r".*$",
    re.IGNORECASE,
)

NUMBERED_REFERENCE_PREFIX_RE = re.compile(r"^\[?\d+\]?\.?\s+\S+")

PRESERVE_TERMS_RE = re.compile(
    r"\b(?:PLGA|nanoparticle|formulation|prepared|preparation|materials?|methods?|Table\s+\d+|"
    r"design\s+matrix|polymer|PVA|encapsulation|particle\s+size|zeta|drug\s+loading)\b",
    re.IGNORECASE,
)
DOE_PRESERVE_RE = re.compile(r"\bDOE\b")


def _preview(line: str) -> str:
    text = " ".join(line.strip().split())
    return text[:160]


def _line_count(text: str) -> int:
    if text == "":
        return 0
    return len(text.splitlines())


def _should_preserve(line: str) -> bool:
    return bool(PRESERVE_TERMS_RE.search(line) or DOE_PRESERVE_RE.search(line))


def _match_non_reference_rule(line: str) -> _Rule | None:
    if _should_preserve(line):
        return None
    for rule in RULES:
        if rule.matches(line):
            return rule
    return None


def denoise_text(raw_text: str, *, paper_key: str, input_text_path: str) -> DenoiseResult:
    """Return a denoised Stage2 text projection and auditable removals."""
    lines = raw_text.splitlines()
    kept_lines: list[str] = []
    removed: list[RemovedEvent] = []
    in_reference_tail = False

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        locator = f"line:{index}"

        if in_reference_tail:
            if _should_preserve(line):
                in_reference_tail = False
            elif ISOLATED_REFERENCE_RE.match(stripped) or NUMBERED_REFERENCE_PREFIX_RE.match(stripped):
                removed.append(
                    RemovedEvent(
                        rule_id="S2_1B_ISOLATED_REFERENCE_LINE_V1",
                        rule_class="isolated_reference_line",
                        rule_confidence=RULE_CONFIDENCE,
                        source_locator=locator,
                        removed_text_preview=_preview(line),
                    )
                )
                continue
            else:
                tail_rule = _match_non_reference_rule(line)
                if tail_rule is not None:
                    removed.append(
                        RemovedEvent(
                            rule_id=tail_rule.rule_id,
                            rule_class=tail_rule.rule_class,
                            rule_confidence=RULE_CONFIDENCE,
                            source_locator=locator,
                            removed_text_preview=_preview(line),
                        )
                    )
                else:
                    removed.append(
                        RemovedEvent(
                            rule_id="S2_1B_REFERENCE_TAIL_V1",
                            rule_class="reference_tail",
                            rule_confidence=RULE_CONFIDENCE,
                            source_locator=locator,
                            removed_text_preview=_preview(line),
                        )
                    )
                continue

        if not _should_preserve(line) and REFERENCE_HEADING_RE.match(stripped):
            in_reference_tail = True
            removed.append(
                RemovedEvent(
                    rule_id="S2_1B_REFERENCE_TAIL_V1",
                    rule_class="reference_tail",
                    rule_confidence=RULE_CONFIDENCE,
                    source_locator=locator,
                    removed_text_preview=_preview(line),
                )
            )
            continue

        if not _should_preserve(line) and (
            ISOLATED_REFERENCE_RE.match(stripped) or NUMBERED_REFERENCE_PREFIX_RE.match(stripped)
        ):
            removed.append(
                RemovedEvent(
                    rule_id="S2_1B_ISOLATED_REFERENCE_LINE_V1",
                    rule_class="isolated_reference_line",
                    rule_confidence=RULE_CONFIDENCE,
                    source_locator=locator,
                    removed_text_preview=_preview(line),
                )
            )
            continue

        rule = _match_non_reference_rule(line)
        if rule is not None:
            removed.append(
                RemovedEvent(
                    rule_id=rule.rule_id,
                    rule_class=rule.rule_class,
                    rule_confidence=RULE_CONFIDENCE,
                    source_locator=locator,
                    removed_text_preview=_preview(line),
                )
            )
            continue

        kept_lines.append(line)

    denoised_text = "\n".join(kept_lines)
    raw_char_count = len(raw_text)
    denoised_char_count = len(denoised_text)
    return DenoiseResult(
        paper_key=paper_key,
        input_text_path=input_text_path,
        denoised_text=denoised_text,
        raw_char_count=raw_char_count,
        denoised_char_count=denoised_char_count,
        removed_char_count=raw_char_count - denoised_char_count,
        raw_line_count=_line_count(raw_text),
        denoised_line_count=_line_count(denoised_text),
        removed_line_count=len(removed),
        removed_events=removed,
    )


def _summary_rows(result: DenoiseResult, output_text_path: Path) -> list[dict[str, str]]:
    base = {
        "paper_key": result.paper_key,
        "input_text_path": result.input_text_path,
        "output_text_path": str(output_text_path),
        "raw_char_count": str(result.raw_char_count),
        "denoised_char_count": str(result.denoised_char_count),
        "removed_char_count": str(result.removed_char_count),
        "raw_line_count": str(result.raw_line_count),
        "denoised_line_count": str(result.denoised_line_count),
        "removed_line_count": str(result.removed_line_count),
    }
    if not result.removed_events:
        return [
            {
                **base,
                "rule_id": "",
                "rule_class": "",
                "rule_confidence": "",
                "source_locator": "",
                "removed_text_preview": "",
                "preservation_exception": "no_high_confidence_noise_removed",
            }
        ]
    rows: list[dict[str, str]] = []
    for event in result.removed_events:
        rows.append({**base, **asdict(event)})
    return rows


def run_denoise_projection(*, inputs: Sequence[tuple[str, Path]], run_dir: Path) -> Path:
    """Write S2-1b denoised text, per-paper audit JSON, and summary TSV."""
    denoised_dir = run_dir / "semantic_stage2_objects" / "s2_1b_denoised_text"
    audit_dir = run_dir / "semantic_stage2_objects" / "s2_1b_denoise_audit"
    analysis_dir = run_dir / "analysis"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = []
    for paper_key, input_path in inputs:
        raw_text = input_path.read_text(encoding="utf-8")
        result = denoise_text(raw_text, paper_key=paper_key, input_text_path=str(input_path))
        output_text_path = denoised_dir / f"{paper_key}.txt"
        output_text_path.write_text(result.denoised_text, encoding="utf-8")
        audit_path = audit_dir / f"{paper_key}_s2_1b_denoise_audit_v1.json"
        audit_payload = {
            "paper_key": paper_key,
            "input_text_path": str(input_path),
            "output_text_path": str(output_text_path),
            "source_text_projection": "s2_1b_denoised",
            "source_raw_clean_text_path": str(input_path),
            "source_s2_1b_denoised_text_path": str(output_text_path),
            "s2_1b_denoise_audit_path": str(audit_path),
            "raw_char_count": result.raw_char_count,
            "denoised_char_count": result.denoised_char_count,
            "removed_char_count": result.removed_char_count,
            "raw_line_count": result.raw_line_count,
            "denoised_line_count": result.denoised_line_count,
            "removed_line_count": result.removed_line_count,
            "removed_events": [asdict(event) for event in result.removed_events],
            "semantic_authority_note": (
                "diagnostic/internal S2-1b source hygiene only; LLM remains Stage2 semantic authority"
            ),
        }
        audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True), encoding="utf-8")
        summary_rows.extend(_summary_rows(result, output_text_path))

    summary_path = analysis_dir / "s2_1b_denoise_summary_v1.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def _read_inputs_tsv(path: Path) -> list[tuple[str, Path]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "paper_key" not in (reader.fieldnames or []) or "input_text_path" not in (reader.fieldnames or []):
            raise ValueError("--inputs-tsv must include paper_key and input_text_path columns")
        return [(row["paper_key"], Path(row["input_text_path"])) for row in reader]


def _parse_input_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--input values must use PAPER_KEY=/path/to/text.txt")
        paper_key, path = spec.split("=", 1)
        if not paper_key or not path:
            raise ValueError("--input values must use non-empty PAPER_KEY and path")
        parsed.append((paper_key, Path(path)))
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Explicit run child directory for outputs")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Explicit paper text input as PAPER_KEY=/path/to/source.txt; repeatable",
    )
    parser.add_argument(
        "--inputs-tsv",
        type=Path,
        help="Optional TSV with paper_key and input_text_path columns",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = _parse_input_specs(args.input)
    if args.inputs_tsv:
        inputs.extend(_read_inputs_tsv(args.inputs_tsv))
    if not inputs:
        raise SystemExit("At least one explicit --input or --inputs-tsv row is required")
    summary_path = run_denoise_projection(inputs=inputs, run_dir=args.run_dir)
    print(f"diagnostic/internal S2-1b denoise summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
