#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_HEADER_ALIAS_LEXICON = REPO_ROOT / "data" / "cleaned" / "reference" / "value_normalization_lexicon_v1.tsv"
OVERLAY_ENV_VAR = "PLGA_VALUE_NORMALIZATION_OVERLAY_TSV"

_LEXICON_CACHE: list[dict[str, str]] | None = None

HEADER_SIGNAL_PATTERNS = [
    r"\bformulation\b",
    r"\bpolymer\b",
    r"\bsurfactant\b",
    r"\bstabilizer\b",
    r"\bemulsifier\b",
    r"\baverage\b",
    r"\bsize\b",
    r"\bpolydispersity\b",
    r"\bpolidispersity\b",
    r"\bp\.?\s*i\.?\b",
    r"\bpdi\b",
    r"\bzeta\b",
    r"\bzp\b",
    r"\bee\b",
    r"\be\.?\s*e\.?\b",
    r"\bloading\b",
    r"\byield\b",
    r"\bdiameter\b",
    r"\bindex\b",
    r"\bused\b",
    r"\bnumber\b",
]

METADATA_ROW_PATTERNS = [
    r"^table\s+\d+\b",
    r"^figure\s+\d+\b",
    r"\bcharacterization of\b",
    r"\bdeveloped\.$",
    r"\btriblocks were used\.?$",
]

SYMBOL_SUFFIX_PATTERN = re.compile(r"^[®™†‡*]+$")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _canonical_text(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


def _compact_text(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9%]+", _canonical_text(value)))


def _load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _overlay_paths_from_env() -> list[Path]:
    raw_value = os.environ.get(OVERLAY_ENV_VAR, "")
    paths: list[Path] = []
    for raw_path in re.split(r"[:;,]", raw_value):
        raw_path = normalize_text(raw_path)
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        paths.append(path)
    return paths


@contextmanager
def dictionary_overlay_paths(paths: Iterable[Path | str]):
    """Temporarily append run-scoped dictionary overlay TSVs to the reference lexicon."""

    global _LEXICON_CACHE
    old_env = os.environ.get(OVERLAY_ENV_VAR)
    old_cache = _LEXICON_CACHE
    os.environ[OVERLAY_ENV_VAR] = os.pathsep.join(str(Path(path)) for path in paths)
    _LEXICON_CACHE = None
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop(OVERLAY_ENV_VAR, None)
        else:
            os.environ[OVERLAY_ENV_VAR] = old_env
        _LEXICON_CACHE = old_cache


def _load_lexicon_rows() -> list[dict[str, str]]:
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    rows: list[dict[str, str]] = []
    if FIELD_HEADER_ALIAS_LEXICON.exists():
        rows = _load_tsv_rows(FIELD_HEADER_ALIAS_LEXICON)
    for overlay_path in _overlay_paths_from_env():
        if overlay_path.exists():
            rows.extend(_load_tsv_rows(overlay_path))
    _LEXICON_CACHE = rows
    return rows


def lexicon_rows_for_family(field_family: str) -> list[dict[str, str]]:
    wanted = normalize_text(field_family)
    return [row for row in _load_lexicon_rows() if normalize_text(row.get("field_family")) == wanted]


def _lexicon_row_is_active(row: dict[str, str]) -> bool:
    status = normalize_text(row.get("status") or row.get("review_status") or row.get("promotion_status")).lower()
    if not status:
        return True
    return status in {
        "approved",
        "approved_global",
        "approved_paper_local",
        "candidate",
        "current_run_paper_local",
        "local_only",
    }


def _lexicon_row_matches(row: dict[str, str], value: str) -> bool:
    raw = _canonical_text(value)
    compact = _compact_text(value)
    surface = normalize_text(row.get("surface_form"))
    if not raw or not surface:
        return False
    rule = normalize_text(row.get("normalization_rule")) or "exact"
    surface_raw = _canonical_text(surface)
    surface_compact = _compact_text(surface)
    if rule in {"header_contains", "casefold_contains", "contains"}:
        return bool(surface_raw and surface_raw in raw)
    if rule in {"header_compact_exact", "compact_exact"}:
        return compact == surface_compact
    if rule in {"header_exact", "casefold_exact", "exact"}:
        return raw == surface_raw or compact == surface_compact
    return raw == surface_raw or compact == surface_compact


def _match_lexicon_value(
    *,
    field_family: str,
    value: str,
    paper_key: str = "",
    include_global: bool = True,
) -> list[str]:
    if not _canonical_text(value):
        return []
    paper_key = normalize_text(paper_key)
    scoped_matches: list[str] = []
    global_matches: list[str] = []
    for row in lexicon_rows_for_family(field_family):
        if not _lexicon_row_is_active(row):
            continue
        canonical = normalize_text(row.get("canonical_form"))
        if not canonical or not normalize_text(row.get("surface_form")):
            continue
        scope = normalize_text(row.get("scope")) or "global"
        alias_type = normalize_text(row.get("alias_type"))
        if field_family == "field_name" and alias_type == "drug_specific_mass_header" and scope != "paper_local":
            continue
        if scope == "paper_local":
            row_paper = normalize_text(row.get("paper_key"))
            # Paper-local rows are unsafe without an explicit paper key on both
            # the row and the lookup.  They must never leak into global scope.
            if not row_paper or not paper_key or row_paper != paper_key:
                continue
            if _lexicon_row_matches(row, value):
                scoped_matches.append(canonical)
            continue
        if _lexicon_row_matches(row, value):
            if include_global:
                global_matches.append(canonical)
    # Paper-local definitions override global definitions for the same surface;
    # multiple scoped matches remain ambiguous and are handled by callers.
    matches = scoped_matches if scoped_matches else global_matches
    return sorted(set(matches))


def canonical_field_for_header(header: str, *, paper_key: str = "") -> str:
    raw = _canonical_text(header)
    if not raw:
        return ""
    if re.search(r"\b(?:recovery|release|viability|cytotoxicity|cell\s+uptake|yield)\b", raw):
        return ""
    if re.search(r"\bminor\s+axis\b", raw) or "feret" in raw:
        return ""

    matches = _match_lexicon_value(field_family="field_name", value=header, paper_key=paper_key)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return ""
    if re.search(r"\b(?:mw|m\.?\s*w\.?|mn|molecular\s+weight|weight[-\s]*average\s+molecular\s+weight)\b", raw):
        if re.search(r"\b(?:plga|polymer|pcl|pla|resomer|purasorb|poly\s*\()\b", raw) or re.search(r"\b(?:kda|da|daltons?|g\s*/\s*mol)\b", raw):
            return "polymer_mw_kDa"
    concentration_header = bool(re.search(r"\b(?:mg\s*/\s*ml|mg\s+ml|milligrams?\s*/\s*ml|mg\s+per\s+ml)\b", raw))
    if concentration_header:
        if re.search(r"\b(?:plga|polymer|pcl|pla|resomer)\b", raw):
            return "polymer_concentration_value"
        if re.search(r"\b(?:drug|payload)\b", raw):
            return "drug_concentration_value"
    mass_header = bool(re.search(r"\b(?:mg|milligram|milligrams)\b", raw))
    if mass_header:
        if re.search(r"\b(?:plga|polymer|pcl|pla|resomer)\b", raw):
            return "polymer_mass_mg"
        header_without_mass_unit = re.sub(r"\(?\s*(?:mg|milligram|milligrams)\s*\)?", " ", raw)
        header_without_mass_unit = re.sub(r"[^a-z0-9%:/.+-]+", " ", header_without_mass_unit).strip()
        if header_without_mass_unit:
            drug_matches = _match_lexicon_value(
                field_family="drug_name",
                value=header_without_mass_unit,
                paper_key=paper_key,
                include_global=False,
            )
            if len(drug_matches) == 1:
                return "drug_mass_mg"
        if re.search(r"\b(?:drug|payload)\b", raw):
            return "drug_mass_mg"
    volume_header = bool(re.search(r"\b(?:ml|milliliter|millilitre)\b", raw))
    if volume_header:
        if re.search(r"\b(?:aqueous\s+phase|external\s+aqueous\s+phase|water|aqueous)\b", raw):
            return "external_aqueous_phase_volume_mL"
        if re.search(r"\b(?:organic\s+phase|organic\s+solvent|acetone|acn|dcm|dichloromethane|ethyl\s+acetate|acetonitrile|chloroform|dmso|ethanol|methanol)\b", raw):
            return "O_volume_mL"
    if re.search(r"\b(?:recovery|release|viability|cytotoxicity|cell\s+uptake|yield)\b", raw):
        return ""
    if re.search(r"\bminor\s+axis\b", raw) or "feret" in raw:
        return ""
    if re.search(r"\bmajor\s+axis\b", raw) and re.search(r"\bnm\b", raw):
        return "particle_size_nm"
    if re.search(r"\b(?:sizes?|particle\s+size|diameter|ps|z\s*[- ]?(?:average|ave)|zaverage|zave)\b", raw) and re.search(r"\bnm\b", raw):
        return "particle_size_nm"
    compact_header = _compact_text(header)
    if (
        re.search(r"\bp\.?\s*i\.?\b", raw)
        or compact_header in {"pi", "pisd", "pisdsd"}
        or re.fullmatch(r"pi[a-z]?", compact_header)
        or compact_header.startswith("pi±")
    ):
        return "pdi"
    if re.search(r"\bindex\b", raw) and re.search(r"\bpol[yi]dispersity\b", raw):
        return "pdi"
    if re.search(r"\bee\b", raw) or re.fullmatch(r"ee[a-z]?%?", _compact_text(header)) or _compact_text(header) in {"ee", "ee%", "%ee", "eepercent"}:
        return "ee_percent"
    if re.search(r"\b(?:d\.?\s*l\.?|drug\s+loading)\b", raw) and ("%" in raw or "percent" in raw or "loading" in raw):
        return "dl_percent"
    if re.search(r"\b(?:l\.?\s*c\.?|loading\s+content|drug\s+content)\b", raw) and ("%" in raw or "percent" in raw or "content" in raw):
        return "lc_percent"
    if (
        re.search(r"\bzeta\b", raw)
        or "ζ" in str(header or "")
        or re.search(r"\bzp\w*\b", raw)
        or "zpmv" in _compact_text(header)
    ):
        return "zeta_mV"
    return ""


def normalize_dictionary_value(field_family: str, value: str, *, paper_key: str = "") -> str:
    matches = _match_lexicon_value(field_family=field_family, value=value, paper_key=paper_key)
    if len(matches) == 1:
        return matches[0]
    return normalize_text(value)


def normalize_dictionary_value_from_rows(
    rows: list[dict[str, str]],
    field_family: str,
    value: str,
    *,
    paper_key: str = "",
) -> str:
    """Normalize using an explicit lexicon row set with shared scope semantics.

    Stage5 comparison code builds its lexicon from an execution artifact rather
    than the default reference TSV.  Keep the same global/paper-local priority
    and normalization_rule behavior by temporarily routing through the shared
    matcher instead of maintaining a second implementation.
    """

    global _LEXICON_CACHE
    previous = _LEXICON_CACHE
    _LEXICON_CACHE = rows
    try:
        return normalize_dictionary_value(field_family, value, paper_key=paper_key)
    finally:
        _LEXICON_CACHE = previous


def is_numeric_index_row(row: list[str]) -> bool:
    compact = [normalize_text(cell) for cell in row if normalize_text(cell)]
    if len(compact) < 4:
        return False
    if not all(re.fullmatch(r"\d+", cell) for cell in compact):
        return False
    numbers = [int(cell) for cell in compact]
    start = numbers[0]
    return numbers == list(range(start, start + len(numbers)))


def is_caption_or_metadata_row(row: list[str]) -> bool:
    compact = [normalize_text(cell) for cell in row if normalize_text(cell)]
    if not compact:
        return False
    joined = normalize_text(" ".join(compact)).lower()
    if any(re.search(pattern, joined, re.IGNORECASE) for pattern in METADATA_ROW_PATTERNS):
        return True
    if len(compact) == 1 and joined.endswith(".") and len(joined.split()) >= 4:
        return True
    return False


def looks_like_header_row(row: list[str]) -> bool:
    compact = [normalize_text(cell) for cell in row if normalize_text(cell)]
    if not compact:
        return False
    joined = " ".join(compact).lower()
    if any(re.search(pattern, joined) for pattern in HEADER_SIGNAL_PATTERNS):
        return True
    alpha_cells = sum(1 for cell in compact if re.search(r"[A-Za-z]", cell))
    numeric_cells = sum(1 for cell in compact if re.search(r"\d", cell))
    return alpha_cells >= max(2, numeric_cells)


def looks_like_transposed_metric_data_row(row: list[str]) -> bool:
    """Detect metric-as-row bodies so value rows do not extend a header block.

    Characterization tables often have formulation/status columns and metric
    labels in the first column (e.g. ``Diameter (nm)`` then numeric values). The
    metric label contains header-like words, but once header rows have started it
    is body data, not another header row.  This keeps execution payload geometry
    intact while preventing numeric values from being folded into flattened
    headers.
    """
    cells = [normalize_text(cell) for cell in row]
    if len(cells) < 3:
        return False
    first = cells[0]
    if not first or not canonical_field_for_header(first):
        return False
    trailing = [cell for cell in cells[1:] if cell]
    if len(trailing) < 2:
        return False
    numeric_like = sum(1 for cell in trailing if re.search(r"\d", cell) or cell in {"—", "-", "--", "–"})
    alpha_like = sum(1 for cell in trailing if re.search(r"[A-Za-z]", cell) and not re.search(r"\d", cell))
    return numeric_like >= 2 and numeric_like >= max(2, alpha_like)


def is_explicit_formulation_label(value: Any) -> bool:
    label = normalize_text(value)
    return bool(re.fullmatch(r"\d{1,3}\s*[\.\):]?", label) or re.fullmatch(r"[Ff]\s*[- ]?\d{1,3}\s*[\.\):]?", label))


def infer_header_structure(rows: list[list[str]]) -> dict[str, Any]:
    if not rows:
        return {
            "header_row_count": 0,
            "column_count": 0,
            "header_rows": [],
            "flattened_headers": [],
            "header_hierarchy_detected": False,
        }
    cleaned_rows = [[normalize_text(cell) for cell in row] for row in rows if isinstance(row, list)]
    column_count = max((len(row) for row in cleaned_rows), default=0)
    header_rows: list[list[str]] = []
    for row in cleaned_rows[:8]:
        compact = [cell for cell in row if cell]
        if not compact:
            continue
        if is_numeric_index_row(row) or is_caption_or_metadata_row(row):
            continue
        if header_rows and looks_like_transposed_metric_data_row(row):
            break
        if looks_like_header_row(row):
            header_rows.append(row)
            continue
        if header_rows:
            break
    if not header_rows:
        for row in cleaned_rows:
            if is_numeric_index_row(row) or is_caption_or_metadata_row(row):
                continue
            header_rows = [row]
            break
    flattened_headers = flatten_header_rows(header_rows, column_count)
    return {
        "header_row_count": len(header_rows),
        "column_count": column_count,
        "header_rows": header_rows,
        "flattened_headers": flattened_headers,
        "header_hierarchy_detected": len(header_rows) > 1,
    }


def flatten_header_rows(header_rows: list[list[str]], column_count: int) -> list[str]:
    flattened: list[str] = []
    for column_index in range(column_count):
        values: list[str] = []
        for row in header_rows:
            if column_index >= len(row):
                continue
            cell = normalize_text(row[column_index])
            if cell and cell.lower() not in {item.lower() for item in values}:
                values.append(cell)
        flattened.append(normalize_text(" ".join(values)))

    duplicate_headers = {
        value.lower()
        for value in flattened
        if value and sum(1 for item in flattened if item.lower() == value.lower()) > 1
    }
    if duplicate_headers:
        for column_index, value in enumerate(list(flattened)):
            if not value or value.lower() not in duplicate_headers:
                continue
            inherited_parts: list[str] = []
            for row in header_rows[:-1]:
                if column_index >= len(row) or normalize_text(row[column_index]):
                    continue
                for left_index in range(column_index - 1, -1, -1):
                    left = normalize_text(row[left_index]) if left_index < len(row) else ""
                    if left:
                        inherited_parts.append(left)
                        break
            if inherited_parts:
                prefix = normalize_text(inherited_parts[-1])
                if prefix and prefix.lower() not in value.lower():
                    flattened[column_index] = normalize_text(f"{prefix} {value}")
    return flattened


def recover_prelude_header_block(
    row_entries: list[dict[str, Any]],
    *,
    first_explicit_row_index: int,
    width: int,
) -> list[list[str]]:
    prelude: list[list[str]] = []
    for entry in row_entries:
        if int(entry.get("row_index") or 0) >= first_explicit_row_index:
            continue
        cells = [normalize_text(cell) for cell in (entry.get("cells") or [])]
        if not any(cells):
            continue
        if is_numeric_index_row(cells) or is_caption_or_metadata_row(cells):
            continue
        prelude.append(cells)
    if not prelude:
        return []
    trailing: list[list[str]] = []
    for cells in reversed(prelude):
        if looks_like_header_row(cells) or trailing:
            trailing.append(cells)
            if len(trailing) >= 4:
                break
        elif trailing:
            break
    trailing.reverse()
    return [row[:width] + [""] * max(0, width - len(row)) for row in trailing]


def recover_row_entry_headers(
    row_entries: list[dict[str, Any]],
    *,
    first_explicit_row_index: int,
    width: int,
) -> list[str]:
    header_rows = recover_prelude_header_block(
        row_entries,
        first_explicit_row_index=first_explicit_row_index,
        width=width,
    )
    return flatten_header_rows(header_rows, width)


def _nonempty_positions(cells: list[str]) -> list[int]:
    return [idx for idx, cell in enumerate(cells) if normalize_text(cell)]


def _join_continuation_fragments(parts: list[str]) -> str:
    output = ""
    for part in parts:
        text = normalize_text(part)
        if not text:
            continue
        if not output:
            output = text
        elif SYMBOL_SUFFIX_PATTERN.fullmatch(text):
            output = f"{output}{text}"
        else:
            output = f"{output} {text}"
    return normalize_text(output)


def extract_continuation_group_context(
    entry: dict[str, Any],
    *,
    variable_columns: list[dict[str, Any]],
    paper_key: str = "",
) -> dict[str, str]:
    cells = [normalize_text(cell) for cell in (entry.get("cells") or [])]
    if not cells or is_explicit_formulation_label(cells[0]):
        return {}
    active_columns = {int(col.get("column_index") or -1): col for col in variable_columns}
    variable_parts: dict[int, list[str]] = {}
    measurement_like_outside = False
    for index, cell in enumerate(cells):
        text = normalize_text(cell)
        if not text:
            continue
        if index == 0:
            continue
        if index not in active_columns:
            if re.search(r"\d", text):
                measurement_like_outside = True
                break
            continue
        variable_parts.setdefault(index, []).append(text)
    if measurement_like_outside or not variable_parts:
        return {}

    ordered_context: dict[str, str] = {}
    last_field = ""
    for index in sorted(variable_parts):
        header = normalize_text(active_columns[index].get("header"))
        field = normalize_text(active_columns[index].get("canonical_field")) or canonical_field_for_header(header, paper_key=paper_key)
        joined = _join_continuation_fragments(variable_parts[index])
        if not joined:
            continue
        if SYMBOL_SUFFIX_PATTERN.fullmatch(joined) and last_field and last_field in ordered_context:
            ordered_context[last_field] = f"{ordered_context[last_field]}{joined}"
            continue
        if field:
            ordered_context[field] = joined
            last_field = field
    return ordered_context
