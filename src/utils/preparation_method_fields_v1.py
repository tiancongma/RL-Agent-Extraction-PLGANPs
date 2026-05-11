from __future__ import annotations

import re
from typing import Any, Mapping


PREPARATION_METHOD_FIELDNAMES = [
    "preparation_method",
    "emulsion_structure",
]


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def _route_from_text(text: str) -> str:
    compact = text.replace(" ", "")
    if any(token in compact for token in ["w1/o/w2", "w/o/w2", "w/o/w", "water-in-oil-in-water"]):
        return "W1/O/W2"
    if any(token in compact for token in ["o/w", "oil-in-water"]):
        return "O/W"
    if any(token in compact for token in ["w/o", "water-in-oil"]):
        return "W/O"
    return ""


def _combined_method_text(row: Mapping[str, Any]) -> str:
    parts = [
        row.get("preparation_method", ""),
        row.get("emulsion_structure", ""),
        row.get("emul_method_value", ""),
        row.get("emul_method_value_text", ""),
        row.get("emul_type_value", ""),
        row.get("emul_type_value_text", ""),
        row.get("emulsion_route_tag", ""),
        row.get("raw_formulation_label", ""),
        row.get("representative_source_raw_formulation_label", ""),
        row.get("evidence_span_text", ""),
    ]
    return " ".join(_normalize_text(part) for part in parts if str(part or "").strip())


def derive_preparation_method_fields_v1(row: Mapping[str, Any]) -> dict[str, str]:
    text = _combined_method_text(row)
    route = _route_from_text(text) or str(row.get("emulsion_route_tag", "") or "").strip()
    route_upper = route.upper()
    route_upper = "W1/O/W2" if route_upper in {"W/O/W", "W/O/W2"} else route_upper

    emulsion_structure = "none"
    if route_upper in {"W1/O/W2", "O/W", "W/O"}:
        emulsion_structure = route_upper

    has_emulsion_context = any(
        token in text
        for token in [
            "emulsion",
            "single emulsion",
            "double emulsion",
            "oil-in-water",
            "water-in-oil",
        ]
    ) or emulsion_structure != "none"

    preparation_method = "unknown"
    if "microfluidic" in text:
        preparation_method = "microfluidic"
    elif "salting out" in text or "salting-out" in text:
        preparation_method = "salting_out"
    elif "nanoprecipitation" in text or "solvent displacement" in text:
        preparation_method = "nanoprecipitation"
    elif "double emulsion" in text or route_upper == "W1/O/W2":
        preparation_method = "double_emulsion"
    elif "single emulsion" in text:
        preparation_method = "single_emulsion"
    elif "solvent diffusion" in text and has_emulsion_context:
        preparation_method = "emulsion_diffusion"
    elif "solvent evaporation" in text and has_emulsion_context:
        preparation_method = "emulsion_solvent_evaporation"

    if emulsion_structure == "none" and preparation_method == "double_emulsion":
        if "w1/o/w2" in text or "w/o/w" in text or "water-in-oil-in-water" in text:
            emulsion_structure = "W1/O/W2"
    if emulsion_structure == "none" and preparation_method in {
        "single_emulsion",
        "emulsion_diffusion",
        "emulsion_solvent_evaporation",
    }:
        route_from_text = _route_from_text(text)
        if route_from_text in {"O/W", "W/O"}:
            emulsion_structure = route_from_text

    return {
        "preparation_method": preparation_method,
        "emulsion_structure": emulsion_structure,
    }


def enrich_preparation_method_fields_v1(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched.update(derive_preparation_method_fields_v1(enriched))
    return enriched
