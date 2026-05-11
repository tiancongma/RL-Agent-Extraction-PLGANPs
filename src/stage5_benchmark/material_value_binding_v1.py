"""Generic material/value binding helpers for Stage5 direct materialization.

This module is deliberately side-effect free. It does not read raw sources,
create formulation rows, call LLMs, or inspect GT. It provides reusable generic
helpers for source-backed material alias resolution, direct value validation,
entity-bound extraction, scope classification, and canonical promotion proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


_MASS_UNIT_RE = re.compile(r"\b(?:mg|g|µg|ug|nanogram|microgram|milligram|gram)s?\b", re.I)
_VOLUME_UNIT_RE = re.compile(r"\b(?:ml|mL|µL|uL|l|liter|litre)s?\b", re.I)
_CONC_UNIT_RE = re.compile(r"\b(?:mg\s*/\s*mL|mg\s*/\s*ml|µg\s*/\s*mL|ug\s*/\s*mL|%\s*w\s*/\s*v|%\s*\(?w/v\)?|%|w/v)\b", re.I)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg\s*/\s*kg|mg\s*/\s*mL|mg\s*/\s*ml|µg\s*/\s*mL|ug\s*/\s*mL|mg|g|µg|ug|mL|ml|µL|uL|%)\b", re.I)
_FULL_ABBR_RE = re.compile(r"(?P<full>[A-Za-z][A-Za-z0-9,\-\s/]+?)\s*\((?P<abbr>[A-Z][A-Z0-9]{1,9})\)")

_POLYMER_CUES = ("plga", "poly(lactic", "poly lactic", "poly(lactide", "polymer")
_DRUG_CUES = ("drug", "curcumin", "paclitaxel", "doxorubicin", "docetaxel", "cisplatin", "rifampicin")
_SURFACTANT_CUES = ("pva", "polyvinyl alcohol", "surfactant", "stabilizer", "stabiliser", "poloxamer", "tween")
_SOLVENT_CUES = ("acetone", "dichloromethane", "ethyl acetate", "solvent", "chloroform")
_PREPARATION_CUES = (
    "prepared",
    "preparation",
    "dissolved",
    "weighed",
    "mixed",
    "added",
    "dropwise",
    "emulsified",
    "stirred",
    "evaporated",
    "nanoparticle",
    "organic phase",
    "aqueous phase",
)
_NEGATIVE_CONTEXT_CUES = (
    "animal study",
    "rats received",
    "mice received",
    "intravenous injection",
    "injected",
    "dose",
    "mg/kg",
    "release",
    "cell uptake",
    "cytotoxicity",
    "pharmacokinetic",
    "auc",
    "cmax",
    "sample preparation",
    "sample-prep",
    "sample prep",
    "assay injection",
)


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm_key(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text or "") if part.strip()]


def _classify_material_role(term: str, local_context: str = "") -> str:
    hay = f"{term} {local_context}".lower()
    term_hay = str(term or "").lower()
    if "drug" in hay or any(cue in term_hay for cue in _DRUG_CUES if cue != "drug"):
        return "drug"
    if "polymer" in hay or any(cue in term_hay for cue in _POLYMER_CUES if cue != "polymer"):
        return "polymer"
    if any(cue in hay for cue in _SURFACTANT_CUES):
        return "surfactant"
    if any(cue in hay for cue in _SOLVENT_CUES):
        return "solvent"
    return "unknown"


def _canonical_unit(unit: str) -> str:
    compact = re.sub(r"\s+", "", unit or "")
    lower = compact.lower()
    if lower == "ug":
        return "ug"
    if lower in {"ml", "mg/ml", "mg/kg"}:
        return lower.replace("ml", "mL")
    return compact


def validate_direct_value(expression: str, value_type: str) -> Dict[str, str]:
    """Validate a direct value expression for mass/volume/concentration fields."""
    expr = _norm(expression)
    kind = (value_type or "").strip().lower()
    percent_match = re.search(r"\b(\d+(?:\.\d+)?)\s*%(?:\s*\(?\s*w\s*/\s*v\s*\)?)?", expr, re.I)
    if kind in {"concentration", "concentration_value"} and percent_match:
        return {
            "status": "valid",
            "reason": "",
            "normalized_value": percent_match.group(1),
            "normalized_unit": "%",
        }
    match = _NUMBER_RE.search(expr)
    if not match:
        return {"status": "invalid", "reason": f"invalid_{kind}_no_numeric_value", "normalized_value": "", "normalized_unit": ""}

    unit = match.group("unit")
    normalized_unit = _canonical_unit(unit)
    if kind == "mass":
        if "/" in unit or "kg" in unit.lower():
            return {"status": "invalid", "reason": "invalid_mass_concentration_only", "normalized_value": "", "normalized_unit": ""}
        if not _MASS_UNIT_RE.search(unit):
            return {"status": "invalid", "reason": "invalid_mass_missing_mass_unit", "normalized_value": "", "normalized_unit": ""}
    elif kind == "volume":
        if not _VOLUME_UNIT_RE.search(unit) or "/" in unit:
            return {"status": "invalid", "reason": "invalid_volume_missing_volume_unit", "normalized_value": "", "normalized_unit": ""}
    elif kind in {"concentration", "concentration_value"}:
        if not _CONC_UNIT_RE.search(expr):
            return {"status": "invalid", "reason": "invalid_concentration_missing_concentration_unit", "normalized_value": "", "normalized_unit": ""}
    else:
        return {"status": "invalid", "reason": "invalid_unknown_value_type", "normalized_value": "", "normalized_unit": ""}

    return {
        "status": "valid",
        "reason": "",
        "normalized_value": match.group("value"),
        "normalized_unit": normalized_unit,
    }


@dataclass
class MaterialAliasGraph:
    aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def add_alias(self, alias: str, role: str, canonical_name: Optional[str] = None, source: str = "cleaned_text") -> None:
        token = _norm(alias)
        if not token:
            return
        key = _norm_key(token)
        current = self.aliases.get(key)
        if current and current.get("role") != "unknown":
            return
        self.aliases[key] = {
            "alias": token,
            "canonical_name": _norm(canonical_name or token),
            "role": role or "unknown",
            "source": source,
        }

    def resolve(self, alias: str) -> Dict[str, str]:
        return self.aliases.get(_norm_key(alias), {"alias": _norm(alias), "canonical_name": _norm(alias), "role": "unknown", "source": "unresolved"})

    def resolve_role(self, alias: str) -> str:
        return self.resolve(alias).get("role", "unknown")

    def known_aliases(self) -> List[Dict[str, str]]:
        return sorted(self.aliases.values(), key=lambda row: (row.get("role", ""), row.get("alias", "").lower()))


def build_material_alias_graph(cleaned_text: str, row_hints: Optional[Iterable[Mapping[str, str]]] = None) -> MaterialAliasGraph:
    """Build a source-backed paper-local alias graph from cleaned text and row hints."""
    graph = MaterialAliasGraph()
    text = _norm(cleaned_text)
    for match in _FULL_ABBR_RE.finditer(text):
        full = _norm(match.group("full"))
        full = re.sub(r"^(?:the\s+)?(?:drug|polymer|stabilizer|stabiliser|surfactant)\s+", "", full, flags=re.I)
        abbr = _norm(match.group("abbr"))
        context = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)]
        role = _classify_material_role(full, context)
        graph.add_alias(full, role, canonical_name=full)
        graph.add_alias(abbr, role, canonical_name=full)

    # Generic explicit PLGA/PVA mentions without relying on paper identity.
    if re.search(r"\bPLGA\b|poly\(lactic|poly\(lactide", text, re.I):
        graph.add_alias("PLGA", "polymer", canonical_name="PLGA")
    if re.search(r"\bPVA\b|polyvinyl alcohol", text, re.I):
        graph.add_alias("PVA", "surfactant", canonical_name="PVA")

    for hint in row_hints or []:
        for key in (
            "material",
            "material_name",
            "component_name",
            "drug_name",
            "drug_name_value",
            "polymer_name",
            "polymer_name_value",
            "surfactant_name",
            "surfactant_name_value",
            "emulsifier_stabilizer_name",
            "emulsifier_stabilizer_name_value",
            "stabilizer_name",
            "stabilizer_name_value",
            "formulation_identity_label",
        ):
            value = _norm(hint.get(key, ""))
            if not value:
                continue
            role = hint.get("role") or key.replace("_value", "").replace("_name", "")
            if role in {"material", "component", "formulation_identity_label"}:
                role = _classify_material_role(value, str(hint))
            if role in {"emulsifier_stabilizer", "stabilizer"}:
                role = "surfactant"
            graph.add_alias(value, role, canonical_name=value, source="row_hint")
    return graph


def _is_negative_context(sentence: str) -> bool:
    lower = sentence.lower()
    return any(cue in lower for cue in _NEGATIVE_CONTEXT_CUES)


def _is_preparation_context(sentence: str) -> bool:
    lower = sentence.lower()
    return any(cue in lower for cue in _PREPARATION_CUES)


def extract_entity_bound_values(cleaned_text: str, alias_graph: MaterialAliasGraph) -> List[Dict[str, str]]:
    """Extract direct entity-bound values from preparation-positive text contexts."""
    candidates: List[Dict[str, str]] = []
    seen = set()
    for sentence in _split_sentences(cleaned_text):
        if _is_negative_context(sentence) or not _is_preparation_context(sentence):
            continue
        roles_in_sentence = {
            info.get("role", "unknown")
            for info in alias_graph.known_aliases()
            if info.get("role") not in {"unknown", "solvent", "helper_material"}
            and re.search(rf"\b{re.escape(info.get('alias', ''))}\b", sentence, re.I)
        }
        shared_scope_basis = "method_shared_preparation_pair" if len(roles_in_sentence) >= 2 else ""
        for alias_info in alias_graph.known_aliases():
            alias = alias_info["alias"]
            role = alias_info.get("role", "unknown")
            if role in {"unknown", "solvent", "helper_material"}:
                continue
            alias_pattern = re.escape(alias)
            patterns = [
                re.compile(rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|µg|ug)\s+of\s+(?P<alias>{alias_pattern})\b", re.I),
                re.compile(rf"(?P<alias>{alias_pattern})\s*\(\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|µg|ug)\s*\)", re.I),
            ]
            for pattern in patterns:
                for match in pattern.finditer(sentence):
                    expression = f"{match.group('value')} {match.group('unit')}"
                    validation = validate_direct_value(expression, "mass")
                    if validation["status"] != "valid":
                        continue
                    key = (alias.upper(), validation["normalized_value"], validation["normalized_unit"], sentence)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "material_alias": alias,
                            "canonical_material": alias_info.get("canonical_name", alias),
                            "entity_role": role,
                            "value_type": "mass",
                            "raw_value_expression": expression,
                            "normalized_value": validation["normalized_value"],
                            "normalized_unit": validation["normalized_unit"],
                            "source_span": sentence,
                            "source_provenance": "direct_text",
                            "source_scope_basis": shared_scope_basis,
                            "extraction_reason": "entity_bound_preparation_context",
                        }
                    )
    return candidates


def infer_value_scope(candidate: Mapping[str, str], admitted_rows: Optional[Sequence[Mapping[str, str]]] = None) -> Dict[str, str]:
    """Infer safe generic scope for a value candidate without creating rows."""
    scope_hint = _norm(candidate.get("scope_hint", "")).lower()
    row_id = _norm(candidate.get("final_formulation_id", "") or candidate.get("row_id", ""))
    if row_id:
        return {"scope_type": "row_local", "scope_reason": "candidate_has_row_identifier", "applies_to": row_id}
    if scope_hint in {"table_footnote", "table_scoped", "method_shared", "paper_global"}:
        return {"scope_type": scope_hint, "scope_reason": "candidate_scope_hint", "applies_to": "admitted_rows"}
    rows = admitted_rows or []
    if rows and candidate.get("source_provenance") == "direct_text":
        if candidate.get("source_scope_basis") == "method_shared_preparation_pair":
            return {"scope_type": "method_shared", "scope_reason": "direct_preparation_pair_context", "applies_to": "admitted_rows"}
        return {"scope_type": "ambiguous", "scope_reason": "direct_text_without_shared_scope_basis", "applies_to": ""}
    return {"scope_type": "ambiguous", "scope_reason": "no_admitted_row_scope", "applies_to": ""}


def _canonical_field_for(candidate: Mapping[str, str]) -> str:
    role = candidate.get("entity_role", "")
    value_type = candidate.get("value_type", "")
    if value_type == "mass" and role == "polymer":
        return "polymer_mass_mg"
    if value_type == "mass" and role == "drug":
        return "drug_mass_mg"
    if value_type == "mass" and role == "surfactant":
        return "surfactant_mass_mg"
    if value_type in {"concentration", "concentration_value"} and role == "surfactant":
        return "surfactant_concentration_value"
    return ""


def _candidate_rejection(candidate: Mapping[str, str], reason: str, field_name: str = "") -> Dict[str, str]:
    return {
        "canonical_field": field_name or _canonical_field_for(candidate),
        "material_alias": _norm(candidate.get("material_alias", "")),
        "entity_role": _norm(candidate.get("entity_role", "")),
        "normalized_value": _norm(candidate.get("normalized_value", "")),
        "normalized_unit": _norm(candidate.get("normalized_unit", "")),
        "scope_type": _norm(candidate.get("scope_type", "")),
        "source_span": _norm(candidate.get("source_span", "")),
        "rejection_reason": reason,
    }


def evaluate_canonical_promotions(candidates: Sequence[Mapping[str, str]], admitted_rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Return auditable direct-value promotion proposals and rejection rows.

    Conflicting shared values for the same canonical field are rejected as a
    group rather than donor-filled across all admitted rows.
    """
    proposals: List[Dict[str, str]] = []
    rejections: List[Dict[str, str]] = []
    scoped_candidates: List[Dict[str, str]] = []

    for candidate in candidates:
        field_name = _canonical_field_for(candidate)
        candidate_copy = dict(candidate)
        if not field_name:
            rejections.append(_candidate_rejection(candidate_copy, "no_canonical_field_mapping"))
            continue
        if candidate_copy.get("source_provenance") != "direct_text":
            rejections.append(_candidate_rejection(candidate_copy, "not_direct_text_source", field_name))
            continue
        if candidate_copy.get("normalized_unit") != "mg" and field_name.endswith("_mass_mg"):
            rejections.append(_candidate_rejection(candidate_copy, "invalid_mass_unit_for_mg_field", field_name))
            continue
        scope = candidate_copy.get("scope_type") or infer_value_scope(candidate_copy, admitted_rows).get("scope_type")
        candidate_copy["scope_type"] = scope
        if scope not in {"row_local", "typed_row_local_assignment", "table_footnote", "table_scoped", "method_shared", "paper_global"}:
            rejections.append(_candidate_rejection(candidate_copy, "ambiguous_or_unsupported_scope", field_name))
            continue
        scoped_candidates.append(candidate_copy)

    shared_groups: Dict[tuple, set] = {}
    for candidate in scoped_candidates:
        scope = candidate.get("scope_type", "")
        field_name = _canonical_field_for(candidate)
        if scope in {"table_footnote", "table_scoped", "method_shared", "paper_global"}:
            shared_groups.setdefault((field_name, scope), set()).add(
                (_norm(candidate.get("normalized_value", "")), _norm(candidate.get("normalized_unit", "")))
            )
    conflicted_groups = {key for key, values in shared_groups.items() if len(values) > 1}

    admitted_row_ids = {
        _norm(row.get("final_formulation_id", "") or row.get("row_id", ""))
        for row in admitted_rows
        if _norm(row.get("final_formulation_id", "") or row.get("row_id", ""))
    }

    for candidate in scoped_candidates:
        field_name = _canonical_field_for(candidate)
        scope = candidate.get("scope_type", "")
        if (field_name, scope) in conflicted_groups:
            rejections.append(_candidate_rejection(candidate, "conflicting_shared_values_for_field", field_name))
            continue
        candidate_row_id = _norm(candidate.get("final_formulation_id", "") or candidate.get("row_id", ""))
        if scope in {"row_local", "typed_row_local_assignment"}:
            if not candidate_row_id:
                rejections.append(_candidate_rejection(candidate, "row_local_missing_row_identifier", field_name))
                continue
            if candidate_row_id not in admitted_row_ids:
                rejections.append(_candidate_rejection(candidate, "row_local_target_not_admitted", field_name))
                continue
        for row in admitted_rows:
            row_id = _norm(row.get("final_formulation_id", "") or row.get("row_id", ""))
            if not row_id:
                continue
            if scope in {"row_local", "typed_row_local_assignment"} and candidate_row_id != row_id:
                continue
            if _norm(row.get(field_name, "")):
                rejections.append(_candidate_rejection(candidate, "target_field_already_has_higher_authority_value", field_name))
                continue
            proposals.append(
                {
                    "final_formulation_id": row_id,
                    "canonical_field": field_name,
                    "normalized_value": _norm(candidate.get("normalized_value", "")),
                    "normalized_unit": _norm(candidate.get("normalized_unit", "")),
                    "material_alias": _norm(candidate.get("material_alias", "")),
                    "entity_role": _norm(candidate.get("entity_role", "")),
                    "scope_type": scope,
                    "source_span": _norm(candidate.get("source_span", "")),
                    "promotion_status": "proposed_direct",
                    "promotion_reason": "role_value_type_scope_valid",
                }
            )
    return {"proposals": proposals, "rejections": rejections}


def propose_canonical_promotions(candidates: Sequence[Mapping[str, str]], admitted_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    """Return direct-value promotion proposals for already admitted rows only."""
    return evaluate_canonical_promotions(candidates, admitted_rows)["proposals"]
