#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        call_gemini,
        call_nvidia_hosted,
        sanitize_stage2_json_text,
    )
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
        call_gemini,
        call_nvidia_hosted,
        sanitize_stage2_json_text,
    )
    from src.utils.model_policy import PRIMARY_DEFAULT, validate_models_or_raise


SCHEMA = {
    "selection_markers": [
        {
            "marker_readiness": "execution_ready|partial_semantic",
            "source_table_id": "string",
            "selected_variable": "string",
            "selected_value": "string",
            "explicit": True,
            "evidence_span": "string",
        }
    ],
    "inheritance_markers": [
        {
            "marker_readiness": "execution_ready|partial_semantic",
            "from_table": "string",
            "to_table": "string",
            "inherit_type": "selected_condition",
            "variable": "string",
            "value": "string",
            "evidence_span": "string",
        }
    ],
}

EVIDENCE_TEXT = (
    "Poloxamer 188 concentration was studied at 2.5, 3, 4, and 10 mg/mL. "
    "3 mg/mL was selected as the optimal surfactant concentration. "
    "After the optimal surfactant concentration had been determined, the remaining studies used that condition."
)

VARIANT_RULES: dict[str, str] = {
    "A_baseline_permissive": (
        "- Sequential-optimization literal-value rule: when a variable is explicitly explored over multiple values, "
        "a specific value is explicitly selected as optimal in nearby text, and later text explicitly says that this "
        "selected value is reused or carried forward in following experiments, you may extract that literal value and keep it literal.\n"
        "- For that rule, you may emit an execution_ready selection_marker with the literal selected_value and an "
        "execution_ready inheritance_marker with inherit_type='selected_condition' and the same literal value.\n"
        "- This is allowed when the selection sentence and the reuse sentence appear in different nearby sentences or adjacent paragraphs.\n"
        "- Do not guess values.\n"
    ),
    "B_current_must": (
        "- Mandatory sequential-optimization literal-value rule: when a variable is explicitly explored over multiple concrete values, "
        "the paper explicitly states that one concrete value was selected or chosen as optimal, and nearby later text explicitly says "
        "that this chosen optimal setting was reused or carried forward in following experiments, you MUST extract that literal chosen value and keep it literal.\n"
        "- For that rule, you MUST emit an execution_ready selection_marker with the literal selected_value and an execution_ready inheritance_marker "
        "with inherit_type='selected_condition' and the same literal value.\n"
        "- For that rule, the selection sentence and the reuse sentence may appear in the same paragraph, the adjacent paragraph, or the nearby discussion immediately "
        "following the relevant optimization table.\n"
        "- For that rule, the optimal value MUST be explicitly stated as a concrete literal such as '3 mg/mL' or '10:1', and reuse MUST be explicitly stated.\n"
        "- Anti-placeholder rule: you MUST NOT output abstract placeholders such as 'optimal concentration', 'optimal ratio', or 'optimal formulation' when a concrete "
        "literal value is explicitly stated in that nearby local evidence window.\n"
        "- If no explicit literal value is present in that nearby local evidence window, abstract wording alone must stay partial_semantic and must NOT become execution_ready.\n"
        "- Do not guess values and do not infer them from unrelated tables, distant sections, or full-document fallback.\n"
    ),
    "C_strengthened_violation": (
        "- Mandatory sequential-optimization literal-value rule: when a variable is explicitly explored over multiple concrete values, "
        "the paper explicitly states that one concrete value was selected or chosen as optimal, and nearby later text explicitly says "
        "that this chosen optimal setting was reused or carried forward in following experiments, you MUST extract that literal chosen value and keep it literal.\n"
        "- Before emitting selection_marker or inheritance_marker, you MUST check whether a concrete literal value for the selected variable appears in the nearby local evidence window. "
        "If yes, you MUST use that literal value.\n"
        "- When both abstract wording and a concrete literal value are present, literal value binding has absolute priority over abstract summarization.\n"
        "- If a concrete literal value exists in the nearby local evidence window and you output an abstract placeholder instead, this is a critical extraction error.\n"
        "- You MUST emit an execution_ready selection_marker with selected_value equal to the literal chosen value and an execution_ready inheritance_marker with value equal to that same literal.\n"
        "- You MUST NOT output abstract placeholders such as 'optimal concentration', 'optimal ratio', or 'optimal formulation' when the literal value is explicitly present nearby.\n"
        "- Nearby local evidence means the same paragraph, the adjacent paragraph, or the nearby discussion immediately following the relevant optimization table. "
        "Do not guess from distant sections or unrelated tables.\n"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic-only A/B/C prompt-strength experiment for the QLYKLPKT sequential-optimization grounding problem."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llm-backend", choices=["gemini", "nvidia"], default="gemini")
    parser.add_argument("--model", default=PRIMARY_DEFAULT)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    return parser.parse_args()


def build_prompt(rule_block: str) -> str:
    return (
        "You are running a diagnostic-only Stage2 prompt-strength test.\n"
        "Task:\n"
        "- Read only the local evidence window below.\n"
        "- Extract only selection_markers and inheritance_markers.\n"
        "- Return valid JSON only.\n"
        "- Use the exact schema shown below.\n"
        "- Do not add any top-level keys beyond the schema.\n"
        "- Do not use distant-section reasoning, full-document guessing, or unrelated table inference.\n"
        "Rules:\n"
        f"{rule_block}"
        "Schema:\n"
        f"{json.dumps(SCHEMA, ensure_ascii=True, indent=2)}\n"
        "Local evidence window:\n"
        f"{EVIDENCE_TEXT}\n"
    )


def call_backend(backend: str, model: str, prompt: str, retries: int, sleep_sec: float, *, label: str) -> str:
    if backend == "gemini":
        return call_gemini(model, prompt, retries, sleep_sec, progress_label=label)
    return call_nvidia_hosted(model, prompt, retries, sleep_sec, progress_label=label)


def parse_response(raw_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    sanitized_text, audit = sanitize_stage2_json_text(raw_text)
    parsed = json.loads(sanitized_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a top-level JSON object.")
    return parsed, audit


def summarize_marker(marker: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: str(marker.get(field, "") or "") for field in fields}


def variant_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    selection = [
        summarize_marker(
            item,
            [
                "marker_readiness",
                "selected_variable",
                "selected_value",
                "source_table_id",
                "evidence_span",
                "risk_label",
            ],
        )
        for item in parsed.get("selection_markers") or []
        if isinstance(item, dict)
    ]
    inheritance = [
        summarize_marker(
            item,
            [
                "marker_readiness",
                "variable",
                "value",
                "from_table",
                "to_table",
                "evidence_span",
                "risk_label",
            ],
        )
        for item in parsed.get("inheritance_markers") or []
        if isinstance(item, dict)
    ]
    values = [item.get("selected_value", "") for item in selection] + [item.get("value", "") for item in inheritance]
    literal_binding = any(value in {"3 mg/mL", "10:1"} for value in values)
    abstract_fallback = any(value in {"optimal concentration", "optimal ratio", "optimal formulation"} for value in values)
    selection_execution_ready = any(item.get("marker_readiness") == "execution_ready" for item in selection)
    inheritance_execution_ready = any(item.get("marker_readiness") == "execution_ready" for item in inheritance)
    return {
        "selection_markers": selection,
        "inheritance_markers": inheritance,
        "selection_execution_ready": selection_execution_ready,
        "inheritance_execution_ready": inheritance_execution_ready,
        "literal_binding_detected": literal_binding,
        "abstract_fallback_detected": abstract_fallback,
    }


def build_report(results: dict[str, Any]) -> str:
    a = results["variants"]["A_baseline_permissive"]
    b = results["variants"]["B_current_must"]
    c = results["variants"]["C_strengthened_violation"]

    def status_line(name: str, payload: dict[str, Any]) -> str:
        return (
            f"- `{name}`: literal_binding=`{payload['literal_binding_detected']}`; "
            f"abstract_fallback=`{payload['abstract_fallback_detected']}`; "
            f"selection_execution_ready=`{payload['selection_execution_ready']}`; "
            f"inheritance_execution_ready=`{payload['inheritance_execution_ready']}`"
        )

    if c["literal_binding_detected"] and not b["literal_binding_detected"]:
        conclusion = "The evidence points mainly to prompt-strength sensitivity under controlled local evidence."
    elif not a["literal_binding_detected"] and not b["literal_binding_detected"] and not c["literal_binding_detected"]:
        conclusion = "The evidence suggests the bottleneck is deeper than wording strength alone, likely model behavior under the fixed schema or local evidence interpretation."
    elif b["literal_binding_detected"] and not a["literal_binding_detected"]:
        conclusion = "The evidence suggests the current MUST wording already improves grounding over the permissive baseline."
    else:
        conclusion = "The evidence is mixed; prompt wording changes affect behavior, but the effect is not cleanly monotonic across variants."

    return "\n".join(
        [
            "# QLYKLPKT Prompt Strength Diagnostic",
            "",
            "- diagnostic_only: `true`",
            f"- llm_backend: `{results['llm_backend']}`",
            f"- model: `{results['model']}`",
            "- evidence_scope: `single synthetic local evidence window`",
            "",
            "## Variant Results",
            status_line("A_baseline_permissive", a),
            status_line("B_current_must", b),
            status_line("C_strengthened_violation", c),
            "",
            "## Comparisons",
            f"- Variant B improved over A: `{b['literal_binding_detected'] and not a['literal_binding_detected']}`",
            f"- Variant C improved over B: `{c['literal_binding_detected'] and not b['literal_binding_detected']}`",
            "",
            "## Conclusion",
            f"- {conclusion}",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    validate_models_or_raise([args.model], context="QLYK prompt-strength diagnostic")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "schema": "qlyk_prompt_strength_diagnostic_v1",
        "diagnostic_only": True,
        "llm_backend": args.llm_backend,
        "model": args.model,
        "temperature": 0,
        "evidence_text": EVIDENCE_TEXT,
        "variants": {},
    }

    for variant_name, rule_block in VARIANT_RULES.items():
        prompt = build_prompt(rule_block)
        raw_text = call_backend(
            args.llm_backend,
            args.model,
            prompt,
            args.request_retries,
            args.retry_sleep_sec,
            label=f"prompt_strength_diag[{variant_name}]",
        )
        raw_path = raw_dir / f"{variant_name}.json"
        raw_path.write_text(raw_text, encoding="utf-8")
        parsed, audit = parse_response(raw_text)
        results["variants"][variant_name] = {
            "rule_block": rule_block,
            "raw_response_path": str(raw_path),
            "parse_audit": audit,
            "parsed": parsed,
            **variant_summary(parsed),
        }

    json_path = out_dir / "qlyk_prompt_strength_diagnostic_v1.json"
    report_path = out_dir / "qlyk_prompt_strength_diagnostic_v1.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(build_report(results), encoding="utf-8")

    print(f"wrote_json={json_path}")
    print(f"wrote_report={report_path}")


if __name__ == "__main__":
    main()
