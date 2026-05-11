#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from src.stage2_sampling_labels.ollama_local_backend_v1 import check_ollama_connectivity, resolve_ollama_base_url
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.stage2_sampling_labels.ollama_local_backend_v1 import check_ollama_connectivity, resolve_ollama_base_url
    from src.utils.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether local Ollama is reachable for diagnostic Stage2 use.")
    parser.add_argument("--base-url", default="", help="Local Ollama base URL. Default: OLLAMA_BASE_URL or http://127.0.0.1:11434.")
    parser.add_argument("--model", default="", help="Optional model name to verify locally.")
    parser.add_argument("--timeout-seconds", type=int, default=10, help="HTTP timeout in seconds.")
    parser.add_argument("--json-out", default="analysis/stage2_audits/ollama_connectivity_check_v1.json", help="Output path for the JSON report.")
    parser.add_argument("--md-out", default="analysis/stage2_audits/ollama_connectivity_check_v1.md", help="Output path for the Markdown report.")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_markdown_report(report: dict[str, object]) -> str:
    available_models = report.get("available_models") or []
    if isinstance(available_models, list) and available_models:
        model_lines = "\n".join(f"- `{str(name)}`" for name in available_models)
    else:
        model_lines = "- none reported"
    version_endpoint = report.get("version_endpoint", {})
    tags_endpoint = report.get("tags_endpoint", {})
    return f"""# Ollama Connectivity Check v1

- checked_at_utc: `{report.get("checked_at_utc", "")}`
- base_url: `{report.get("base_url", "")}`
- requested_model: `{report.get("requested_model", "")}`
- reachable: `{report.get("reachable", False)}`
- model_available: `{report.get("model_available", None)}`
- success: `{report.get("success", False)}`

## Endpoint Results

- version_endpoint_ok: `{version_endpoint.get("ok", False)}`
- version_status_code: `{version_endpoint.get("status_code", 0)}`
- version_error: `{version_endpoint.get("error", "")}`
- tags_endpoint_ok: `{tags_endpoint.get("ok", False)}`
- tags_status_code: `{tags_endpoint.get("status_code", 0)}`
- tags_error: `{tags_endpoint.get("error", "")}`

## Available Models

{model_lines}
"""


def main() -> None:
    args = parse_args()
    report = check_ollama_connectivity(
        base_url=resolve_ollama_base_url(args.base_url),
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()

    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_out.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"json_report={json_out}")
    print(f"markdown_report={md_out}")
    print(f"base_url={report['base_url']}")
    print(f"reachable={report['reachable']}")
    print(f"requested_model={report['requested_model']}")
    print(f"model_available={report['model_available']}")
    print(f"success={report['success']}")


if __name__ == "__main__":
    main()
