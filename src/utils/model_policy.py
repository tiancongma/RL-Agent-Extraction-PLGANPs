from __future__ import annotations

from typing import Iterable


DISALLOWED_PREFIXES = ("gemini-2.0-",)
PRIMARY_DEFAULT = "gemini-2.5-flash"
SECONDARY_DEFAULT = "gemini-2.5-flash-lite"


def is_disallowed_model(model_name: str) -> bool:
    s = str(model_name or "").strip().lower()
    return any(s.startswith(p) for p in DISALLOWED_PREFIXES)


def validate_models_or_raise(models: Iterable[str], context: str = "model selection") -> None:
    bad = [str(m).strip() for m in models if is_disallowed_model(str(m))]
    if bad:
        raise ValueError(
            f"{context}: deprecated model(s) not allowed: {', '.join(bad)}. "
            f"Policy: disable all gemini-2.0-*; use {PRIMARY_DEFAULT} and {SECONDARY_DEFAULT}."
        )

