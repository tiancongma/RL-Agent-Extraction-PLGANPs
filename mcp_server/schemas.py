from __future__ import annotations

from typing import Any


JSONDict = dict[str, Any]
JSONList = list[Any]


def success_response(**payload: Any) -> JSONDict:
    return {"ok": True, **payload}


def error_response(message: str, **payload: Any) -> JSONDict:
    return {"ok": False, "error": message, **payload}
