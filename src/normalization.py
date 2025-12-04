import re
from typing import Any


def normalize_payload(obj: Any) -> Any:
    """Recursively normalize payload: lowercase, collapse whitespace.

    - Strings: trim, collapse whitespace to single space, lowercase
    - Maps: lower keys (strings) and normalize values recursively
    - Lists: normalize members recursively
    - Other values: returned unchanged
    """
    if isinstance(obj, str):
        return re.sub(r"\s+", " ", obj).strip().lower()
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            key = k.lower() if isinstance(k, str) else k
            new[key] = normalize_payload(v)
        return new
    if isinstance(obj, list):
        return [normalize_payload(i) for i in obj]
    return obj


__all__ = ["normalize_payload"]
