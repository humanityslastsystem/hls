"""Vendored verbatim from mini-SWE-agent ``src/minisweagent/utils/serialize.py``.

Upstream: https://github.com/SWE-agent/mini-swe-agent at
``04d809ceab9df28f9adaed044884180159172930``. MIT licensed, Copyright (c) 2025
Kilian A. Lieret and Carlos E. Jimenez; see ``LICENSE`` beside this file.

No divergences.
"""

from typing import Any

UNSET = object()


def recursive_merge(*dictionaries: dict | None) -> dict:
    """Merge multiple dictionaries recursively.

    Later dictionaries take precedence over earlier ones.
    Nested dictionaries are merged recursively.
    UNSET values are skipped.
    """
    if not dictionaries:
        return {}
    result: dict[str, Any] = {}
    for d in dictionaries:
        if d is None:
            continue
        for key, value in d.items():
            if value is UNSET:
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            elif isinstance(value, dict):
                # Recursively merge dict values to filter out nested UNSET values
                result[key] = recursive_merge(value)
            else:
                result[key] = value
    return result
