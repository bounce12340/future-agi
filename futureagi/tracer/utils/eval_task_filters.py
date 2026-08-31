"""Shared reading of the id-list keys stored on ``EvalTask.filters``.

``EvalTask.filters`` is consumed by three independent translators that must
agree on what a stored id value means:

* ``selectors/eval_tasks/row_resolver.py`` — the live engine's row set,
* ``utils/eval_tasks.py::parsing_evaltask_filters`` — the retired PG
  dispatcher's ``Q``,
* ``services/clickhouse/v2/span_reader.py::parsing_evaltask_filters_for_ch``
  — that dispatcher's ClickHouse companion.

They disagreed, and the disagreement was the bug: two of them dropped
``trace_id``/``span_id`` entirely while the serializer kept accepting and
storing both. Keeping the normalisation here means a key added to one
translator reads the same way in all three.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _as_id_list(value: Any) -> list[str]:
    """Normalise a stored id filter value to a list of non-empty string ids.

    ``EvalTaskFiltersField`` coerces a scalar to a one-item list on write, but
    rows stored before that field existed — and any writer that bypasses the
    serializer — can still hold a bare string.
    """
    if value is None or value == "":
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def id_filter(filters: Mapping[str, Any], key: str) -> list[str] | None:
    """The ids stored under ``key``, or ``None`` when it carries no constraint.

    An absent key, ``None`` and ``""`` all mean "not set" — the same reading
    ``EvalTaskFiltersField`` applies on write, where a ``None`` id key is
    skipped rather than validated. An empty *list* is a real constraint that
    matches nothing, so callers must not conflate it with ``None``.
    """
    value = filters.get(key)
    if value is None or value == "":
        return None
    return _as_id_list(value)
