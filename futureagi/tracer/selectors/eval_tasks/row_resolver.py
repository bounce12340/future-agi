"""Resolve an eval task's desired (in-scope) row set, deterministically.

The "did the row set change?" axis of the reconciler — the counterpart to the
config hash. Streams the in-scope identity ids (span / trace / session ids, per
the task's row_type) in deterministic order, in batches, so a large historical
task never holds its whole row set in memory.

Selection reuses the UI list builders' filter compilation (the same builders
``list_spans_observe`` / ``list_voice_calls`` / ``list_traces_of_session`` /
``list_sessions`` use) so the eval set matches the list endpoints for the same
filters; on top of that filtered id set we apply deterministic hash sampling and
the row limit. The entry FKs are batch-resolved by the materializer later.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from tracer.models.eval_task import RunType
from tracer.services.clickhouse.v2 import get_reader
from tracer.services.clickhouse.v2.id_remap_sql import (
    remap_left_join,
    resolved_id_expr,
)
from tracer.utils.eval_task_filters import id_filter

if TYPE_CHECKING:
    from tracer.models.eval_task import EvalTask

# row_type → (UI list builder query type, identity column the builder emits).
_BUILDER_BY_ROW_TYPE = {
    "spans": ("SPAN_LIST", "id"),
    "voiceCalls": ("VOICE_CALL_LIST", "id"),
    "traces": ("TRACE_LIST", "trace_id"),
    "sessions": ("SESSION_LIST", "session_id"),
}

# Predicate for a filter that is set but can match no row.
_MATCH_NOTHING = "AND 1 = 0"

# Join alias for the session id-remap inside the scope subquery. Distinct from
# the builders' own ``ts_remap`` so the two never collide in one statement.
_SCOPE_REMAP_ALIAS = "scope_ts_remap"


def iter_desired_rows(
    task: EvalTask, *, batch_size: int = 10_000, ceiling: datetime | None = None
) -> Iterator[list[str]]:
    # Row limit applies to historical tasks only; continuous runs forever.
    limit = task.spans_limit if task.run_type == RunType.HISTORICAL else None
    sampling_rate = task.sampling_rate if task.sampling_rate is not None else 100.0
    created_at_ceiling = ceiling if task.run_type == RunType.CONTINUOUS else None

    sql, params = _build_sample_query(
        project_id=str(task.project_id),
        row_type=task.row_type,
        salt=str(task.id),
        sampling_rate=float(sampling_rate),
        filters=task.filters or {},
        limit=limit,
        created_at_floor=_continuous_floor(task),
        created_at_ceiling=created_at_ceiling,
    )
    reader = get_reader()
    try:
        yield from reader.stream_query(sql, params, batch_size=batch_size)
    finally:
        reader.close()


def _continuous_floor(task: EvalTask) -> datetime | None:
    """Lower ``created_at`` bound for a continuous task's desired set.

    A continuous task only evaluates rows that arrive after it starts — it must
    never backfill the project history that pre-dates it. The floor is the
    forward watermark once the reconciler has advanced it, falling back to the
    task's start (then creation) on the first pass. Historical tasks have no
    floor here (they carve their window from ``filters`` + ``spans_limit``).
    """
    if task.run_type != RunType.CONTINUOUS:
        return None
    return task.continuous_cursor or task.start_time or task.created_at


def _build_sample_query(
    *,
    project_id: str,
    row_type: str,
    salt: str,
    sampling_rate: float,
    filters: dict | None,
    limit: int | None,
    created_at_floor: datetime | None = None,
    created_at_ceiling: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    """Sampled-row-ids SQL for the row_type: take the UI list builder's filtered
    id set and wrap it with deterministic hash sampling, a stable order, and the
    row limit."""
    from tracer.services.clickhouse.v2.dispatch import get_v2_class

    try:
        query_type, id_col = _BUILDER_BY_ROW_TYPE[row_type]
    except KeyError:
        raise ValueError(f"Unsupported row_type: {row_type!r}") from None

    # Reshape the eval task's stored filters into the frontend filter list the UI
    # builder consumes; the date range is read via parse_time_range.
    f = filters or {}
    ui_filters = list(f.get("filters") or [])
    dr = f.get("date_range")
    if isinstance(dr, list | tuple) and len(dr) == 2:
        ui_filters.append(
            {
                "column_id": "created_at",
                "filter_config": {
                    "filter_type": "datetime",
                    "filter_op": "between",
                    "filter_value": [dr[0], dr[1]],
                },
            }
        )

    # Continuous floor: passed to build_id_query to bind on arrival (created_at),
    # not injected as a filter (parse_time_range would fold it onto start_time).
    # None (historical) keeps the builder's start_time window.
    builder = get_v2_class(query_type)(project_id=str(project_id), filters=ui_filters)
    inner_sql, params = builder.build_id_query(
        created_at_floor=created_at_floor, created_at_ceiling=created_at_ceiling
    )
    params = {
        **params,
        "salt": str(salt),
        "rate": float(sampling_rate),
        "scope_project_id": str(project_id),
    }

    # observation_type / trace_id / span_id are legacy top-level keys, not
    # filter-builder columns; constrain the id set against spans directly.
    # ``src`` is the expression on a span row that carries this row_type's
    # identity. A session's is read through the id-remap, because SESSION_LIST
    # projects the resolved (survivor) id — matching a raw ``trace_session_id``
    # against it would drop every cross-cutover straddler. No other row_type's
    # identity is re-keyed, so they need no join.
    if row_type == "sessions":
        scope_join = remap_left_join(
            "spans.trace_session_id", "trace_session_id_remap", _SCOPE_REMAP_ALIAS
        )
        src = resolved_id_expr("spans.trace_session_id", _SCOPE_REMAP_ALIAS)
    else:
        scope_join = ""
        src = id_col
    scope_preds: list[str] = []

    ot = f.get("observation_type")
    if ot:
        params["otypes"] = tuple(
            str(o) for o in (ot if isinstance(ot, list | tuple | set) else [ot])
        )
        # For traces, the trace list derives observation_type from the ROOT span
        # (it scans parent_span_id IS NULL), so match root spans only for parity.
        root_pred = (
            " AND (parent_span_id IS NULL OR parent_span_id = '')"
            if row_type == "traces"
            else ""
        )
        scope_preds.append(
            _span_scope_pred(
                id_col, src, scope_join, f"observation_type IN %(otypes)s{root_pred}"
            )
        )

    for key, column, param in (
        ("trace_id", "trace_id", "f_trace_ids"),
        ("span_id", "id", "f_span_ids"),
    ):
        ids = id_filter(f, key)
        if ids is None:
            continue
        if not ids:
            # An explicitly empty id list scopes the task to nothing; dropping
            # it would widen the run back to the whole project. Mirrors
            # ``Q(trace_id__in=[])`` and ``count_with_filters(trace_ids=[])``.
            scope_preds.append(_MATCH_NOTHING)
            continue
        params[param] = tuple(ids)
        scope_preds.append(
            _span_scope_pred(id_col, src, scope_join, f"{column} IN %({param})s")
        )

    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT %(lim)s"
        params["lim"] = int(limit)

    # modulo() not `%` — clickhouse-connect treats a literal `%` as a
    # parameter-format marker. Order by the id for a stable limit prefix.
    sql = (
        f"SELECT {id_col} FROM ({inner_sql}) "
        f"WHERE modulo(cityHash64(%(salt)s, toString({id_col})), 100) < %(rate)s "
        f"{' '.join(scope_preds)} "
        f"ORDER BY {id_col} {limit_sql}"
    )
    return sql, params


def _span_scope_pred(id_col: str, src: str, join: str, predicate: str) -> str:
    """``AND <id_col> IN (SELECT <src> FROM spans <join> WHERE <predicate> …)``.

    Scoped like the outer scan (project + not-deleted) so a top-level key can
    never match ids from another project or from soft-deleted rows. ``join`` is
    the id-remap join the session identity needs and is empty otherwise; the
    remap map carries only ``any_id``/``survivor_id``, so the span columns stay
    unqualified and unambiguous under it.
    """
    return (
        f"AND {id_col} IN "
        f"(SELECT {src} FROM spans {join} "
        f"WHERE {predicate} "
        f"AND project_id = %(scope_project_id)s AND is_deleted = 0)"
    )
