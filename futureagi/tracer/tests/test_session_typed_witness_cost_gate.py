"""The witness-cost gate decides a typed-leaf Session page's lane, and only that.

A page filtered by a typed span-attribute leaf has two exact lanes: the
candidate cursor statement, seeded by a whole-window any-span witness that
ClickHouse materialises while planning, and the bounded walk. On the
high-volume tenant the witness scan is 238 M rows and the statement dies at
the wall; on the sparse tenant the same statement is the one complete 2 s
answer the walk turned into thirty-two seed statements. The gate prices the
witness scan from the index (``EXPLAIN ESTIMATE`` of exactly that scan, at
the statement's own read settings) and pins the lane before the first
statement; a cursor carries the lane to every later hop.

These tests pin the guards the gate must be able to fail:

(a) an estimate over the target routes to the walk;
(b) an estimate under the target routes to the seeded statement;
(c) a probe that raises, cannot be read, names no part, returns after its
    budget, has no budget to run in, or is switched off by the setting routes
    to the walk - never the seeded statement;
(d) the probe runs at the costed statement's settings, threads included, with
    projections pinned off, and its scan is the statement's witness scan;
(e) a cursor that carries a lane keeps it: no second probe, same lane; the
    codec rejects any other lane; the view's continuation carries the pin.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from django.core import signing
from django.test import override_settings

from tracer.selectors import session_witness_cost_gate as gate
from tracer.selectors.session_witness_cost_gate import decide_typed_witness_lane
from tracer.services.clickhouse.list_cursor import (
    CURSOR_SALT,
    TYPED_WITNESS_LANE_SEEDED,
    TYPED_WITNESS_LANE_WALK,
    ListCursorError,
    decode_list_cursor,
    encode_list_cursor,
)
from tracer.services.clickhouse.read_budget import ReadDeadline
from tracer.services.clickhouse.v2.query_builders.session_list import (
    SessionListQueryBuilderV2,
)

pytestmark = pytest.mark.unit

PROJECT_ID = "00000000-0000-4000-8000-000000000001"
USER_ID = "00000000-0000-4000-8000-000000000003"
END = datetime(2026, 9, 12, 0, 0)
START = END - timedelta(days=365)
TARGET = 2_000_000
BUDGET_MS = 1_500

_WITNESS_SCAN = re.compile(
    r"\(SELECT groupUniqArray\(assumeNotNull\(trace_session_id\)\)\s*(.*?)\s*\)"
    r" AS candidate_witness_session_ids",
    re.S,
)
_PROBE_SCAN = re.compile(r"EXPLAIN ESTIMATE\s*SELECT count\(\)\s*(.*)$", re.S)


def _window() -> dict:
    return {
        "column_id": "created_at",
        "filter_config": {
            "filter_type": "datetime",
            "filter_op": "between",
            "filter_value": [START.isoformat(), END.isoformat()],
        },
    }


def _attribute(key: str, kind: str, operation: str, value) -> dict:
    return {
        "column_id": key,
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": kind,
            "filter_op": operation,
            "filter_value": value,
        },
    }


def _user() -> dict:
    return {
        "column_id": "end_user_id",
        "filter_config": {
            "col_type": "SYSTEM_METRIC",
            "filter_type": "text",
            "filter_op": "in",
            "filter_value": [USER_ID],
        },
    }


def _builder(filters: list[dict] | None = None) -> SessionListQueryBuilderV2:
    return SessionListQueryBuilderV2(
        project_id=PROJECT_ID,
        filters=filters or [_window(), _attribute("flag", "boolean", "equals", True)],
        page_number=0,
        page_size=25,
        bounded_internal_scan=True,
    )


def _read_settings(max_result_rows: int) -> dict:
    return {
        "max_threads": 2,
        "max_block_size": 8192,
        "max_memory_usage": 4 * 1024**3,
        "max_result_rows": int(max_result_rows),
    }


class _Analytics:
    """A ClickHouse transport that answers every probe with one estimate table."""

    def __init__(
        self, rows: int | None = 100, *, table: str = "spans", raise_with=None
    ):
        self.calls: list[dict] = []
        self._rows = rows
        self._table = table
        self._raise_with = raise_with

    def execute_ch_query(self, query, params, *, timeout_ms, settings):
        self.calls.append(
            {
                "query": query,
                "params": params,
                "timeout_ms": timeout_ms,
                "settings": settings,
            }
        )
        if self._raise_with is not None:
            raise self._raise_with
        columns = ["database", "table", "parts", "rows", "marks"]
        if self._rows is None:
            return SimpleNamespace(data=[], columns=columns)
        return SimpleNamespace(
            data=[
                {
                    "database": "d",
                    "table": self._table,
                    "parts": 3,
                    "rows": int(self._rows),
                    "marks": 12,
                }
            ],
            columns=columns,
        )


def _decide(builder, analytics, *, pinned_lane=None, deadline=None):
    return decide_typed_witness_lane(
        builder=builder,
        analytics=analytics,
        deadline=deadline or ReadDeadline.start(30_000),
        read_settings=_read_settings,
        pinned_lane=pinned_lane,
    )


_GATE = override_settings(
    SESSION_LIST_TYPED_WITNESS_SEEDED_MAX_ESTIMATED_ROWS=TARGET,
    SESSION_LIST_TYPED_WITNESS_PROBE_BUDGET_MS=BUDGET_MS,
)


# -- (a) over target -> walk ---------------------------------------------


@_GATE
def test_an_estimate_over_the_target_takes_the_walk():
    builder = _builder()
    analytics = _Analytics(rows=TARGET + 1)
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "over_target"
    assert decision.estimated_rows == TARGET + 1
    assert decision.probe_statements == 1
    assert builder.typed_witness_lane() == TYPED_WITNESS_LANE_WALK
    assert builder.prefers_bounded_filter_page() is True


# -- (b) under target -> seeded -------------------------------------------


@_GATE
@pytest.mark.parametrize("rows", [0, 1, TARGET - 1, TARGET])
def test_an_estimate_at_or_under_the_target_takes_the_seeded_statement(rows):
    builder = _builder()
    analytics = _Analytics(rows=rows)
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_SEEDED
    assert decision.reason == "under_target"
    assert decision.estimated_rows == rows
    assert builder.typed_witness_lane() == TYPED_WITNESS_LANE_SEEDED
    # The candidate lane: the page is no longer preferred bounded, and the
    # cursor statement the view issues on that lane is supported for it.
    assert builder.prefers_bounded_filter_page() is False
    assert builder.supports_candidate_cursor_page() is True
    assert builder.supports_candidate_first_page() is True


@_GATE
def test_the_seeded_lane_is_pinned_only_by_the_gate():
    """With no probe run, a typed leaf walks: navigation, exports, any caller."""

    builder = _builder()
    assert builder.typed_witness_lane() is None
    assert builder.prefers_bounded_filter_page() is True
    with pytest.raises(ValueError):
        builder.pin_typed_witness_lane("bogus")


# -- (c) a probe that cannot answer -> walk, never seeded ------------------


@_GATE
def test_a_probe_that_raises_takes_the_walk():
    builder = _builder()
    analytics = _Analytics(rows=1, raise_with=RuntimeError("socket closed"))
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "probe_failed"
    assert decision.probe_statements == 1
    assert builder.prefers_bounded_filter_page() is True


@_GATE
def test_an_unreadable_estimate_takes_the_walk():
    builder = _builder()
    analytics = _Analytics(rows=1, table="not_spans")
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "unreadable"
    assert decision.estimated_rows is None
    assert builder.prefers_bounded_filter_page() is True


@_GATE
def test_an_estimate_naming_no_part_takes_the_walk():
    builder = _builder()
    analytics = _Analytics(rows=None)
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "empty_estimate"
    assert builder.prefers_bounded_filter_page() is True


@_GATE
def test_a_probe_that_returns_after_its_budget_takes_the_walk(monkeypatch):
    clock = iter([100.0, 100.0 + (BUDGET_MS + 1) / 1000.0])
    monkeypatch.setattr(gate, "monotonic", lambda: next(clock))
    builder = _builder()
    analytics = _Analytics(rows=1)
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "probe_over_budget"
    # The estimate was small and is reported; it did not license the statement.
    assert decision.estimated_rows == 1
    assert decision.probe_ms > BUDGET_MS
    assert builder.prefers_bounded_filter_page() is True


@_GATE
def test_a_request_with_no_budget_left_runs_no_probe_and_walks():
    builder = _builder()
    analytics = _Analytics(rows=1)
    decision = _decide(builder, analytics, deadline=ReadDeadline.start(BUDGET_MS - 1))
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "no_budget"
    assert decision.probe_statements == 0
    assert analytics.calls == []
    assert builder.prefers_bounded_filter_page() is True


@_GATE
def test_an_exhausted_deadline_runs_no_probe_and_walks():
    builder = _builder()
    analytics = _Analytics(rows=1)
    exhausted = ReadDeadline(total_ms=1, started=0.0)
    decision = _decide(builder, analytics, deadline=exhausted)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "no_budget"
    assert analytics.calls == []


@override_settings(
    SESSION_LIST_TYPED_WITNESS_SEEDED_MAX_ESTIMATED_ROWS=0,
    SESSION_LIST_TYPED_WITNESS_PROBE_BUDGET_MS=BUDGET_MS,
)
def test_a_zero_target_switches_the_gate_off_and_every_page_walks():
    builder = _builder()
    analytics = _Analytics(rows=1)
    decision = _decide(builder, analytics)
    assert decision.lane == TYPED_WITNESS_LANE_WALK
    assert decision.reason == "gate_off"
    assert analytics.calls == []
    assert builder.prefers_bounded_filter_page() is True


@_GATE
def test_a_shape_the_policy_does_not_govern_runs_no_probe_and_pins_nothing():
    builder = _builder(
        [_window(), _user(), _attribute("n", "number", "greater_than", 7)]
    )
    analytics = _Analytics(rows=1)
    decision = _decide(builder, analytics)
    assert decision.lane is None
    assert decision.reason == "not_governed"
    assert analytics.calls == []
    assert builder.typed_witness_lane() is None
    assert builder.build_typed_witness_cost_probe_query() is None
    # The user-detail shape keeps its own candidate lane, as before.
    assert builder.prefers_bounded_filter_page() is False


# -- (d) the probe runs at the costed statement's settings ------------------


@_GATE
def test_the_probe_runs_at_the_costed_statements_settings_with_projections_off():
    builder = _builder()
    analytics = _Analytics(rows=1)
    _decide(builder, analytics)
    (call,) = analytics.calls
    expected = {**_read_settings(256), "optimize_use_projections": 0}
    assert call["settings"] == expected
    assert call["settings"]["max_threads"] == _read_settings(1)["max_threads"]
    assert 0 < call["timeout_ms"] <= BUDGET_MS


@_GATE
def test_the_probe_uses_the_views_session_read_settings_threads():
    """Through the view's own callable the probe carries the session-list workers."""

    from django.conf import settings

    import tracer.views.trace_session as view

    builder = _builder()
    analytics = _Analytics(rows=1)
    decide_typed_witness_lane(
        builder=builder,
        analytics=analytics,
        deadline=ReadDeadline.start(30_000),
        read_settings=lambda rows: view._session_read_settings(max_result_rows=rows),
    )
    (call,) = analytics.calls
    assert call["settings"]["max_threads"] == settings.SESSION_LIST_READ_MAX_THREADS
    assert call["settings"]["optimize_use_projections"] == 0
    assert call["settings"]["max_result_rows"] == 256


@pytest.mark.parametrize(
    ("kind", "operation", "value"),
    [
        ("boolean", "equals", True),
        ("number", "greater_than", 7),
        ("text", "equals", "Rejected"),
    ],
)
def test_the_probe_prices_exactly_the_statements_witness_scan(kind, operation, value):
    builder = _builder([_window(), _attribute("k", kind, operation, value)])
    statement, statement_params = builder.build_candidate_cursor_page_query()
    probe_sql, probe_params = builder.build_typed_witness_cost_probe_query()
    witness = _WITNESS_SCAN.search(statement)
    probe = _PROBE_SCAN.search(probe_sql.strip())
    assert witness is not None and probe is not None
    assert " ".join(witness.group(1).split()) == " ".join(probe.group(1).split())
    # v2 storage names, no legacy token, no trailing SETTINGS on an EXPLAIN.
    assert "attrs_" in probe_sql and "span_attr_" not in probe_sql
    assert "SETTINGS" not in probe_sql
    # Every binding the probe names is bound, to the statement's value.
    for name in re.findall(r"%\((\w+)\)s", probe_sql):
        assert name in probe_params
        assert probe_params[name] == statement_params[name]


def test_the_estimate_reader_sums_part_rows_for_the_spans_table_only():
    builder = _builder()
    columns = ["database", "table", "parts", "rows", "marks"]
    rows = [
        {"database": "d", "table": "spans", "parts": 1, "rows": 40, "marks": 1},
        {"database": "d", "table": "spans", "parts": 2, "rows": 2, "marks": 1},
    ]
    assert builder.typed_witness_cost_estimate(rows, columns) == 42
    assert builder.typed_witness_cost_estimate(rows, ["rows"]) is None
    assert (
        builder.typed_witness_cost_estimate([], columns) is gate.EMPTY_DENSITY_ESTIMATE
    )


# -- (e) cursor hops keep the lane -----------------------------------------


@_GATE
@pytest.mark.parametrize("lane", [TYPED_WITNESS_LANE_SEEDED, TYPED_WITNESS_LANE_WALK])
def test_a_cursor_that_carries_a_lane_keeps_it_without_a_second_probe(lane):
    builder = _builder()
    # A probe that would decide the OTHER lane, to prove it is never consulted.
    analytics = _Analytics(rows=TARGET + 1 if lane == TYPED_WITNESS_LANE_SEEDED else 1)
    decision = _decide(builder, analytics, pinned_lane=lane)
    assert decision.lane == lane
    assert decision.reason == "cursor_pinned"
    assert decision.probe_statements == 0
    assert analytics.calls == []
    assert builder.typed_witness_lane() == lane
    assert builder.prefers_bounded_filter_page() is (lane == TYPED_WITNESS_LANE_WALK)


@_GATE
def test_an_unknown_pinned_lane_is_refused():
    with pytest.raises(ValueError):
        _decide(_builder(), _Analytics(rows=1), pinned_lane="bogus")


def _scope() -> dict:
    return {"principal_id": "p", "project_ids": [PROJECT_ID]}


def _query() -> dict:
    return {"filters": [_window()], "page_size": 25}


def _encode(**extra) -> str:
    return encode_list_cursor(
        resource="observe_sessions",
        scope=_scope(),
        query=_query(),
        page_size=25,
        window_start=START,
        window_end=END,
        order=(END - timedelta(days=3), "s"),
        seen_rows=25,
        **extra,
    )


def _decode(token: str):
    return decode_list_cursor(
        token, resource="observe_sessions", scope=_scope(), query=_query(), page_size=25
    )


@pytest.mark.parametrize("lane", [TYPED_WITNESS_LANE_SEEDED, TYPED_WITNESS_LANE_WALK])
def test_the_cursor_carries_the_lane_and_a_legacy_token_carries_none(lane):
    assert _decode(_encode(typed_witness_lane=lane)).typed_witness_lane == lane
    assert _decode(_encode()).typed_witness_lane is None
    assert _decode(_encode(typed_witness_lane=None)).typed_witness_lane is None


def test_the_codec_refuses_any_other_lane():
    with pytest.raises(ValueError):
        _encode(typed_witness_lane="bogus")
    payload = signing.loads(
        _encode(),
        key=__import__("django.conf").conf.settings.SECRET_KEY,
        salt=CURSOR_SALT,
    )
    payload["typed_witness_lane"] = "bogus"
    forged = signing.dumps(
        payload,
        key=__import__("django.conf").conf.settings.SECRET_KEY,
        salt=CURSOR_SALT,
        compress=True,
    )
    with pytest.raises(ListCursorError):
        _decode(forged)


def test_a_token_without_the_field_is_byte_identical_to_the_pre_field_payload():
    """Absent, not null: a page with no lane mints the payload it always did."""

    with_none = signing.loads(
        _encode(typed_witness_lane=None),
        key=__import__("django.conf").conf.settings.SECRET_KEY,
        salt=CURSOR_SALT,
    )
    assert "typed_witness_lane" not in with_none


@_GATE
def test_the_views_continuation_carries_the_lane_the_page_took():
    """The view mints the pinned lane into the next hop's cursor."""

    from tracer.views.trace_session import SessionPageSelection

    builder = _builder()
    _decide(builder, _Analytics(rows=1))
    assert builder.typed_witness_lane() == TYPED_WITNESS_LANE_SEEDED
    last = {
        "session_start": (END - timedelta(days=3)).replace(tzinfo=UTC),
        "session_id": "s",
    }
    selection = SessionPageSelection(
        builder=builder,
        filters=builder.filters,
        attested_filters=builder.filters,
        page_candidates=[last],
        candidate_total_count=None,
        bounded_page=None,
        candidate_cursor=True,
        candidate_cursor_has_more=True,
        candidate_total_is_lower_bound=False,
        cursor_enabled=True,
        cursor_state=None,
        cursor_scope=_scope(),
        cursor_query=_query(),
        end_user_display=None,
    )
    seen, token, has_more = selection.cursor(None)
    assert has_more is True and seen == 1
    assert _decode(token).typed_witness_lane == TYPED_WITNESS_LANE_SEEDED


# -- the classifier worker budget (an owner capacity call, default unchanged) --


def _walk_kwargs():
    from unittest import mock

    import tracer.views.trace_session as view

    with mock.patch.object(view, "read_bounded_filter_page") as walk:
        walk.return_value = SimpleNamespace(rows=[], complete=True)
        view._read_session_filter_page(
            _builder(), SimpleNamespace(), ReadDeadline.start(30_000)
        )
    (call,) = walk.call_args_list
    return call.kwargs


@override_settings(SESSION_LIST_CLASSIFY_MAX_THREADS=0)
def test_by_default_the_walk_states_no_classifier_worker_budget():
    from django.conf import settings

    kwargs = _walk_kwargs()
    assert kwargs["classify_read_settings"] is None
    assert (
        kwargs["read_settings"]["max_threads"] == settings.SESSION_LIST_READ_MAX_THREADS
    )


@override_settings(SESSION_LIST_CLASSIFY_MAX_THREADS=4)
def test_a_flipped_setting_states_the_classifier_worker_budget_alone():
    from django.conf import settings

    kwargs = _walk_kwargs()
    assert kwargs["classify_read_settings"] == {"max_threads": 4}
    # Every other statement kind keeps the page's count.
    assert (
        kwargs["read_settings"]["max_threads"] == settings.SESSION_LIST_READ_MAX_THREADS
    )
