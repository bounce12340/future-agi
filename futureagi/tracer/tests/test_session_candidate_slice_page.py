"""Session candidate pages read from a bounded slice, and stay exact.

The statement under test replays latest state over whatever window it is
given, so on a high-volume tenant a month-long window does not return inside
the request wall. These tests pin the two halves of the repair: the scan floor
moves without the SQL text moving, and a narrowed scan never publishes a row
whose start it inflated.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock
from uuid import UUID

import pytest

from tracer.selectors.session_candidate_slice import (
    SessionCandidateSlicePage,
    read_candidate_slice_page,
)
from tracer.services.clickhouse.read_budget import ReadDeadline
from tracer.services.clickhouse.v2.query_builders.session_list import (
    SessionListQueryBuilderV2,
)

PROJECT = str(UUID(int=1))
END = datetime(2026, 9, 12)
WINDOW_HOURS = 720
START = END - timedelta(hours=WINDOW_HOURS)
# The production shape this repair exists for: the newest stretch of the window
# is nearly empty and everything older is dense, so a width chosen by duration
# alone lands either short of the data or on top of all of it.
SPARSE_HOURS = 216
DENSE_ROWS_PER_HOUR = 100_000
BUDGET = 1_000_000


def _filters(start=START, end=END):
    return [
        {
            "column_id": "created_at",
            "filter_config": {
                "col_type": "SYSTEM_METRIC",
                "filter_type": "datetime",
                "filter_op": "between",
                "filter_value": [start.isoformat(), end.isoformat()],
            },
        }
    ]


def _builder(page_size=2, **kwargs):
    return SessionListQueryBuilderV2(
        project_id=PROJECT,
        filters=_filters(**kwargs),
        page_number=0,
        page_size=page_size,
        bounded_internal_scan=True,
    )


def _sha(sql: str) -> str:
    return hashlib.sha256(sql.strip().rstrip(";").encode()).hexdigest()[:16]


def _sid(index: int) -> str:
    return str(UUID(int=100 + index))


def _moment(micros: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(microseconds=micros)


class _Server:
    """A fake CH that answers the three statements this lane issues.

    ``rows`` are the candidates, each with the start its NEWEST-side root gives
    it; a slice with floor ``T`` discovers exactly those whose start is at or
    above ``T``, which is the real statement's semantics. ``full_state`` maps a
    session id to its true start over the whole window, so a session present in
    both with different values is a displaced one.

    Density follows the production shape this repair exists for: nothing in the
    newest ``SPARSE_HOURS``, and a flat dense rate before that.
    """

    def __init__(self, *, rows, full_state, estimate_columns=("table", "rows")):
        self.rows = rows
        self.full_state = full_state
        self.estimate_columns = estimate_columns
        self.calls: list[tuple[str, dict]] = []

    @property
    def kinds(self) -> list[str]:
        return [kind for kind, _params in self.calls]

    def execute_ch_query(self, query, params, *, timeout_ms, settings):
        if query.lstrip().startswith("EXPLAIN ESTIMATE"):
            self.calls.append(("probe", params))
            return self._estimate(params)
        if "AS remaining_count" in query:
            self.calls.append(("slice", params))
            return self._slice(params)
        self.calls.append(("full_state", params))
        return self._match(params)

    def _estimate(self, params):
        floor = _moment(params["candidate_density_start_us"])
        dense_hours = max(
            0.0, ((END - timedelta(hours=SPARSE_HOURS)) - floor) / timedelta(hours=1)
        )
        rows = int(dense_hours * DENSE_ROWS_PER_HOUR)
        if self.estimate_columns is None:
            return SimpleNamespace(data=[], columns=None)
        return SimpleNamespace(
            data=[{"table": "spans", "rows": rows}] if rows else [],
            columns=list(self.estimate_columns),
        )

    def _slice(self, params):
        floor = _moment(params["start_date_us"])
        found = sorted(
            (row for row in self.rows if row["session_start"] >= floor),
            key=lambda row: row["session_start"],
            reverse=True,
        )
        return SimpleNamespace(
            data=[{**row, "remaining_count": len(found)} for row in found],
            columns=["session_id", "session_start", "remaining_count"],
        )

    def _match(self, params):
        wanted = set(params["candidate_filter_session_id_array"])
        return SimpleNamespace(
            data=[
                {"session_id": sid, "start_time": start}
                for sid, start in self.full_state.items()
                if sid in wanted
            ],
            columns=["session_id", "start_time"],
        )


def _read(builder, server, **kwargs) -> SessionCandidateSlicePage:
    return read_candidate_slice_page(
        builder=builder,
        analytics=server,
        deadline=ReadDeadline.start(30_000),
        read_settings=lambda rows: {"max_result_rows": rows},
        query_timeout_ms=9_000,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The statement itself
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_raised_scan_floor_moves_only_the_binding_not_the_statement():
    builder = _builder()
    unsliced, unsliced_params = builder.build_candidate_cursor_page_query()
    floor = END - timedelta(hours=SPARSE_HOURS)
    sliced, sliced_params = builder.build_candidate_cursor_page_query(
        scan_start_time=floor
    )

    assert sliced == unsliced, "a narrowed scan must be the same statement"
    assert _sha(sliced) == _sha(unsliced)
    assert sliced_params["start_date_us"] > unsliced_params["start_date_us"]
    assert sliced_params["end_date_us"] == unsliced_params["end_date_us"]
    # The verifier reads the request window out of the builder's own params.
    # A floor bound there instead would make it verify the slice against
    # itself, which is not a verification at all.
    assert builder.params["start_date_us"] == unsliced_params["start_date_us"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "floor", [START - timedelta(hours=1), END, END + timedelta(hours=1)]
)
def test_scan_floor_outside_the_request_window_is_refused(floor):
    with pytest.raises(ValueError, match="scan floor"):
        _builder().build_candidate_cursor_page_query(scan_start_time=floor)


@pytest.mark.unit
def test_sliced_continuation_narrows_the_scan_to_the_cursor_instant():
    builder = _builder()
    cursor_at = END - timedelta(hours=SPARSE_HOURS + 1)
    floor = END - timedelta(hours=SPARSE_HOURS * 2)
    _sql, params = builder.build_candidate_cursor_page_query(
        before_start_time=cursor_at,
        before_session_id=_sid(0),
        scan_start_time=floor,
    )
    # Half-open, so the cursor instant itself is still read and the keyset's
    # id tie-break - not the scan - separates a session tied on that start.
    assert params["end_date_us"] == params["cursor_before_start_us"] + 1
    # Without a raised floor the whole-window statement keeps its bindings, so
    # its text and its cost are untouched by this lane.
    _unsliced_sql, unsliced = builder.build_candidate_cursor_page_query(
        before_start_time=cursor_at, before_session_id=_sid(0)
    )
    assert unsliced["end_date_us"] > unsliced["cursor_before_start_us"]


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_displaced_candidate_is_dropped_and_the_rest_of_the_page_publishes():
    """The production case: the slice's NEWEST row truly started far earlier."""
    newest = END - timedelta(hours=100)
    second = END - timedelta(hours=120)
    third = END - timedelta(hours=140)
    server = _Server(
        rows=[
            {"session_id": _sid(0), "session_start": newest},
            {"session_id": _sid(1), "session_start": second},
            {"session_id": _sid(2), "session_start": third},
        ],
        full_state={
            # Inflated: its first root is three weeks older than the slice saw.
            _sid(0): START + timedelta(hours=4),
            _sid(1): second,
            _sid(2): third,
        },
    )
    page = _read(_builder(page_size=2), server)

    assert [row["session_id"] for row in page.rows] == [_sid(1), _sid(2)]
    assert page.has_more is True
    assert page.slice_start is not None
    assert server.kinds.count("full_state") == 1
    assert "slice" in server.kinds


@pytest.mark.unit
def test_page_short_of_survivors_widens_and_ends_on_the_unsliced_statement():
    kept = END - timedelta(hours=120)
    server = _Server(
        rows=[
            {"session_id": _sid(0), "session_start": END - timedelta(hours=100)},
            {"session_id": _sid(1), "session_start": kept},
            {"session_id": _sid(2), "session_start": END - timedelta(hours=140)},
        ],
        full_state={
            _sid(0): START + timedelta(hours=4),
            _sid(1): kept,
            _sid(2): START + timedelta(hours=6),
        },
    )
    page = _read(_builder(page_size=2), server)

    # One survivor cannot fill a two-row page, and what is below the floor is
    # unseen, so the read may not publish a short page. It widens instead, and
    # the last statement is the unsliced one - today's behaviour, still exact.
    assert server.kinds.count("slice") > 1
    assert page.slice_start is None
    assert [row["session_id"] for row in page.rows] == [_sid(0), _sid(1)]


@pytest.mark.unit
def test_a_slice_too_small_to_settle_a_page_widens_without_a_verifier():
    rows = [{"session_id": _sid(0), "session_start": END - timedelta(hours=100)}]
    server = _Server(rows=rows, full_state={_sid(0): rows[0]["session_start"]})
    page = _read(_builder(page_size=2), server)

    # One candidate cannot settle a two-row page whatever it resolves to, so
    # the full-window verifier - the expensive statement here - is never run
    # against a slice that could not publish anyway.
    assert "full_state" not in server.kinds
    assert page.slice_start is None
    assert [row["session_id"] for row in page.rows] == [_sid(0)]
    assert page.has_more is False


@pytest.mark.unit
def test_every_candidate_surviving_publishes_without_a_second_statement():
    rows = [
        {"session_id": _sid(0), "session_start": END - timedelta(hours=100)},
        {"session_id": _sid(1), "session_start": END - timedelta(hours=120)},
        {"session_id": _sid(2), "session_start": END - timedelta(hours=140)},
    ]
    server = _Server(
        rows=rows,
        full_state={row["session_id"]: row["session_start"] for row in rows},
    )
    page = _read(_builder(page_size=2), server)

    assert [row["session_id"] for row in page.rows] == [_sid(0), _sid(1)]
    assert page.has_more is True
    assert page.slice_start is not None
    assert server.kinds.count("slice") == 1


@pytest.mark.unit
def test_a_candidate_full_state_drops_entirely_is_not_published():
    rows = [
        {"session_id": _sid(0), "session_start": END - timedelta(hours=100)},
        {"session_id": _sid(1), "session_start": END - timedelta(hours=120)},
        {"session_id": _sid(2), "session_start": END - timedelta(hours=140)},
    ]
    server = _Server(
        rows=rows,
        # Full state no longer places the newest candidate in the window at all.
        full_state={
            _sid(1): rows[1]["session_start"],
            _sid(2): rows[2]["session_start"],
        },
    )
    page = _read(_builder(page_size=2), server)

    assert [row["session_id"] for row in page.rows] == [_sid(1), _sid(2)]


# --------------------------------------------------------------------------
# The width
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_the_issued_slice_stays_inside_the_row_budget():
    rows = [
        {"session_id": _sid(0), "session_start": END - timedelta(hours=100)},
        {"session_id": _sid(1), "session_start": END - timedelta(hours=120)},
        {"session_id": _sid(2), "session_start": END - timedelta(hours=140)},
    ]
    server = _Server(
        rows=rows,
        full_state={row["session_id"]: row["session_start"] for row in rows},
    )
    page = _read(_builder(page_size=2), server)

    assert page.slice_start is not None
    dense_hours = max(
        0.0,
        ((END - timedelta(hours=SPARSE_HOURS)) - page.slice_start) / timedelta(hours=1),
    )
    assert dense_hours * DENSE_ROWS_PER_HOUR <= BUDGET
    # The probe reads the index only, and the search is geometric, so a whole
    # request window is bracketed in a bounded handful of them.
    from tracer.selectors.session_candidate_slice import _MAX_DENSITY_PROBES

    assert 1 <= server.kinds.count("probe") <= _MAX_DENSITY_PROBES


@pytest.mark.unit
def test_a_window_inside_the_budget_is_never_narrowed():
    builder = _builder(page_size=2, start=END - timedelta(hours=SPARSE_HOURS))
    rows = [
        {"session_id": _sid(0), "session_start": END - timedelta(hours=100)},
        {"session_id": _sid(1), "session_start": END - timedelta(hours=120)},
    ]
    server = _Server(
        rows=rows,
        full_state={row["session_id"]: row["session_start"] for row in rows},
    )
    page = _read(builder, server)

    assert page.slice_start is None
    assert server.kinds == ["probe", "slice"]
    assert page.has_more is False


@pytest.mark.unit
def test_an_unreadable_estimate_reads_the_window_whole_rather_than_guessing():
    rows = [{"session_id": _sid(0), "session_start": END - timedelta(hours=100)}]
    server = _Server(
        rows=rows,
        full_state={_sid(0): rows[0]["session_start"]},
        estimate_columns=None,
    )
    page = _read(_builder(page_size=2), server)

    assert page.slice_start is None
    assert server.kinds == ["probe", "slice"]


@pytest.mark.unit
def test_a_builder_without_the_verifier_reads_the_window_whole():
    builder = _builder(page_size=2)
    rows = [{"session_id": _sid(0), "session_start": END - timedelta(hours=100)}]
    server = _Server(rows=rows, full_state={})
    with mock.patch.object(builder, "supports_bounded_filter_scan", return_value=False):
        page = _read(builder, server)

    # Narrowing without the full-state verifier would publish inflated starts,
    # so a request that cannot run one is not narrowed at all - and it never
    # pays for a probe either.
    assert page.slice_start is None
    assert server.kinds == ["slice"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "rows,columns,expected",
    [
        (
            [{"table": "spans", "rows": 5}, {"table": "spans", "rows": 7}],
            ["table", "rows"],
            12,
        ),
        ([{"table": "other", "rows": 5}], ["table", "rows"], None),
        ([{"table": "spans", "rows": "many"}], ["table", "rows"], None),
        ([{"table": "spans", "rows": 5}], ["parts"], None),
    ],
)
def test_density_estimate_reduces_only_shapes_it_can_read(rows, columns, expected):
    estimate = _builder().candidate_slice_density_estimate(rows, columns)
    assert (estimate if isinstance(estimate, int) else None) == expected


@pytest.mark.unit
def test_an_estimate_naming_no_part_is_not_reported_as_the_integer_zero():
    from tracer.selectors.filter_seed_width import EMPTY_DENSITY_ESTIMATE

    assert (
        _builder().candidate_slice_density_estimate([], ["table", "rows"])
        is EMPTY_DENSITY_ESTIMATE
    )


@pytest.mark.unit
@pytest.mark.parametrize("narrowed", [False, True])
def test_a_narrowed_page_publishes_its_count_as_a_lower_bound(narrowed):
    """A floor the scan cannot see below cannot produce a window total."""
    from tracer.tests.test_session_list_bounded_view import _view_and_request
    from tracer.views.trace_session import TraceSessionView

    rows = [
        {"session_id": _sid(0), "session_start": END - timedelta(hours=100)},
        {"session_id": _sid(1), "session_start": END - timedelta(hours=120)},
        {"session_id": _sid(2), "session_start": END - timedelta(hours=140)},
    ]
    server = _Server(
        rows=rows,
        full_state={row["session_id"]: row["session_start"] for row in rows},
        # Without a readable estimate the read is not narrowed, so the same
        # page is published with the exact count it has always carried.
        estimate_columns=("table", "rows") if narrowed else None,
    )
    view, request = _view_and_request()
    selected = TraceSessionView._select_session_page(
        view,
        request,
        project_id=PROJECT,
        project=None,
        analytics=server,
        validated_data={
            "filters": _filters(),
            "sort_params": [],
            "page_number": 0,
            "page_size": 2,
            "cursor_mode": True,
        },
    )

    assert selected.candidate_cursor is True
    assert selected.candidate_total_is_lower_bound is narrowed
    assert [row["session_id"] for row in selected.page_candidates] == [
        _sid(0),
        _sid(1),
    ]


@pytest.mark.unit
def test_the_density_probe_reads_the_index_and_names_no_projection_key():
    sql, params = _builder().build_candidate_slice_density_probe_query(
        slice_start=END - timedelta(hours=SPARSE_HOURS), slice_end=END
    )
    assert sql.lstrip().startswith("EXPLAIN ESTIMATE")
    # Spelled as the key expression, the optimizer could answer this from an
    # aggregate projection and report ITS rows, approving the widest slice.
    assert "toStartOfHour" not in sql
    assert " IN (" not in sql and "FINAL" not in sql
    assert "is_deleted" not in sql
    assert params["candidate_density_start_us"] < params["candidate_density_end_us"]


@pytest.mark.unit
def test_the_density_probe_stays_inside_the_request_window():
    builder = _builder()
    with pytest.raises(ValueError, match="inside the window"):
        builder.build_candidate_slice_density_probe_query(
            slice_start=START - timedelta(hours=1), slice_end=END
        )
