"""A candidate slice is not issued where the statement has no root scan.

The user-detail Sessions page with a scalar-attribute filter proves membership
and root-ness from one all-span replay seeded by the user's own sessions. Its
builder emits no root CTE, so the floor a slice binds is never read: the
"sliced" statement is the unsliced one byte for byte and reads the same rows.
Measured on the high-volume tenant through the view's entry point at three,
six and twelve months, the slice reader spent eleven or twelve density probes
and one or two such slices around the unsliced statement, 4.3-6.9 s of page
wall for a page that statement alone answered in 1.3-1.8 s warm.

These tests pin the repair from both sides: the builder answers whether a
raised floor narrows its statement by asking the rendered text, and the reader
issues that statement once, unsliced, when the answer is no - while every
shape whose statement does carry a root scan keeps the slice it had.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from tracer.selectors.session_candidate_slice import read_candidate_slice_page
from tracer.services.clickhouse.read_budget import ReadDeadline
from tracer.services.clickhouse.v2.query_builders.session_list import (
    SessionListQueryBuilderV2,
)

PROJECT = str(UUID(int=1))
USER = str(UUID(int=7))
END = datetime(2026, 9, 12)
START = END - timedelta(days=92)
FLOOR_TOKEN = "%(candidate_root_scan_start_us)s"


def _window():
    return {
        "column_id": "created_at",
        "filter_config": {
            "col_type": "SYSTEM_METRIC",
            "filter_type": "datetime",
            "filter_op": "between",
            "filter_value": [START.isoformat(), END.isoformat()],
        },
    }


def _user():
    return {
        "column_id": "end_user_id",
        "filter_config": {
            "col_type": "SYSTEM_METRIC",
            "filter_type": "text",
            "filter_op": "in",
            "filter_value": [USER],
        },
    }


def _long_text_in():
    return {
        "column_id": "metadata",
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": "text",
            "filter_op": "in",
            "filter_value": ["x" * 508, "y" * 498],
            "attribute_value_types": ["string", "string"],
        },
    }


def _boolean_equals():
    return {
        "column_id": "lk.interrupted",
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": "boolean",
            "filter_op": "equals",
            "filter_value": True,
        },
    }


def _number_equals():
    return {
        "column_id": "tier",
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": "number",
            "filter_op": "equals",
            "filter_value": 1,
        },
    }


# The shapes the view routes through the slice reader, by whether the
# statement they render carries a root scan for the floor to bound.
FUSED_SHAPES = {
    "user_detail_long_text_in": [_long_text_in(), _user(), _window()],
    "user_detail_boolean_equals": [_boolean_equals(), _user(), _window()],
    "user_detail_number_equals": [_number_equals(), _user(), _window()],
}
ROOT_SCAN_SHAPES = {
    "default_date_only": [_window()],
    "user_only": [_user(), _window()],
    "attribute_without_user": [_number_equals(), _window()],
}


def _builder(filters, page_size=25):
    return SessionListQueryBuilderV2(
        project_id=PROJECT,
        filters=filters,
        page_number=0,
        page_size=page_size,
        bounded_internal_scan=True,
    )


# --------------------------------------------------------------------------
# The builder's answer
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("shape", sorted(FUSED_SHAPES))
def test_a_fused_statement_reports_that_the_floor_narrows_nothing(shape):
    builder = _builder(FUSED_SHAPES[shape])
    assert builder.supports_candidate_cursor_page()

    assert builder.candidate_slice_narrows_root_scan() is False

    unsliced, _ = builder.build_candidate_cursor_page_query()
    sliced, params = builder.build_candidate_cursor_page_query(
        scan_start_time=END - timedelta(hours=195)
    )
    # The floor is bound and nothing reads it: same text, same rows.
    assert sliced == unsliced
    assert "candidate_root_scan_start_us" in params
    assert FLOOR_TOKEN not in sliced
    assert "candidate_root_identities" not in sliced


@pytest.mark.unit
@pytest.mark.parametrize("shape", sorted(ROOT_SCAN_SHAPES))
def test_a_statement_with_a_root_scan_reports_that_the_floor_narrows_it(shape):
    builder = _builder(ROOT_SCAN_SHAPES[shape])
    assert builder.supports_candidate_cursor_page()

    assert builder.candidate_slice_narrows_root_scan() is True

    sliced, _ = builder.build_candidate_cursor_page_query(
        scan_start_time=END - timedelta(hours=195)
    )
    assert FLOOR_TOKEN in sliced
    assert "candidate_root_identities" in sliced


@pytest.mark.unit
def test_asking_the_builder_leaves_its_bindings_untouched():
    """The verifier reads the request window out of ``builder.params``.

    The answer comes from a render, and a render that bound the floor into
    the builder's own params would make the full-state verifier verify the
    slice against itself.
    """

    builder = _builder(ROOT_SCAN_SHAPES["default_date_only"])
    before = dict(builder.params)

    builder.candidate_slice_narrows_root_scan()

    assert builder.params == before
    assert "candidate_root_scan_start_us" not in builder.params


# --------------------------------------------------------------------------
# The reader's schedule
# --------------------------------------------------------------------------


class _Server:
    """A fake CH that records what the reader asks of it.

    The density probe answers with an estimate far above the row budget, so
    a reader that asked it WOULD narrow; the candidate statement answers with
    the configured rows whatever floor it is bound with.
    """

    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    def execute_ch_query(self, query, params, *, timeout_ms, settings):
        if query.lstrip().startswith("EXPLAIN ESTIMATE"):
            self.calls.append(("probe", params))
            return SimpleNamespace(
                data=[{"table": "spans", "rows": 300_000_000}],
                columns=["table", "rows"],
            )
        if "AS remaining_count" in query:
            self.calls.append(("candidate", params))
            return SimpleNamespace(
                data=[{**row, "remaining_count": len(self.rows)} for row in self.rows],
                columns=["session_id", "session_start", "remaining_count"],
            )
        self.calls.append(("full_state", params))
        return SimpleNamespace(data=[], columns=["session_id", "start_time"])

    @property
    def kinds(self):
        return [kind for kind, _params in self.calls]


def _read(builder, server):
    return read_candidate_slice_page(
        builder=builder,
        analytics=server,
        deadline=ReadDeadline.start(30_000),
        read_settings=lambda rows: {"max_result_rows": rows},
        query_timeout_ms=9_000,
    )


def _rows(count):
    return [
        {
            "session_id": str(UUID(int=100 + i)),
            "session_start": END - timedelta(hours=i),
        }
        for i in range(count)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("shape", sorted(FUSED_SHAPES))
@pytest.mark.parametrize("found", [0, 18], ids=["empty", "short_of_a_page"])
def test_a_fused_statement_is_issued_once_unsliced_with_no_probe(shape, found):
    """One statement, the whole window, and the answer is the page.

    Before the repair, both of these pages - the empty one and the one short
    of ``page_size`` - widened: probes, a slice, more probes, a second slice,
    and finally the unsliced statement, every candidate statement identical.
    """

    server = _Server(rows=_rows(found))
    page = _read(_builder(FUSED_SHAPES[shape]), server)

    assert server.kinds == ["candidate"]
    ((_kind, params),) = server.calls
    # The whole request window, not a raised floor.
    assert params["candidate_root_scan_start_us"] == params["start_date_us"]
    assert page.slice_start is None
    assert page.statement_count == 1
    assert [row["session_id"] for row in page.rows] == [
        row["session_id"] for row in _rows(found)
    ]
    assert page.has_more is False
    # ``count() OVER()`` of the unsliced statement is the exact total.
    assert page.remaining_count == found


@pytest.mark.unit
def test_a_statement_with_a_root_scan_still_narrows():
    """The slice the default page relies on is not what this repair removes."""

    server = _Server(rows=_rows(30))
    page = _read(_builder(ROOT_SCAN_SHAPES["default_date_only"]), server)

    assert server.kinds[0] == "probe"
    assert "candidate" in server.kinds
    assert page.rows


@pytest.mark.unit
def test_a_builder_that_cannot_answer_is_taken_to_narrow():
    """Every builder this lane had before the question existed narrows."""

    inner = _builder(ROOT_SCAN_SHAPES["default_date_only"])

    class _Silent:
        def __getattr__(self, name):
            if name == "candidate_slice_narrows_root_scan":
                raise AttributeError(name)
            return getattr(inner, name)

    server = _Server(rows=_rows(30))
    _read(_Silent(), server)

    assert server.kinds[0] == "probe"
