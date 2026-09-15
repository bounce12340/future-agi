"""Scope rules for the user-detail trace candidate seed.

``matching_user_trace_identities`` is a trace-membership superset read over
the whole ``spans`` table. Two properties keep it both exact and affordable,
and each has a way of quietly regressing:

* it is bounded by the **request window** plus the adjacent-day envelope —
  not by the narrower slice, which would drop a witness a keyset
  continuation still needs, and not by nothing at all, which rescans the
  project's entire span history for a one-day page;
* it compares ``end_user_id`` on its own type whenever the supplied value is
  already canonical UUID text, which is what lets the read use the column's
  skip index and the ``(project_id, end_user_id, start_time)`` projection
  prefix. Any other spelling stays on the textual comparison, because
  ``toString`` only ever emits canonical text and a native comparison would
  match rows the textual one does not.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tracer.services.clickhouse.query_builders.trace_list import TraceListQueryBuilder
from tracer.services.clickhouse.v2.query_builders.trace_list import (
    TraceListQueryBuilderV2,
)
from tracer.tests.test_bounded_trace_filter_reads import (
    END,
    PROJECT_ID,
    _render_driver_sql,
    _time_filter,
)

pytestmark = pytest.mark.unit

CANONICAL_USER = "50f8845d-e410-5ceb-9bb5-a0d5e7ca6773"
SECOND_USER = "6f1c0b22-6b1e-5a3a-9f21-2c0f8b5d4e77"
REQUEST_START = END - timedelta(days=30)


def _user_filter(value, operation="in", column_id="end_user_id"):
    return {
        "column_id": column_id,
        "filter_config": {
            "col_type": "SYSTEM_METRIC",
            "filter_type": "text",
            "filter_op": operation,
            "filter_value": value,
        },
    }


def _seed(builder, *, slice_start=REQUEST_START, slice_end=END, limit=200):
    sql, params = builder.build_filter_candidate_seed_page(
        slice_start=slice_start,
        slice_end=slice_end,
        limit=limit,
    )
    _render_driver_sql(sql, params)
    cte = sql.split("SELECT trace_id, id AS root_span_id", 1)[0]
    return sql, cte, params


def _builder(filters, cls=TraceListQueryBuilderV2):
    builder = cls(project_id=PROJECT_ID, filters=filters)
    assert builder.supports_filter_candidate_seed_page() is True
    return builder


def test_candidate_cte_is_bounded_by_the_request_window_not_the_slice():
    builder = _builder(
        [_time_filter(REQUEST_START, END), _user_filter([CANONICAL_USER])]
    )
    # A slice far narrower than the request window: the envelope must still
    # follow the window, or a later keyset page loses witnesses that this
    # page's slice never covered.
    slice_start = END - timedelta(hours=1)
    sql, cte, params = _seed(builder, slice_start=slice_start)

    assert "matching_user_trace_identities AS" in cte
    assert "start_time >= %(start_date)s - INTERVAL 1 DAY" in cte
    assert "start_time < %(end_date)s + INTERVAL 1 DAY" in cte
    assert params["start_date"] == REQUEST_START
    assert params["end_date"] == END
    # The slice bounds the ordered root read only, never the membership CTE.
    assert "filter_slice_start_us" not in cte
    assert "filter_slice_end_us" not in cte
    assert "fromUnixTimestamp64Micro(%(filter_slice_start_us)s)" in sql
    # Still never truncated: an inner LIMIT could hide an older matching root.
    assert "LIMIT" not in cte


def test_external_user_candidate_cte_carries_the_same_envelope():
    builder = _builder(
        [
            _time_filter(REQUEST_START, END),
            _user_filter("10000004", operation="equals", column_id="user_id"),
        ]
    )
    _sql, cte, params = _seed(builder)

    # The external identifier resolves through the curated end-user dimension;
    # the spans side of that membership read needs the envelope just the same.
    assert "FROM end_user_id_remap AS remap_match FINAL" in cte
    assert "start_time >= %(start_date)s - INTERVAL 1 DAY" in cte
    assert "start_time < %(end_date)s + INTERVAL 1 DAY" in cte
    assert params["col_1"] == "10000004"


def test_canonical_uuid_membership_compares_the_native_column():
    builder = _builder(
        [
            _time_filter(REQUEST_START, END),
            _user_filter([CANONICAL_USER, SECOND_USER]),
        ]
    )
    _sql, cte, params = _seed(builder)

    assert (
        "end_user_id IN (toUUID(%(end_user_uuid_1)s), toUUID(%(end_user_uuid_2)s))"
        in cte
    )
    assert "toString(end_user_id)" not in cte
    assert params["end_user_uuid_1"] == CANONICAL_USER
    assert params["end_user_uuid_2"] == SECOND_USER


def test_canonical_uuid_equality_compares_the_native_column():
    builder = _builder(
        [
            _time_filter(REQUEST_START, END),
            _user_filter(CANONICAL_USER, operation="equals"),
        ]
    )
    _sql, cte, params = _seed(builder)

    assert "end_user_id = toUUID(%(end_user_uuid_1)s)" in cte
    assert params["end_user_uuid_1"] == CANONICAL_USER


@pytest.mark.parametrize(
    "spelling",
    [
        CANONICAL_USER.upper(),
        "{" + CANONICAL_USER + "}",
        "urn:uuid:" + CANONICAL_USER,
        CANONICAL_USER.replace("-", ""),
    ],
)
def test_non_canonical_uuid_spelling_keeps_the_textual_predicate(spelling):
    # These all parse as the same UUID, but ``toString(end_user_id)`` never
    # emits them, so today they match nothing. Comparing natively would start
    # matching the user's rows — a different result set, not a faster one.
    builder = _builder(
        [_time_filter(REQUEST_START, END), _user_filter([spelling])]
    )
    _sql, cte, params = _seed(builder)

    assert "toString(end_user_id) IN %(col_1)s" in cte
    assert "toUUID(" not in cte
    assert params["col_1"] == (spelling,)


@pytest.mark.parametrize(
    "operation", ["contains", "starts_with", "ends_with", "not_equals", "not_in"]
)
def test_negated_and_substring_user_filters_never_reach_the_candidate_seed(
    operation,
):
    # Only a positive exact user leaf is a necessary trace-membership
    # condition, so these keep the existing classifier path. The native
    # comparison therefore never has to reason about them.
    value = [CANONICAL_USER] if operation == "not_in" else CANONICAL_USER[:8]
    builder = TraceListQueryBuilderV2(
        project_id=PROJECT_ID,
        filters=[
            _time_filter(REQUEST_START, END),
            _user_filter(value, operation=operation),
        ],
    )

    assert builder.supports_filter_candidate_seed_page() is False


def test_mixed_canonical_and_free_text_membership_stays_textual():
    builder = _builder(
        [
            _time_filter(REQUEST_START, END),
            _user_filter([CANONICAL_USER, "not-a-uuid"]),
        ]
    )
    _sql, cte, params = _seed(builder)

    assert "toString(end_user_id) IN %(col_1)s" in cte
    assert params["col_1"] == (CANONICAL_USER, "not-a-uuid")


def test_legacy_builder_membership_uses_its_own_event_time_column():
    builder = _builder(
        [_time_filter(REQUEST_START, END), _user_filter([CANONICAL_USER])],
        cls=TraceListQueryBuilder,
    )
    _sql, cte, _params = _seed(builder)

    # The legacy physical table prunes on ``created_at``; only the bound's
    # column differs, never its presence.
    assert "created_at >= %(start_date)s - INTERVAL 1 DAY" in cte
    assert "end_user_id IN (toUUID(%(end_user_uuid_1)s))" in cte
