"""The filtered graph must decide its lane from the index, not from the wall.

A filtered trace/span graph reads every physical span in its window. On the
highest-volume reference tenant that window is hundreds of millions of rows:
the interactive statement expires at thirty seconds and publishes nothing, and
the background worker then runs the identical SQL anyway. These tests pin the
routing decision that has to happen BEFORE the statement, and in particular
pin that a probe which cannot answer never licenses an unbounded read.

Every assertion here is about which statements are ISSUED. None of them
changes what a statement returns: the graph SQL and its window parameters are
byte-identical on both lanes, which is the property
``test_gate_never_narrows_the_statement_window`` states directly.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tracer.services.clickhouse import graph_dispatch, graph_read_cost

PROJECT_ID = "3f7d5b3a-2c41-4f8e-9b0d-5a1c7e2f4d60"
ORG_ID = "6a2b1c4d-8e3f-4a5b-9c7d-0e1f2a3b4c5d"

GRAPH_COLUMNS = [
    "time_bucket",
    "avg_latency",
    "total_tokens",
    "avg_cost",
    "traffic_count",
    "prompt_tokens",
    "completion_tokens",
    "error_rate",
]
ESTIMATE_COLUMNS = ["database", "table", "parts", "rows", "marks"]


def _window(days: int = 30) -> dict:
    end = datetime(2026, 9, 16, tzinfo=UTC)
    start = end - timedelta(days=days)
    return {
        "column_id": "created_at",
        "filter_config": {
            "col_type": "SYSTEM_METRIC",
            "filter_type": "datetime",
            "filter_op": "between",
            "filter_value": [start.isoformat(), end.isoformat()],
        },
    }


def _attribute_filter(key: str = "deployment.environment", value: str = "production"):
    return {
        "column_id": key,
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": "text",
            "filter_op": "equals",
            "filter_value": value,
        },
    }


def _estimate(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=[
            {
                "database": "default",
                "table": "spans",
                "parts": 207,
                "rows": rows,
                "marks": max(1, rows // 8192),
            }
        ],
        columns=list(ESTIMATE_COLUMNS),
        query_time_ms=1,
    )


class Analytics:
    """Answer the cost probe with *estimated_rows*; record every statement."""

    supports_per_query_read_settings = True

    def __init__(self, *, estimated_rows, seed_estimate=None, seed_raises=False):
        self.calls = []
        self._estimated_rows = estimated_rows
        self._seed_estimate = seed_estimate
        self._seed_raises = seed_raises

    def execute_ch_query(self, query, params, **kwargs):
        self.calls.append((query, dict(params), kwargs))
        if "EXPLAIN ESTIMATE" not in query:
            return SimpleNamespace(
                data=[], columns=list(GRAPH_COLUMNS), query_time_ms=1
            )
        if "graph_cost_project_id" in query:
            if self._estimated_rows is None:
                return SimpleNamespace(data=[], columns=[], query_time_ms=1)
            return _estimate(self._estimated_rows)
        if self._seed_raises:
            raise TimeoutError("seed probe exceeded its budget")
        if self._seed_estimate is None:
            return SimpleNamespace(data=[], columns=[], query_time_ms=1)
        return _estimate(self._seed_estimate)

    @property
    def statements(self):
        return [call for call in self.calls if "EXPLAIN ESTIMATE" not in call[0]]

    @property
    def cost_probes(self):
        return [call for call in self.calls if "graph_cost_project_id" in call[0]]

    @property
    def seed_probes(self):
        return [
            call
            for call in self.calls
            if "EXPLAIN ESTIMATE" in call[0] and "graph_cost_project_id" not in call[0]
        ]


@pytest.fixture
def scheduled(monkeypatch):
    calls = []

    def _read_or_schedule(namespace, identity, **kwargs):
        calls.append((namespace, identity, kwargs))
        if not kwargs.get("refresh"):
            # A cold cache probe. Returning the pending envelope here would
            # short-circuit every case below before any read is routed.
            return None
        return dict(kwargs["pending_payload"])

    monkeypatch.setattr(
        graph_dispatch,
        "read_or_schedule_exact_snapshot",
        _read_or_schedule,
    )
    return calls


def _fetch(analytics, *, observe_type="trace", organization_id=ORG_ID, filters=None):
    return graph_dispatch.fetch_system_metric_graph_ch(
        analytics=analytics,
        project_id=PROJECT_ID,
        filters=filters if filters is not None else [_window(), _attribute_filter()],
        interval="day",
        metric_id="traffic",
        observe_type=observe_type,
        organization_id=organization_id,
        workspace_id=None,
    )


# --- the read that cannot finish must never be issued ----------------------


@pytest.mark.unit
def test_unaffordable_trace_graph_is_scheduled_without_issuing_the_statement(
    scheduled,
):
    """87.4M estimated rows on a 30 s wall: schedule, do not spend it."""

    analytics = Analytics(estimated_rows=87_400_000, seed_estimate=87_300_000)
    response = _fetch(analytics)

    assert analytics.statements == [], "the graph statement must not be issued"
    assert len(analytics.cost_probes) == 1
    refreshes = [call for call in scheduled if call[2]["refresh"] is True]
    assert len(refreshes) == 1
    assert refreshes[0][0] == "observe-system-graph"
    assert response["query_status"] == "pending"


@pytest.mark.unit
def test_unaffordable_span_graph_is_scheduled_although_it_never_seeds(scheduled):
    """A span graph compiles no trace witness, so the window IS the read."""

    analytics = Analytics(estimated_rows=87_400_000)
    response = _fetch(analytics, observe_type="span")

    assert analytics.statements == []
    assert analytics.seed_probes == [], "a span graph has no seed to probe"
    assert response["query_status"] == "pending"


@pytest.mark.unit
def test_seed_probe_failure_on_an_unaffordable_read_schedules_not_scans(scheduled):
    """The deleted fallback: a probe that times out must not mean 'read all'.

    This is the exact production shape. The seed probe exceeded its own
    1,500 ms budget on the high-volume tenant, the dispatcher swallowed the
    failure, and the unseeded full-window statement then ran to the thirty
    second wall and returned nothing.
    """

    analytics = Analytics(estimated_rows=87_400_000, seed_raises=True)
    response = _fetch(analytics)

    assert analytics.statements == [], (
        "a seed probe that cannot answer is not a licence to read the window"
    )
    assert len(analytics.seed_probes) >= 1
    assert response["query_status"] == "pending"


@pytest.mark.unit
def test_a_read_no_wall_can_absorb_is_refused_not_scheduled(scheduled):
    """The background lane is a wider wall, not an unbounded one.

    Twelve months on the high-volume tenant is 311.9M estimated rows. At the
    measured rate the 180 s background wall affords 174.2M, so the worker
    would expire too - and a failed cold refresh leaves no snapshot while the
    next poll is free to ask for another one. Scheduling that is a spinner
    with no terminal state and one full-window scan per poll cycle.
    """
    from django.conf import settings

    hopeless = settings.GRAPH_BACKGROUND_WALL_MS * graph_read_cost._RAW_SCAN_ROWS_PER_MS
    analytics = Analytics(estimated_rows=hopeless + 1, seed_raises=True)
    response = _fetch(analytics)

    assert analytics.statements == []
    assert [call for call in scheduled if call[2]["refresh"] is True] == []
    assert response["query_status"] == "degraded"
    assert response["query_error_code"] == "read_budget_exceeded"
    assert response["query_provenance"] == "read_cost_gate"


@pytest.mark.unit
def test_a_read_the_background_wall_can_absorb_is_still_scheduled(scheduled):
    """Thirty days on the same tenant fits the worker, so it must reach it."""
    from django.conf import settings

    hopeless = settings.GRAPH_BACKGROUND_WALL_MS * graph_read_cost._RAW_SCAN_ROWS_PER_MS
    analytics = Analytics(estimated_rows=hopeless, seed_raises=True)
    response = _fetch(analytics)

    assert analytics.statements == []
    assert len([call for call in scheduled if call[2]["refresh"] is True]) == 1
    assert response["query_status"] == "pending"


@pytest.mark.unit
def test_unaffordable_read_without_a_background_lane_fails_fast(scheduled):
    """No organization to schedule under: refuse now, not in thirty seconds."""

    analytics = Analytics(estimated_rows=87_400_000, seed_raises=True)
    response = _fetch(analytics, organization_id=None)

    assert analytics.statements == []
    assert [call for call in scheduled if call[2]["refresh"] is True] == []
    assert response["query_status"] == "degraded"
    assert response["query_error_code"] == "read_budget_exceeded"
    assert response["query_provenance"] == "read_cost_gate"


# --- the read that can finish must still run inline ------------------------


@pytest.mark.unit
def test_affordable_read_still_runs_inline(scheduled):
    """The mid-volume tenant completes today and must keep completing."""

    analytics = Analytics(estimated_rows=10_500_000)
    response = _fetch(analytics)

    assert len(analytics.statements) == 1
    assert response["query_complete"] is True
    assert [call for call in scheduled if call[2]["refresh"] is True] == []


@pytest.mark.unit
def test_an_unknown_estimate_does_not_divert_the_read(scheduled):
    """The gate fires on proof of a large scan, never on the absence of one."""

    analytics = Analytics(estimated_rows=None)
    response = _fetch(analytics)

    assert len(analytics.statements) == 1
    assert response["query_complete"] is True


@pytest.mark.unit
def test_an_admitted_seed_rescues_an_unaffordable_window(scheduled):
    """A selective witness is the one thing that can bound this read inline."""

    analytics = Analytics(estimated_rows=87_400_000, seed_estimate=1_600_000)
    response = _fetch(analytics)

    assert len(analytics.statements) == 1
    statement = analytics.statements[0][0]
    assert "GROUP BY trace_id" in statement, "the statement must carry the seed set"
    assert response["query_complete"] is True


# --- exactness: routing must not change what is read -----------------------


@pytest.mark.unit
def test_gate_never_narrows_the_statement_window(scheduled):
    """A routed read covers the same window an ungated one would.

    The gate decides WHICH lane runs the statement. If it ever changed the
    window, a published chart would silently describe less than the user
    asked for, which is worse than a slow one.
    """

    ungated = Analytics(estimated_rows=None)
    _fetch(ungated)
    _query, ungated_params, _kwargs = ungated.statements[0]

    seeded = Analytics(estimated_rows=87_400_000, seed_estimate=1_600_000)
    _fetch(seeded)
    _query, seeded_params, _kwargs = seeded.statements[0]

    for key in (
        "start_date",
        "end_date",
        "graph_witness_start_date",
        "graph_witness_end_date",
    ):
        assert seeded_params[key] == ungated_params[key], key


# --- the probes themselves -------------------------------------------------


@pytest.mark.unit
def test_cost_probe_is_metadata_only(scheduled):
    """It must read no parts: no predicate, no indexHint, no IN subquery."""

    analytics = Analytics(estimated_rows=10_500_000)
    _fetch(analytics)
    query, params, kwargs = analytics.cost_probes[0]

    assert query.strip().startswith("EXPLAIN ESTIMATE")
    lowered = query.lower()
    assert "indexhint" not in lowered
    assert "attrs_string" not in lowered
    assert "attrs_number" not in lowered
    assert " in (" not in lowered, "an EXPLAIN executes an IN subquery to plan it"
    assert " final" not in lowered
    assert "sample " not in lowered
    assert "is_deleted" not in lowered, "not in the primary key; it cannot narrow"
    # An estimate routed to a projection describes the projection, not the
    # base table the graph statement reads.
    assert kwargs["settings"]["optimize_use_projections"] == 0
    assert params["graph_cost_project_id"] == PROJECT_ID


@pytest.mark.unit
def test_cost_probe_covers_the_widest_window_the_statement_can_scan(scheduled):
    """The raw statement widens its scan by a day at each end; so must this."""

    analytics = Analytics(estimated_rows=10_500_000)
    _fetch(analytics)
    _query, cost_params, _kwargs = analytics.cost_probes[0]
    _query, statement_params, _kwargs = analytics.statements[0]

    assert (
        cost_params["graph_cost_scan_start"]
        <= statement_params["graph_witness_start_date"]
    )
    assert (
        cost_params["graph_cost_scan_end"] >= statement_params["graph_witness_end_date"]
    )


@pytest.mark.unit
def test_seed_probe_is_not_pinned_to_one_worker(scheduled):
    """Index analysis parallelises, and this probe is nothing but that.

    Measured on production against the highest-volume reference tenant, one
    worker cost 831 ms at thirty days and 2,777 ms at twelve months - past the
    probe's own 1,500 ms budget. The same estimate at the graph statement's
    own worker count took 63 ms and 314 ms.
    """
    from django.conf import settings

    analytics = Analytics(estimated_rows=87_400_000, seed_estimate=1_600_000)
    _fetch(analytics)
    _query, _params, kwargs = analytics.seed_probes[0]

    assert (
        kwargs["settings"]["max_threads"] == settings.DASHBOARD_TRACE_READ_MAX_THREADS
    )
    assert kwargs["settings"]["max_threads"] > 1
    assert kwargs["settings"]["optimize_use_projections"] == 0


# --- the estimate reducer --------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "columns", "expected"),
    [
        (
            [{"table": "spans", "rows": 5}, {"table": "spans", "rows": 7}],
            ESTIMATE_COLUMNS,
            12,
        ),
        ([], ESTIMATE_COLUMNS, 0),
        ([{"table": "spans", "rows": 5}], ["time_bucket", "traffic_count"], None),
        ([{"table": "other", "rows": 5}], ESTIMATE_COLUMNS, None),
        ([{"table": "spans", "rows": "many"}], ESTIMATE_COLUMNS, None),
        ([{"table": "spans"}], ESTIMATE_COLUMNS, None),
    ],
)
def test_estimate_reducer_tells_zero_from_unknown(rows, columns, expected):
    assert graph_read_cost._reduce_estimate(rows, columns) == expected


@pytest.mark.unit
def test_affordable_rows_scale_with_the_wall_not_with_a_window():
    """Doubling the deadline must double what it can afford."""

    rows = graph_read_cost._RAW_SCAN_ROWS_PER_MS * 10_000
    assert graph_read_cost.raw_graph_scan_fits_wall(rows, remaining_ms=10_000) is True
    assert (
        graph_read_cost.raw_graph_scan_fits_wall(rows + 1, remaining_ms=10_000) is False
    )
    assert (
        graph_read_cost.raw_graph_scan_fits_wall(rows + 1, remaining_ms=20_000) is True
    )
