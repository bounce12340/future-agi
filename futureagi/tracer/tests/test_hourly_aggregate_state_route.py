"""The unfiltered graph and dashboard reads must emit one -State/-Merge shape.

These are the guards for issue #2839. The unfiltered system-metric graphs used
to read ``spans_hourly_rollup``, a materialized view fed once per insert
delivery, while ``spans`` collapses replayed deliveries back to one row per
dedup key. The two drifted apart permanently, so "All" rendered at a multiple
of the sum of its own filtered parts.

What makes the replacement readable at all is easy to break by tidying, so the
mechanical rules are pinned here rather than only at the call sites:

* aggregates written as ``-State`` inside and ``-Merge`` outside, because a
  ``PROJECTION`` body applies ``-State`` implicitly and these were declared
  with a second explicit one — a plainly-written ``count()`` looks for
  ``AggregateFunction(count)``, finds ``AggregateFunction(countState)``, and
  can never match;
* no cast around the token columns;
* the window predicate on ``toStartOfHour(start_time)``, the key expression;
* no projection named anywhere, because the optimiser chooses and with a real
  ``WHERE`` the cost model may legitimately prefer a different one.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import mock

import pytest

from tracer.services.clickhouse import graph_dispatch
from tracer.services.clickhouse.query_builders.hourly_aggregate_states import (
    hourly_aggregate_state_source,
)
from tracer.services.clickhouse.query_builders.time_series import (
    TimeSeriesQueryBuilder,
)
from tracer.views import dashboard as dashboard_view
from tracer.views.dashboard import _read_dashboard_rollup_fast_path

PROJECT_ID = "3f1d4b7a-0c2e-4a58-9f6b-1d2c3e4f5a6b"

_GRAPH_COLUMNS = [
    "time_bucket",
    "avg_latency",
    "total_tokens",
    "avg_cost",
    "traffic_count",
    "prompt_tokens",
    "completion_tokens",
    "error_rate",
]

_STATE_PAIRS = (
    ("countState() AS n", "countMerge(n)"),
    ("sumState(cost) AS cost_sum", "sumMerge(cost_sum)"),
    ("sumState(total_tokens) AS total_tokens_sum", "sumMerge(total_tokens_sum)"),
    ("sumState(prompt_tokens) AS prompt_tokens_sum", "sumMerge(prompt_tokens_sum)"),
    (
        "sumState(completion_tokens) AS completion_tokens_sum",
        "sumMerge(completion_tokens_sum)",
    ),
    (
        "quantilesTDigestState(0.5, 0.95, 0.99)(latency_ms) AS latency_q",
        "quantilesTDigestMerge(0.5, 0.95, 0.99)(latency_q)",
    ),
)


def _unfiltered_graph_sql(interval="day"):
    builder = TimeSeriesQueryBuilder(
        project_id=PROJECT_ID,
        filters=[],
        interval=interval,
        start_date=datetime(2026, 6, 1, tzinfo=UTC),
        end_date=datetime(2026, 7, 1, tzinfo=UTC),
    )
    query, _ = builder.build()
    return query


@pytest.mark.unit
@pytest.mark.parametrize("interval", ["hour", "day", "week", "month"])
def test_unfiltered_graph_pairs_every_state_with_its_merge(interval):
    query = _unfiltered_graph_sql(interval)
    for state, merge in _STATE_PAIRS:
        assert state in query, state
        assert merge in query, merge


@pytest.mark.unit
def test_unfiltered_graph_reads_spans_and_not_the_retired_rollup():
    query = _unfiltered_graph_sql()
    assert "FROM spans\n" in query
    assert "spans_hourly_rollup" not in query
    assert query.count("FROM spans") == 1


@pytest.mark.unit
def test_unfiltered_graph_keeps_status_in_the_inner_grouping():
    """`error_rate` is merged conditionally, so `status` must survive inward."""

    query = _unfiltered_graph_sql()
    assert "GROUP BY project_id, hour, status" in query
    assert "countMergeIf(n, status = 'ERROR')" in query


@pytest.mark.unit
def test_unfiltered_graph_windows_on_the_hour_key_expression():
    query = _unfiltered_graph_sql()
    assert "toStartOfHour(start_time) >= %(start_date)s" in query
    assert "toStartOfHour(start_time) < %(end_date)s" in query


@pytest.mark.unit
def test_unfiltered_graph_casts_nothing_and_pins_no_projection():
    query = _unfiltered_graph_sql()
    assert "toInt64(" not in query
    assert "proj_" not in query
    assert "FINAL" not in query.upper()
    assert "SAMPLE" not in query.upper()


@pytest.mark.unit
def test_no_source_file_names_a_projection():
    """The optimiser picks; naming one would freeze a cost-model decision."""

    import tracer.services.clickhouse.query_builders.hourly_aggregate_states as module

    with open(module.__file__) as handle:
        source = handle.read()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    body = code.split('"""', 2)[-1]
    assert "proj_metrics_hourly" not in body
    assert "force_optimize_projection_name" not in body


@pytest.mark.unit
def test_state_source_scopes_by_the_predicate_it_is_given():
    single = hourly_aggregate_state_source("project_id = %(project_id)s")
    many = hourly_aggregate_state_source("project_id IN %(project_ids)s")
    assert "WHERE project_id = %(project_id)s" in single
    assert "WHERE project_id IN %(project_ids)s" in many
    for rendered in (single, many):
        assert "is_deleted" not in rendered
        assert "PREWHERE" not in rendered


@pytest.mark.unit
@pytest.mark.parametrize(
    ("metric_id", "expected_exact"),
    [
        ("traffic", True),
        ("tokens", True),
        ("total_tokens", True),
        ("prompt_tokens", True),
        ("completion_tokens", True),
        ("cost", True),
        ("error_rate", True),
        ("latency", False),
    ],
)
def test_graph_reports_exactness_per_metric(metric_id, expected_exact):
    """Counts and sums are the base table's own numbers; latency is a tDigest."""

    analytics = mock.Mock()
    analytics.supports_per_query_read_settings = True
    analytics.execute_ch_query.return_value = SimpleNamespace(
        data=[
            {
                "time_bucket": datetime(2026, 6, 1, tzinfo=UTC),
                "avg_latency": 12.5,
                "total_tokens": 900,
                "avg_cost": 0.002,
                "traffic_count": 11,
                "prompt_tokens": 600,
                "completion_tokens": 300,
                "error_rate": 9.0,
            }
        ],
        columns=list(_GRAPH_COLUMNS),
    )

    response = graph_dispatch.fetch_system_metric_graph_ch(
        analytics=analytics,
        project_id=PROJECT_ID,
        filters=[],
        interval="day",
        metric_id=metric_id,
        observe_type="trace",
    )

    assert response["query_status"] == "complete"
    assert response["query_exact"] is expected_exact
    assert "spans_hourly_rollup" not in analytics.execute_ch_query.call_args.args[0]


class _CapturingAnalytics:
    supports_per_query_read_settings = True

    def __init__(self):
        self.queries = []

    def execute_ch_query(self, query, params, *, timeout_ms, settings):
        self.queries.append(query)
        aliases = [
            part.split()[0].rstrip(",")
            for part in query.split(" AS ")
            if part.startswith("metric_")
        ]
        row = {"time_bucket": params["start_date"]}
        row.update(dict.fromkeys(aliases, 1))
        return SimpleNamespace(data=[row], columns=["time_bucket", *aliases])


def _widget_config(metric_id, aggregation):
    return {
        "project_ids": [PROJECT_ID],
        "time_range": {"preset": "30D"},
        "granularity": "day",
        "metrics": [
            {
                "id": metric_id,
                "name": metric_id,
                "type": "system_metric",
                "source": "traces",
                "aggregation": aggregation,
                "filters": [],
            }
        ],
        "filters": [],
        "breakdowns": [],
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("metric_id", "aggregation", "expected_exact"),
    [
        ("tokens", "sum", True),
        ("cost", "avg", True),
        ("error_rate", "avg", True),
        ("span_count", "count", True),
        ("project", "count_distinct", True),
        ("latency", "avg", False),
        ("latency", "p95", False),
    ],
)
def test_dashboard_widget_routes_and_reports_exactness(
    monkeypatch, metric_id, aggregation, expected_exact
):
    analytics = _CapturingAnalytics()
    monkeypatch.setattr(dashboard_view, "V2AnalyticsQueryService", lambda: analytics)

    result = _read_dashboard_rollup_fast_path(_widget_config(metric_id, aggregation))

    assert result["query_status"] == "complete"
    assert result["query_exact"] is expected_exact
    assert result["metrics"][0]["query_exact"] is expected_exact
    query = analytics.queries[0]
    assert "spans_hourly_rollup" not in query
    assert "FROM spans\n" in query
    assert "countState() AS n" in query
    assert "toStartOfHour(start_time) >= %(start_date)s" in query
    assert "proj_" not in query
    assert "FINAL" not in query.upper()


@pytest.mark.unit
def test_dashboard_widget_scopes_the_inner_read_to_the_requested_projects():
    """The scope has to sit inside, on the states' own key column."""

    analytics = _CapturingAnalytics()
    with mock.patch.object(
        dashboard_view, "V2AnalyticsQueryService", lambda: analytics
    ):
        _read_dashboard_rollup_fast_path(_widget_config("tokens", "sum"))

    query = analytics.queries[0]
    assert "WHERE project_id IN %(project_ids)s" in query
    assert "PREWHERE" not in query
