"""Ask a live ClickHouse whether the shipped unfiltered shape actually routes.

The rendered-SQL guards in ``test_hourly_aggregate_state_route`` pin the text.
They cannot tell you the optimiser accepts it — and the whole defect behind
issue #2839 was a shape that looked right and silently full-scanned, because
the projections on ``spans`` store a doubled ``-State`` and so never match a
plainly-written ``count()``.

The detector is ``force_optimize_projection = 1``, which raises when **no**
projection is used. The by-name variant (``force_optimize_projection_name``)
raises on "not chosen", which is indistinguishable from "not a candidate" and
misleads; it is deliberately not used here, and nothing in the product names a
projection either — the optimiser picks.

Everything below is ``EXPLAIN`` only: no rows are read, nothing is written and
no DDL is issued. The target is still fenced to a loopback ``test_`` database
on a port that is not one of the operator port-forwards, because a live-CH
test that quietly defaults to a forwarded production port has happened here
before.
"""

from __future__ import annotations

import os

import pytest

from tracer.services.clickhouse.query_builders.hourly_aggregate_states import (
    hourly_aggregate_state_source,
)

# Ports the operator forwards a remote ClickHouse onto. A read-only EXPLAIN is
# harmless, but a live test must never silently adopt one as its target.
_FORWARDED_PORTS = frozenset({19010, 19000, 19001, 19002, 18230, 18231, 18232})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _client():
    clickhouse_connect = pytest.importorskip("clickhouse_connect")
    host = os.getenv("CH25_HOST") or os.getenv("CH_HOST") or ""
    database = os.getenv("CH25_DATABASE") or os.getenv("CH_DATABASE") or ""
    port_text = os.getenv("CH25_HTTP_PORT") or os.getenv("CH_HTTP_PORT") or ""
    if not host or not database or not port_text:
        pytest.skip("no test ClickHouse configured")
    if host.strip().lower() not in _LOOPBACK_HOSTS:
        pytest.skip("test ClickHouse is not a loopback target")
    if not database.lower().lstrip("_").startswith("test_"):
        pytest.skip("test ClickHouse database is not a test_* database")
    port = int(port_text)
    assert port not in _FORWARDED_PORTS, (
        f"refusing to probe ClickHouse on forwarded port {port}"
    )
    try:
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=os.getenv("CH25_USER") or os.getenv("CH_USERNAME") or "default",
            password=os.getenv("CH25_PASSWORD") or os.getenv("CH_PASSWORD") or "",
            connect_timeout=5,
            send_receive_timeout=30,
        )
        client.query("SELECT 1")
    except Exception:  # noqa: BLE001 - any connection failure means "not available"
        pytest.skip("test ClickHouse is not reachable")
    return client


def _busiest_project_window(client):
    """Return a project and window that actually has rows, or skip."""

    rows = client.query(
        "SELECT project_id,"
        " toString(toStartOfHour(min(start_time))),"
        " toString(toStartOfHour(max(start_time)) + INTERVAL 1 HOUR)"
        " FROM spans GROUP BY project_id ORDER BY count() DESC LIMIT 1"
    ).result_rows
    if not rows:
        pytest.skip("test ClickHouse has no spans to plan against")
    return rows[0]


def _shipped_statement(project_id, window_start, window_end):
    source = hourly_aggregate_state_source(
        f"project_id = toUUID('{project_id}')",
        start_param="ignored_start",
        end_param="ignored_end",
    )
    source = source.replace(
        "%(ignored_start)s", f"toDateTime('{window_start}')"
    ).replace("%(ignored_end)s", f"toDateTime('{window_end}')")
    return (
        "SELECT toStartOfDay(hour) AS time_bucket,\n"
        "       (quantilesTDigestMerge(0.5, 0.95, 0.99)(latency_q))[1]"
        " AS avg_latency,\n"
        "       sumMerge(total_tokens_sum) AS total_tokens,\n"
        "       sumMerge(cost_sum) / greatest(countMerge(n), 1) AS avg_cost,\n"
        "       countMerge(n) AS traffic_count,\n"
        "       countMergeIf(n, status = 'ERROR') AS error_count\n"
        f"FROM {source}\n"
        "GROUP BY time_bucket\n"
        "ORDER BY time_bucket"
    )


@pytest.mark.integration
def test_optimiser_selects_a_projection_for_the_shipped_shape():
    """`force_optimize_projection = 1` raises unless a projection is used."""

    client = _client()
    if not client.query(
        "SELECT count() FROM system.projections"
        " WHERE database = currentDatabase() AND table = 'spans'"
    ).result_rows[0][0]:
        pytest.skip("spans carries no projections on this ClickHouse")
    project_id, window_start, window_end = _busiest_project_window(client)

    plan = client.query(
        "EXPLAIN indexes = 1 "
        + _shipped_statement(project_id, window_start, window_end),
        settings={"optimize_use_projections": 1, "force_optimize_projection": 1},
    ).result_rows

    rendered = "\n".join(str(row[0]) for row in plan)
    assert "ReadFromMergeTree" in rendered
    # The plan names whichever projection the cost model chose. Assert only
    # that it chose one: with a real WHERE it may legitimately prefer a
    # different projection than it does without, which is exactly why the
    # product code names none.
    assert "ReadFromMergeTree (" in rendered, rendered


@pytest.mark.integration
def test_plainly_written_aggregates_do_not_reach_the_hourly_states():
    """The doubled ``-State`` is why the obvious query was full-scanning.

    Same grouping, same window, aggregates written plainly. It cannot match an
    aggregate projection, because those store ``AggregateFunction(countState)``
    while ``count()`` looks for ``AggregateFunction(count)``. The plan must
    therefore not be an hourly-metrics projection read.
    """

    client = _client()
    if not client.query(
        "SELECT count() FROM system.projections"
        " WHERE database = currentDatabase() AND table = 'spans'"
    ).result_rows[0][0]:
        pytest.skip("spans carries no projections on this ClickHouse")
    project_id, window_start, window_end = _busiest_project_window(client)

    plan = client.query(
        "EXPLAIN indexes = 1 "
        "SELECT toStartOfDay(toStartOfHour(start_time)) AS time_bucket,"
        " count() AS traffic_count, sum(total_tokens) AS total_tokens"
        " FROM spans"
        f" WHERE project_id = toUUID('{project_id}')"
        f" AND toStartOfHour(start_time) >= toDateTime('{window_start}')"
        f" AND toStartOfHour(start_time) < toDateTime('{window_end}')"
        " GROUP BY time_bucket ORDER BY time_bucket",
        settings={"optimize_use_projections": 1},
    ).result_rows

    rendered = "\n".join(str(row[0]) for row in plan)
    assert "proj_metrics_hourly" not in rendered, rendered
