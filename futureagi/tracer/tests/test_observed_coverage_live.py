"""Live-ClickHouse contract for the coverage probes' arrival-time gate.

``test_observed_coverage.py`` drives ``observed_scope_coverage`` with doubles
that decide "in flight" versus "settled" by looking for the gate's literal text
in the statement. That pins the shape of the SQL; it cannot tell whether the
predicate ClickHouse actually evaluates does what the text says. A gate that is
present but silently always-false would make every unindexed project read as
covered -- the one direction this module must never fail in -- and every unit
test would stay green. So the verdicts here come from the real
``observed_scope_coverage`` reading real tables, with ``created_at`` set
explicitly so arrival and the span's own clock can disagree.

Gated on the same explicitly isolated test ClickHouse as the other live catalog
contracts: unset, it skips; configured, a wrong verdict is a failure.
"""

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

KEY_COLUMNS = (
    "organization_id",
    "workspace_id",
    "project_id",
    "source_kind",
    "attribute_key",
    "attribute_type",
    "key_folded",
    "first_seen",
    "last_seen",
)


@pytest.fixture
def live():
    import clickhouse_connect

    host = os.environ.get("OBSERVED_CATALOG_TEST_CH_HOST")
    if not host:
        pytest.skip("requires an explicitly isolated observed catalog test ClickHouse")
    assert host in {"clickhouse", "127.0.0.1", "localhost"}
    client = clickhouse_connect.get_client(
        host=host,
        port=int(os.environ.get("OBSERVED_CATALOG_TEST_CH_PORT", "8123")),
        username=os.environ.get("OBSERVED_CATALOG_TEST_CH_USER", "test"),
        password=os.environ.get("OBSERVED_CATALOG_TEST_CH_PASSWORD", "test"),
    )
    database = "test_observed_coverage_" + uuid4().hex
    client.command(f"CREATE DATABASE {database}")
    try:
        client.command(
            f"""
CREATE TABLE {database}.observed_attribute_keys (
    organization_id String, workspace_id String, project_id String,
    source_kind LowCardinality(String), attribute_key String,
    attribute_type LowCardinality(String), key_folded String,
    first_seen SimpleAggregateFunction(min, DateTime64(6, 'UTC')),
    last_seen  SimpleAggregateFunction(max, DateTime64(6, 'UTC')))
ENGINE = AggregatingMergeTree
ORDER BY (organization_id, workspace_id, project_id, source_kind, attribute_key, attribute_type)
"""
        )
        # The columns the probes touch, with the production default for arrival.
        client.command(
            f"""
CREATE TABLE {database}.spans (
    project_id String,
    start_time DateTime64(6, 'UTC'),
    created_at DateTime64(6, 'UTC') DEFAULT now64(6, 'UTC'),
    attrs_string Map(LowCardinality(String), String),
    attrs_number Map(LowCardinality(String), Float64),
    attrs_bool Map(LowCardinality(String), UInt8),
    model LowCardinality(String) DEFAULT '',
    attributes_extra JSON(max_dynamic_paths=0))
ENGINE = MergeTree PARTITION BY toDate(start_time) ORDER BY (project_id, start_time)
"""
        )
        yield SimpleNamespace(client=client, database=database, reads=[])
    finally:
        client.command(f"DROP DATABASE IF EXISTS {database}")


def _coverage(live, scope):
    from tracer.services.clickhouse.v2.property_catalog.coverage import (
        observed_scope_coverage,
    )
    from tracer.services.clickhouse.v2.property_catalog.reader import ObservedRead

    class Executor:
        def execute(self, sql, params, timeout_ms=None, settings=None):
            result = live.client.query(sql, parameters=params)
            return SimpleNamespace(
                data=[dict(zip(result.column_names, row)) for row in result.result_rows]
            )

    class Client:
        # The probes name the bare ``spans`` table; point them at the fixture.
        def execute_read(self, sql, params=None, timeout_ms=None, settings=None):
            result = live.client.query(
                sql.replace("FROM spans", f"FROM {live.database}.spans"),
                parameters=params or {},
                settings={"max_threads": 1, "max_block_size": 1024},
            )
            live.reads.append(int((result.summary or {}).get("read_rows", -1)))
            return (
                list(result.result_rows),
                result.column_names,
                len(result.result_rows),
            )

        def execute(self, *args, **kwargs):
            raise AssertionError("coverage must use execute_read")

    deadline = SimpleNamespace(remaining_ms=lambda floor_ms=1, **_: 5_000)
    return observed_scope_coverage(
        scope=scope,
        deadline=deadline,
        observed=ObservedRead(
            Executor(), catalog_database=live.database, deadline=deadline
        ),
        client=Client(),
    )


def _scope(*projects):
    return {
        "organization_id": "org",
        "workspace_id": "ws",
        "project_ids": list(projects),
    }


def _spans(live, rows, *, bare=False):
    """Insert (project, start_time, created_at) rows; attributed unless ``bare``."""
    attrs = {} if bare else {"k": "v"}
    live.client.insert(
        f"{live.database}.spans",
        [[*row, attrs] for row in rows],
        column_names=["project_id", "start_time", "created_at", "attrs_string"],
    )


def _index(live, project, seen):
    live.client.insert(
        f"{live.database}.observed_attribute_keys",
        [["org", "ws", project, "custom_attribute", "k", "string", "k", seen, seen]],
        column_names=list(KEY_COLUMNS),
    )


def test_a_project_whose_spans_all_arrived_within_the_margin_is_not_a_gap(live):
    """Spans planted yesterday by their own clock, arrived seconds ago: in flight."""
    now = datetime.now(UTC)
    fresh = str(uuid4())
    _spans(
        live,
        [
            [fresh, now - timedelta(days=1), now - timedelta(seconds=20)],
            [fresh, now - timedelta(days=1, hours=-2), now - timedelta(minutes=30)],
        ],
    )
    result = _coverage(live, _scope(fresh))
    assert (result.complete, result.reason) == (True, "covered")


def test_a_span_that_arrived_before_the_margin_and_is_unindexed_is_a_gap(live):
    """The un-backfilled upgrade: history arrived long ago, index knows nothing.

    This is the case the unit doubles cannot protect: a gate that never matched
    would turn it into ``covered``.
    """
    now = datetime.now(UTC)
    stale = str(uuid4())
    _spans(
        live,
        [
            [stale, now - timedelta(days=1), now - timedelta(hours=3)],
            [stale, now - timedelta(seconds=5), now - timedelta(seconds=5)],
        ],
    )
    result = _coverage(live, _scope(stale))
    assert (result.complete, result.reason) == (False, "project_unindexed")


def test_an_in_flight_project_does_not_mask_a_settled_gap_in_the_same_scope(live):
    now = datetime.now(UTC)
    fresh, stale = str(uuid4()), str(uuid4())
    _spans(
        live,
        [
            [fresh, now - timedelta(days=1), now - timedelta(seconds=20)],
            [stale, now - timedelta(days=1), now - timedelta(hours=3)],
        ],
    )
    result = _coverage(live, _scope(fresh, stale))
    assert (result.complete, result.reason) == (False, "project_unindexed")


def test_a_late_arrival_below_the_floor_is_tolerated_until_it_settles(live):
    """An indexed project receiving yesterday's trace from a client buffer."""
    now = datetime.now(UTC)
    project = str(uuid4())
    _index(live, project, now - timedelta(minutes=10))
    _spans(
        live,
        [
            [project, now - timedelta(minutes=10), now - timedelta(minutes=10)],
            [project, now - timedelta(days=1), now - timedelta(seconds=10)],
        ],
    )
    assert _coverage(live, _scope(project)).complete is True

    # The same span, had it arrived two hours ago and still not been indexed.
    live.client.command(
        f"ALTER TABLE {live.database}.spans UPDATE created_at = now64(6, 'UTC') - INTERVAL 2 HOUR "
        f"WHERE project_id = '{project}' AND start_time < now64(6, 'UTC') - INTERVAL 12 HOUR "
        "SETTINGS mutations_sync = 2"
    )
    settled = _coverage(live, _scope(project))
    assert (settled.complete, settled.reason) == (False, "source_predates_index")


def test_a_project_of_bare_spans_is_not_a_gap_but_one_attributed_span_makes_it_one(
    live,
):
    """Nothing the catalog would index means nothing is missing, at any age."""
    now = datetime.now(UTC)
    project = str(uuid4())
    _spans(
        live, [[project, now - timedelta(days=2), now - timedelta(hours=3)]], bare=True
    )
    assert _coverage(live, _scope(project)).reason == "covered"

    _spans(live, [[project, now - timedelta(days=2), now - timedelta(hours=3)]])
    assert _coverage(live, _scope(project)).reason == "project_unindexed"


def test_a_span_carrying_only_extra_attributes_still_counts(live):
    """attributes_extra feeds the catalog too; an old unindexed one is a gap."""
    project = str(uuid4())
    live.client.command(
        f"INSERT INTO {live.database}.spans (project_id, start_time, created_at, attributes_extra) "
        f"VALUES ('{project}', now64(6, 'UTC') - INTERVAL 2 DAY, now64(6, 'UTC') - INTERVAL 3 HOUR, "
        '\'{"nested": {"a": 1}}\')'
    )
    assert _coverage(live, _scope(project)).reason == "project_unindexed"


def test_a_covered_project_costs_zero_rows_on_the_floor_probe(live):
    """The steady state of a healthy install must not scan history.

    Two days of hourly spans, all indexed from the oldest one: the floor probe
    has nothing below the floor to find, and with the scope-wide bound the
    partitions it would have to read do not exist. Measured through the
    server's own read_rows, so a bound that merely looks right cannot pass.
    """
    now = datetime.now(UTC)
    project = str(uuid4())
    oldest = (now - timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
    _spans(
        live,
        [
            [project, oldest + timedelta(hours=h), oldest + timedelta(hours=h)]
            for h in range(48)
        ],
    )
    _index(live, project, oldest)
    result = _coverage(live, _scope(project))
    assert (result.complete, result.reason) == (True, "covered")
    assert live.reads[-1] == 0, live.reads
