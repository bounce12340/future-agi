"""The observed index must never report history it does not hold as complete.

Upgrading onto the observed read path leaves the index empty for everything
ingested before it existed. These tests pin the contract that makes that state
visible instead of silently indistinguishable from "this project has no custom
attributes".
"""

from types import SimpleNamespace

import pytest

from tracer.services.clickhouse.v2.property_catalog.coverage import (
    _PROBE_SETTINGS,
    _PROBE_WALL_MS,
    Coverage,
    observed_scope_coverage,
)

SCOPE = {
    "organization_id": "org-1",
    "workspace_id": "ws-1",
    "project_ids": ("p1",),
}
TWO_PROJECTS = {**SCOPE, "project_ids": ("p1", "p2")}


class _Observed:
    """Stands in for the catalog executor, recording the SQL it is given."""

    def __init__(self, rows=None, raises=None):
        self.rows = rows if rows is not None else []
        self.raises = raises
        self.database = "property_catalog"
        self.queries = []

    def execute(self, sql, params, limit):
        self.queries.append((sql, params, limit))
        if self.raises:
            raise self.raises
        return self.rows


class _Client:
    """Stands in for the source span client.

    ``below`` names the projects that have spans older than their index floor.
    """

    def __init__(self, below=(), raises=None, any_span=None, in_flight=()):
        self.below = set(below)
        # Projects that have at least one span. Defaults to "the ones below",
        # so an empty project is the default for anything unindexed.
        self.any_span = set(below) if any_span is None else set(any_span)
        # Projects whose every span ARRIVED within the margin (created_at is
        # recent). The real table would exclude them under the arrival
        # predicate and include them without it, so the double does the same:
        # it honours the gate only when the statement actually carries it.
        self.in_flight = set(in_flight)
        self.raises = raises
        self.calls = []

    def _settled(self, sql, project):
        gated = "created_at < now64(6, 'UTC') - INTERVAL 1 HOUR" in sql
        return not (gated and project in self.in_flight)

    def execute_read(self, sql, params=None, timeout_ms=None, settings=None):
        self.calls.append({"sql": sql, "parameters": params, "settings": settings,
                           "timeout_ms": timeout_ms})
        if self.raises:
            raise self.raises
        ids = list((params or {}).get("project_ids") or ())
        if "start_time" not in sql:
            # The index-less probe asks "do any of these hold a settled span".
            rows = [(1,)] if any(
                p in self.any_span and self._settled(sql, p) for p in ids
            ) else []
        else:
            hit = next(
                (p for p in ids if p in self.below and self._settled(sql, p)), None
            )
            rows = [(hit,)] if hit else []
        # execute_read returns (rows, column_types, elapsed).
        return rows, [], 0.0

    def execute(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError(
            "coverage must use execute_read: ClickHouseClient.execute() discards "
            "`settings` on a server-readonly client, silently dropping the "
            "bounded probe settings"
        )


def _coverage(*, rows=None, below=(), scope=SCOPE, observed=None, client=None):
    return observed_scope_coverage(
        scope=scope,
        deadline=SimpleNamespace(remaining_ms=lambda floor_ms=1: 10_000),
        observed=observed or _Observed(rows if rows is not None else []),
        client=client or _Client(below),
    )


@pytest.mark.unit
def test_fully_covered_index_reports_complete():
    """The steady state: nothing in the source predates the index."""
    result = _coverage(rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}])
    assert result.complete is True
    assert result.status == "complete"
    assert result.reason == "covered"


@pytest.mark.unit
def test_source_older_than_index_reports_partial_not_complete():
    """The upgrade case. This is the defect the module exists to prevent."""
    result = _coverage(
        rows=[{"project_id": "p1", "floor": "2026-06-01 00:00:00"}],
        below=("p1",),
    )
    assert result.complete is False
    assert result.status == "partial"
    assert result.reason == "source_predates_index"
    # The floor is surfaced so a caller can say what the answer covers.
    assert result.floor == "2026-06-01 00:00:00"


@pytest.mark.unit
def test_unindexed_project_with_spans_is_a_gap():
    """Spans present but nothing indexed is exactly the un-backfilled upgrade."""
    result = _coverage(rows=[], scope=SCOPE, client=_Client(any_span=("p1",)))
    assert result.complete is False
    assert result.reason == "project_unindexed"


@pytest.mark.unit
def test_unindexed_project_without_spans_is_not_a_gap():
    """An empty project has nothing to index, so it is genuinely covered.

    Regression: treating every floor-less project as suspicious dragged any
    scope containing an empty project to "partial". Workspaces routinely hold
    projects with no traces yet, so this made healthy installs report partial
    and broke dashboard flows asserting completeness.
    """
    result = _coverage(rows=[], scope=SCOPE, client=_Client(any_span=()))
    assert result.complete is True
    assert result.reason == "covered"


@pytest.mark.unit
def test_unindexed_project_with_only_in_flight_spans_is_not_a_gap():
    """A project created seconds ago is ingestion in flight, not a missing backfill.

    Its spans are in the source (the collector wrote them) and not yet in the
    index (the consumer is seconds behind). Reporting that as ``partial`` made
    every freshly created project answer incomplete for the length of the
    consumer lag -- whatever the spans' own timestamps said, because e2e
    fixtures plant yesterday's spans into a project created today. The
    Playwright trace of the failing flow read exactly:

        query_complete=false reason=project_unindexed values=[]

    on the first poll after ingest, and the same flow passed on the next run.
    """
    result = _coverage(
        rows=[], scope=SCOPE, client=_Client(any_span=("p1",), in_flight=("p1",))
    )
    assert result.complete is True
    assert result.reason == "covered"


@pytest.mark.unit
def test_in_flight_project_does_not_mask_a_settled_gap_in_the_same_scope():
    """One IN-query, LIMIT 1: the fresh project must not hide the stale one."""
    scope = {**SCOPE, "project_ids": ["fresh", "stale"]}
    result = _coverage(
        rows=[],
        scope=scope,
        client=_Client(any_span=("fresh", "stale"), in_flight=("fresh",)),
    )
    assert result.complete is False
    assert result.reason == "project_unindexed"


@pytest.mark.unit
def test_late_arrival_below_the_floor_is_not_a_gap_until_it_settles():
    """An indexed project receiving yesterday's trace from a client buffer.

    By its own clock the span is below the floor the moment it lands; by
    arrival it is seconds old and still on its way to the index. Only once it
    has been in the source for the margin without being indexed is it a gap.
    """
    rows = [{"project_id": "p1", "floor": "2026-01-02 00:00:00"}]
    settling = _coverage(rows=rows, client=_Client(below=("p1",), in_flight=("p1",)))
    assert settling.complete is True
    assert settling.reason == "covered"

    settled = _coverage(rows=rows, client=_Client(below=("p1",)))
    assert settled.complete is False
    assert settled.reason == "source_predates_index"


@pytest.mark.unit
def test_both_probes_gate_on_arrival_not_the_spans_own_clock():
    """The gate is created_at (server-assigned at insert), on both statements.

    start_time is the producer's clock and is routinely days old for a span
    that arrived a second ago, so it cannot tell "in flight" from "missing".
    """
    # p2 is index-less but empty, so the index-less probe finds nothing and the
    # floor probe for p1 still runs: both statements are then on record.
    client = _Client(below=(), any_span=())
    scope = {**SCOPE, "project_ids": ["p1", "p2"]}
    _coverage(
        rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}],
        scope=scope,
        client=client,
    )
    gate = "created_at < now64(6, 'UTC') - INTERVAL 1 HOUR"
    assert len(client.calls) == 2
    unindexed, floor = client.calls[0]["sql"], client.calls[1]["sql"]
    assert "start_time" not in unindexed and gate in unindexed
    assert "start_time <" in floor and gate in floor


@pytest.mark.unit
def test_an_empty_project_does_not_mask_a_real_gap_elsewhere():
    """One empty project must not make a genuinely uncovered sibling look fine."""
    result = _coverage(
        rows=[{"project_id": "p2", "floor": "2026-06-01 00:00:00"}],
        scope=TWO_PROJECTS,
        client=_Client(below=("p2",), any_span=("p2",)),
    )
    assert result.complete is False
    assert result.reason == "source_predates_index"


@pytest.mark.unit
def test_one_uncovered_project_makes_the_whole_scope_partial():
    """A multi-project scope is only as complete as its worst project."""
    result = _coverage(
        rows=[
            {"project_id": "p1", "floor": "2026-01-01 00:00:00"},
            {"project_id": "p2", "floor": "2026-06-01 00:00:00"},
        ],
        below=("p2",),
        scope=TWO_PROJECTS,
    )
    assert result.complete is False
    assert result.reason == "source_predates_index"


@pytest.mark.unit
def test_empty_scope_is_genuinely_complete():
    """No projects means no history to miss, so the empty answer is honest."""
    result = _coverage(scope={**SCOPE, "project_ids": ()})
    assert result.complete is True
    assert result.reason == "empty_scope"


@pytest.mark.unit
@pytest.mark.parametrize(
    "observed,client,reason",
    [
        (_Observed(raises=RuntimeError("catalog down")), None, "floor_unavailable"),
        (None, _Client(raises=RuntimeError("source down")), "probe_unavailable"),
    ],
)
def test_a_failed_check_never_claims_completeness(observed, client, reason):
    """Fail closed: if we could not verify coverage, we do not assert it."""
    result = _coverage(
        rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}],
        observed=observed,
        client=client,
    )
    assert result.complete is False
    assert result.status == "partial"
    assert result.reason == reason


@pytest.mark.unit
def test_probe_is_pinned_to_bounded_read_settings():
    """The perf contract, not a style preference.

    Measured on a 2M-row production-shaped fixture, probing a project whose whole
    history sits below the floor: default parallelism reads 409,600 rows / 9.38
    MiB because LIMIT 1 only applies after concurrent granules land, while
    max_threads=1 + max_block_size=1024 reads 2,048 rows / 48 KiB. The
    un-backfilled state runs this probe on every request, so dropping these
    settings reintroduces the cost this feature exists to remove.
    """
    client = _Client(below=())
    _coverage(rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}], client=client)

    assert client.calls, "the source probe never ran"
    settings = client.calls[-1]["settings"]
    for key, value in _PROBE_SETTINGS.items():
        assert settings[key] == value, f"{key} must stay pinned to {value}"
    assert _PROBE_SETTINGS == {"max_threads": 1, "max_block_size": 1024}


@pytest.mark.unit
def test_probe_is_scoped_and_bounded_by_construction():
    """The probe must ask only about one project, below its own floor, LIMIT 1."""
    client = _Client(below=())
    _coverage(rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}], client=client)

    call = client.calls[-1]
    assert "LIMIT 1" in call["sql"]
    assert "project_id IN %(project_ids)s" in call["sql"]
    assert "INTERVAL 1 HOUR" in call["sql"]
    # Values are bound, never interpolated into SQL text.
    assert call["parameters"]["project_ids"] == ["p1"]
    assert call["parameters"]["floors"] == ["2026-01-01 00:00:00"]
    assert "p1" not in call["sql"]


@pytest.mark.unit
def test_tz_aware_floor_is_rendered_without_an_offset():
    """Regression: a tz-aware floor must not reach ClickHouse as "...+00:00".

    The catalog driver returns tz-aware datetimes, and their default string form
    carries a "+00:00" offset that DateTime64 refuses:
    ``Cannot convert string '2026-09-11 06:52:48.031961+00:00' to type
    DateTime64(6, 'UTC')``. That failure was invisible in unit tests, which fed
    plain strings, and invisible in production too, because coverage fails
    closed -- so a broken probe and an un-backfilled index looked identical.
    """
    from datetime import UTC, datetime

    client = _Client(below=())
    observed = _Observed(
        [{"project_id": "p1", "floor": datetime(2026, 1, 1, 12, 30, 45, 123456, tzinfo=UTC)}]
    )
    result = _coverage(observed=observed, client=client)

    floor = client.calls[-1]["parameters"]["floors"][0]
    assert floor == "2026-01-01 12:30:45.123456"
    assert "+00:00" not in floor and "T" not in floor and not floor.endswith("Z")
    # The surfaced floor is rendered the same way, so callers see one format.
    assert result.floor == "2026-01-01 12:30:45.123456"


@pytest.mark.unit
def test_probe_ignores_sub_floor_jitter_but_not_real_absence():
    """Regression: an exact floor comparison reported "partial" on healthy data.

    The floor is the earliest span carrying a catalog-eligible attribute, but
    the probe can only ask about spans, and spans without such attributes
    legitimately predate it. On a real stack the oldest span sat **one
    microsecond** before the floor, so every project reported partial and 11
    E2E flows that assert ``query_complete: true`` broke.

    Measured across 846 indexed projects: 831 had no gap, 15 had a sub-second
    gap, and none fell between 1 second and 1 hour -- so the margin separates
    jitter from absence without masking anything real.
    """
    from tracer.services.clickhouse.v2.property_catalog.coverage import (
        _COVERAGE_MARGIN,
    )

    client = _Client(below=())
    _coverage(rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}], client=client)

    # The margin is applied in SQL, so a span a microsecond below the floor can
    # no longer match; only data older than the margin can.
    assert _COVERAGE_MARGIN == "INTERVAL 1 HOUR"
    assert f"- {_COVERAGE_MARGIN}" in client.calls[-1]["sql"]


@pytest.mark.unit
def test_probe_cost_is_flat_in_scope_size():
    """Coverage must not cost one round trip per project.

    The first implementation looped per project: measured at 765 ms for one
    project, 2.0 s for five and 5.3 s for fifteen -- linear, on the interactive
    path this feature exists to make fast. Both probes are now set-based, so a
    scope of any size costs at most two queries.
    """
    many = tuple(f"p{i}" for i in range(40))
    client = _Client(below=())
    observed = _Observed(
        [{"project_id": p, "floor": "2026-01-01 00:00:00"} for p in many]
    )
    observed_scope_coverage(
        scope={**SCOPE, "project_ids": many},
        deadline=SimpleNamespace(remaining_ms=lambda floor_ms=1: 10_000),
        observed=observed,
        client=client,
    )

    assert len(client.calls) <= 2, f"{len(client.calls)} queries for 40 projects"
    assert len(client.calls[-1]["parameters"]["project_ids"]) == 40


@pytest.mark.unit
def test_unmappable_project_id_fails_closed_not_open():
    """A row whose id does not map back into the bound array is UNCOVERED.

    The probe locates each row's own floor with
    ``arrayElement(floors, indexOf(project_ids, toString(project_id)))``. On a
    miss ``indexOf`` returns 0, and ClickHouse's ``arrayElement(arr, 0)`` yields
    the type default -- an empty string -- whose comparison is false. Without the
    guard that silently reports the project COVERED, which is the single
    direction this check must never fail in.

    Verified against ClickHouse 25.3: indexOf(['a','b'],'missing') = 0,
    arrayElement(['x','y'],0) = '' (empty() = 1), and a `<` against it is false.
    """
    client = _Client(below=())
    _coverage(rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}], client=client)

    sql = client.calls[-1]["sql"]
    assert "indexOf(%(project_ids)s, toString(project_id)) = 0" in sql
    # The guard must be an OR arm of the predicate, so an unmappable row matches
    # and is reported as a gap rather than skipped -- and it must stay
    # unconditional: the arrival gate belongs to the predates-floor arm only,
    # never to the backstop, or an unmappable row that arrived recently would
    # be silently skipped.
    guard = sql.index("= 0")
    arm = sql.index("OR (", guard)
    assert "start_time <" in sql[arm:]
    assert "created_at <" not in sql[:arm]
    assert "created_at <" in sql[arm:]


@pytest.mark.unit
def test_probe_uses_the_guarded_read_path_that_preserves_settings():
    """The bounded settings must actually reach ClickHouse, not just be passed.

    Regression, and a lesson about doubles. An earlier revision called
    ``ClickHouseClient.execute()``, which sets ``settings = None``
    unconditionally on a server-readonly client -- so ``max_threads=1``,
    ``max_block_size=1024`` and the wall clock were silently discarded, and the
    200x read amplification they exist to prevent was reintroduced.

    The old test could not see it: it asserted the settings dict handed to a fake
    that faithfully recorded whatever it was given. The fake now refuses
    ``execute()`` outright and mirrors ``execute_read``'s real signature and
    triple return, so the same mistake fails here instead of in production.
    """
    client = _Client(below=())
    _coverage(rows=[{"project_id": "p1", "floor": "2026-01-01 00:00:00"}], client=client)

    call = client.calls[-1]
    assert call["settings"]["max_threads"] == 1
    assert call["settings"]["max_block_size"] == 1024
    # A wall clock must be supplied to the guarded path, not left unbounded.
    assert call["timeout_ms"] == _PROBE_WALL_MS and _PROBE_WALL_MS > 0


@pytest.mark.unit
def test_source_client_allows_settings_under_server_readonly():
    """The client must be built so its settings survive server-readonly mode."""
    import inspect

    from tracer.services.clickhouse.v2.property_catalog import coverage as mod

    src = inspect.getsource(mod._source_client)
    assert "allow_query_settings_with_server_readonly=True" in src, (
        "without this flag ClickHouseClient drops settings on a readonly client"
    )


@pytest.mark.unit
def test_floor_query_is_tenant_scoped():
    """Coverage must not be derived from another tenant's rows."""
    observed = _Observed([{"project_id": "p1", "floor": "2026-01-01 00:00:00"}])
    _coverage(observed=observed)

    sql, params, _ = observed.queries[0]
    assert "k.organization_id = %(organization_id)s" in sql
    assert "k.workspace_id = %(workspace_id)s" in sql
    assert "k.project_id IN %(project_ids)s" in sql
    assert params["organization_id"] == "org-1"
    assert params["workspace_id"] == "ws-1"


@pytest.mark.unit
def test_status_maps_to_the_frontend_degraded_contract():
    """`query_complete: false` with a non-"sampled" status renders as degraded.

    getQueryReadState (frontend/src/utils/queryReadState.js) treats that pair as
    degraded, so the caveat surfaces without a frontend change.
    """
    assert Coverage(False, "source_predates_index").status == "partial"
    assert Coverage(False, "source_predates_index").status != "sampled"
    assert Coverage(True, "covered").status == "complete"
