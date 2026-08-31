"""Tests for the deterministic eligible-rows resolver (the "did the row set
change?" axis). Exercises iter_desired_rows end-to-end against CH-seeded spans:
deterministic hash sampling, the fixed order/limit, batched streaming, per-
row_type id resolution, project scoping, and the reused v2 filter compiler."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from tracer.models.eval_task import EvalTask, EvalTaskStatus, RowType, RunType
from tracer.models.observation_span import ObservationSpan
from tracer.models.trace import Trace
from tracer.models.trace_session import TraceSession
from tracer.selectors.eval_tasks.row_resolver import iter_desired_rows
from tracer.tests._ch_seed import seed_ch_spans


def _ids(task, **kwargs):
    """Collect the streamed batches into one ordered list for assertions."""
    return [row_id for batch in iter_desired_rows(task, **kwargs) for row_id in batch]


def _make_task(
    project,
    *,
    row_type=RowType.SPANS,
    sampling_rate=100.0,
    spans_limit=1_000_000,
    filters=None,
    run_type=RunType.HISTORICAL,
):
    return EvalTask.objects.create(
        project=project,
        name="resolver-task",
        filters=filters or {},
        sampling_rate=sampling_rate,
        spans_limit=spans_limit,
        run_type=run_type,
        status=EvalTaskStatus.PENDING,
        row_type=row_type,
    )


def _make_spans(
    project,
    n,
    *,
    observation_type="llm",
    session=None,
    shared_trace=None,
    span_attributes=None,
    prefix="s",
):
    spans = []
    for i in range(n):
        trace = shared_trace or Trace.objects.create(
            project=project,
            name=f"trace-{prefix}-{i}",
            session=session,
        )
        spans.append(
            ObservationSpan.objects.create(
                id=f"{prefix}-{i}-{uuid.uuid4().hex[:8]}",
                project=project,
                trace=trace,
                name=f"span-{prefix}-{i}",
                observation_type=observation_type,
                span_attributes=span_attributes or {},
                start_time=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
    # created_at is auto_now_add; override the in-memory value the CH seed reads
    # so rows land safely inside the builders' default time window.
    for s in spans:
        s.created_at = datetime.now(UTC) - timedelta(minutes=1)
    seed_ch_spans(spans)
    return spans


def _make_child_span(project, trace, parent, *, prefix="child"):
    """A non-root span under ``parent``. The trace list scans root spans only,
    so a child is the shape that tells a root-only predicate apart from one
    that matches any span of the trace."""
    span = ObservationSpan.objects.create(
        id=f"{prefix}-{uuid.uuid4().hex[:8]}",
        project=project,
        trace=trace,
        parent_span_id=parent.id,
        name=f"span-{prefix}",
        observation_type="llm",
        start_time=datetime.now(UTC) - timedelta(minutes=1),
    )
    span.created_at = datetime.now(UTC) - timedelta(minutes=1)
    seed_ch_spans([span])
    return span


@pytest.mark.integration
@pytest.mark.django_db
class TestSamplingAndDeterminism:
    def test_rate_100_returns_all_spans(self, project):
        spans = _make_spans(project, 12)
        task = _make_task(project, sampling_rate=100.0)
        assert set(_ids(task)) == {s.id for s in spans}

    def test_rate_0_returns_none(self, project):
        _make_spans(project, 12)
        task = _make_task(project, sampling_rate=0.0)
        assert _ids(task) == []

    def test_deterministic_same_rows_and_order_on_repeat(self, project):
        _make_spans(project, 20)
        task = _make_task(project, sampling_rate=60.0)
        assert _ids(task) == _ids(task)

    def test_rate_down_is_strict_subset_up_is_superset(self, project):
        _make_spans(project, 40)
        task = _make_task(project, sampling_rate=100.0)
        full = set(_ids(task))
        task.sampling_rate = 50.0
        task.save()
        half = set(_ids(task))
        task.sampling_rate = 25.0
        task.save()
        quarter = set(_ids(task))
        assert quarter <= half <= full
        assert len(quarter) < len(full)  # 40 rows: sampling actually narrows


@pytest.mark.integration
@pytest.mark.django_db
class TestOrderLimitAndBatching:
    def test_limit_returns_deterministic_prefix(self, project):
        _make_spans(project, 30)
        task = _make_task(project, sampling_rate=100.0, spans_limit=1_000_000)
        full = _ids(task)
        task.spans_limit = 5
        task.save()
        limited = _ids(task)
        assert len(limited) == 5
        assert limited == full[:5]

    def test_batches_chunk_to_batch_size_and_preserve_order(self, project):
        _make_spans(project, 25)
        task = _make_task(project, sampling_rate=100.0)
        batches = list(iter_desired_rows(task, batch_size=10))
        assert [len(b) for b in batches] == [10, 10, 5]
        flat = [rid for b in batches for rid in b]
        assert flat == _ids(task)  # streamed order == single-shot order

    def test_batch_size_one_yields_one_row_per_batch(self, project):
        spans = _make_spans(project, 3)
        task = _make_task(project, sampling_rate=100.0)
        batches = list(iter_desired_rows(task, batch_size=1))
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)
        assert {b[0] for b in batches} == {s.id for s in spans}


@pytest.mark.integration
@pytest.mark.django_db
class TestRowTypes:
    def test_spans_returns_span_ids(self, project):
        spans = _make_spans(project, 6)
        task = _make_task(project, row_type=RowType.SPANS)
        assert set(_ids(task)) == {s.id for s in spans}

    def test_traces_returns_distinct_trace_ids(self, project):
        trace_ids = set()
        for t in range(3):
            trace = Trace.objects.create(project=project, name=f"t-{t}")
            trace_ids.add(str(trace.id))
            _make_spans(project, 4, shared_trace=trace, prefix=f"t{t}")
        task = _make_task(project, row_type=RowType.TRACES)
        assert set(_ids(task)) == trace_ids

    def test_sessions_returns_distinct_session_ids(self, project):
        session_ids = set()
        for s in range(3):
            session = TraceSession.objects.create(project=project, name=f"sess-{s}")
            session_ids.add(str(session.id))
            _make_spans(project, 2, session=session, prefix=f"sess{s}")
        task = _make_task(project, row_type=RowType.SESSIONS)
        assert set(_ids(task)) == session_ids


@pytest.mark.integration
@pytest.mark.django_db
class TestScopingAndFilters:
    def test_project_pinning_excludes_other_projects(self, project, observe_project):
        mine = _make_spans(project, 5, prefix="mine")
        _make_spans(observe_project, 5, prefix="other")
        task = _make_task(project)
        assert set(_ids(task)) == {s.id for s in mine}

    def test_observation_type_top_level_filter(self, project):
        llm = _make_spans(project, 4, observation_type="llm", prefix="llm")
        _make_spans(project, 4, observation_type="tool", prefix="tool")
        task = _make_task(project, filters={"observation_type": ["llm"]})
        assert set(_ids(task)) == {s.id for s in llm}

    def test_span_attribute_filter_compiles_via_v2_builder(self, project):
        match = _make_spans(
            project, 1, span_attributes={"function.name": "execute_query"}, prefix="m"
        )
        _make_spans(project, 1, span_attributes={"function.name": "other"}, prefix="n")
        task = _make_task(
            project,
            filters={
                "filters": [
                    {
                        "column_id": "function.name",
                        "filter_config": {
                            "col_type": "SPAN_ATTRIBUTE",
                            "filter_op": "in",
                            "filter_type": "text",
                            "filter_value": ["execute_query"],
                        },
                    }
                ]
            },
        )
        assert set(_ids(task)) == {match[0].id}

    def test_empty_when_no_rows(self, project):
        task = _make_task(project)
        assert _ids(task) == []

    def test_unsupported_row_type_raises(self, project):
        # Defensive: a row_type outside the enum fails fast, not silently.
        task = _make_task(project)
        task.row_type = "galaxies"
        with pytest.raises(ValueError):
            list(iter_desired_rows(task))

    def test_empty_observation_type_is_no_constraint(self, project):
        # An empty observation_type list is treated as "no filter" (returns
        # all), matching the existing count_with_filters semantics — not as
        # "match nothing".
        spans = _make_spans(project, 5)
        task = _make_task(project, filters={"observation_type": []})
        assert set(_ids(task)) == {s.id for s in spans}


@pytest.mark.integration
@pytest.mark.django_db
class TestLinkedSourceIdFilters:
    """``trace_id`` / ``span_id`` are stored top-level on a task's filters and
    must scope the resolved row set to those ids, for every row_type. An id
    that matches nothing resolves to no rows — never to the whole project."""

    def test_spans_scoped_to_one_trace(self, project):
        picked = Trace.objects.create(project=project, name="picked")
        wanted = _make_spans(project, 3, shared_trace=picked, prefix="picked")
        _make_spans(project, 4, prefix="rest")
        task = _make_task(project, filters={"trace_id": [str(picked.id)]})
        assert set(_ids(task)) == {s.id for s in wanted}

    def test_spans_scoped_to_one_span(self, project):
        spans = _make_spans(project, 5)
        task = _make_task(project, filters={"span_id": [spans[2].id]})
        assert _ids(task) == [spans[2].id]

    def test_traces_scoped_to_one_trace(self, project):
        picked = Trace.objects.create(project=project, name="picked")
        _make_spans(project, 2, shared_trace=picked, prefix="picked")
        _make_spans(project, 3, prefix="rest")
        task = _make_task(
            project, row_type=RowType.TRACES, filters={"trace_id": [str(picked.id)]}
        )
        assert _ids(task) == [str(picked.id)]

    def test_traces_scoped_by_a_non_root_span(self, project):
        # The span_id predicate matches any span of the trace, unlike the
        # root-only observation_type one.
        picked = Trace.objects.create(project=project, name="picked")
        root = _make_spans(project, 1, shared_trace=picked, prefix="root")[0]
        child = _make_child_span(project, picked, root)
        _make_spans(project, 3, prefix="rest")
        task = _make_task(
            project, row_type=RowType.TRACES, filters={"span_id": [child.id]}
        )
        assert _ids(task) == [str(picked.id)]

    def test_sessions_scoped_to_the_traces_session(self, project):
        picked_session = TraceSession.objects.create(project=project, name="picked")
        other_session = TraceSession.objects.create(project=project, name="other")
        picked_trace = Trace.objects.create(
            project=project, name="picked", session=picked_session
        )
        _make_spans(project, 2, shared_trace=picked_trace, prefix="picked")
        _make_spans(project, 2, session=other_session, prefix="other")
        task = _make_task(
            project,
            row_type=RowType.SESSIONS,
            filters={"trace_id": [str(picked_trace.id)]},
        )
        assert _ids(task) == [str(picked_session.id)]

    def test_unknown_trace_id_selects_no_rows(self, project):
        _make_spans(project, 6)
        task = _make_task(project, filters={"trace_id": [str(uuid.uuid4())]})
        assert _ids(task) == []

    def test_unknown_span_id_selects_no_rows(self, project):
        _make_spans(project, 6)
        task = _make_task(project, filters={"span_id": ["absent-span-id"]})
        assert _ids(task) == []

    def test_empty_trace_id_list_matches_nothing(self, project):
        # Contrast with test_empty_observation_type_is_no_constraint: an id list
        # the user emptied scopes the task to nothing rather than widening back
        # to the project.
        _make_spans(project, 4)
        task = _make_task(project, filters={"trace_id": []})
        assert _ids(task) == []

    def test_empty_span_id_list_matches_nothing(self, project):
        _make_spans(project, 4)
        task = _make_task(project, filters={"span_id": []})
        assert _ids(task) == []

    def test_null_id_keys_are_no_constraint(self, project):
        # The filters field skips a null id key rather than validating it, so a
        # stored null means "not set" — not the emptied list above.
        spans = _make_spans(project, 4)
        task = _make_task(project, filters={"trace_id": None, "span_id": None})
        assert set(_ids(task)) == {s.id for s in spans}

    def test_trace_id_and_observation_type_intersect(self, project):
        picked = Trace.objects.create(project=project, name="picked")
        llm = _make_spans(
            project, 2, shared_trace=picked, observation_type="llm", prefix="llm"
        )
        _make_spans(
            project, 2, shared_trace=picked, observation_type="tool", prefix="tool"
        )
        _make_spans(project, 2, observation_type="llm", prefix="elsewhere")
        task = _make_task(
            project,
            filters={"trace_id": [str(picked.id)], "observation_type": ["llm"]},
        )
        assert set(_ids(task)) == {s.id for s in llm}

    def test_scalar_trace_id_scopes_like_a_one_item_list(self, project):
        # Rows stored before the filters field coerced scalars still hold a
        # bare string.
        picked = Trace.objects.create(project=project, name="picked")
        wanted = _make_spans(project, 2, shared_trace=picked, prefix="picked")
        _make_spans(project, 3, prefix="rest")
        task = _make_task(project, filters={"trace_id": str(picked.id)})
        assert set(_ids(task)) == {s.id for s in wanted}

    def test_trace_id_of_another_project_selects_no_rows(
        self, project, observe_project
    ):
        theirs = Trace.objects.create(project=observe_project, name="theirs")
        _make_spans(observe_project, 3, shared_trace=theirs, prefix="theirs")
        _make_spans(project, 3, prefix="mine")
        task = _make_task(project, filters={"trace_id": [str(theirs.id)]})
        assert _ids(task) == []


def _builder_ids(builder):
    """Run a UI builder's build_id_query against CH and return its id set."""
    from tracer.services.clickhouse.v2 import get_reader

    sql, params = builder.build_id_query()
    reader = get_reader()
    try:
        rows = reader._client.query(sql, parameters=params).result_rows
    finally:
        reader.close()
    return {str(r[0]) for r in rows}


@pytest.mark.integration
@pytest.mark.django_db
class TestApiFilterParity:
    """The resolver must select the same rows the UI list endpoints return for
    the same filters (the 'same as API' guarantee)."""

    def test_span_set_equals_list_builder_with_date_filter(self, project):
        from tracer.services.clickhouse.v2.dispatch import get_v2_class

        spans = _make_spans(project, 8)
        now = datetime.now(UTC)
        rng = [
            (now - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
            (now + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        ]
        task = _make_task(project, filters={"date_range": rng}, sampling_rate=100.0)
        resolver_ids = set(_ids(task))

        ui_filters = [
            {
                "column_id": "created_at",
                "filter_config": {
                    "filter_type": "datetime",
                    "filter_op": "between",
                    "filter_value": rng,
                },
            }
        ]
        ui_ids = _builder_ids(
            get_v2_class("SPAN_LIST")(project_id=str(project.id), filters=ui_filters)
        )

        assert resolver_ids == ui_ids
        assert resolver_ids == {s.id for s in spans}

    def test_past_only_date_filter_excludes_recent_spans(self, project):
        _make_spans(project, 4)
        rng = ["2000-01-01T00:00:00.000Z", "2000-01-02T00:00:00.000Z"]
        task = _make_task(project, filters={"date_range": rng}, sampling_rate=100.0)
        assert _ids(task) == []

    def test_voicecalls_select_only_conversation_roots(self, project):
        calls = _make_spans(project, 3, observation_type="conversation", prefix="call")
        _make_spans(project, 4, observation_type="llm", prefix="llm")
        task = _make_task(project, row_type=RowType.VOICE_CALLS, sampling_rate=100.0)
        assert set(_ids(task)) == {c.id for c in calls}
