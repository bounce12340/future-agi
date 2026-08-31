"""
Tests for parsing_evaltask_filters dispatcher.

Verifies the eval-task path now routes each ``col_type`` through the right
FilterEngine handler — mirroring ``list_spans_observe``
(``tracer/views/observation_span.py:1755-1826``) — and silently ignores
unrecognised col_types. Structural assertions: we check the shape and
``repr`` of the returned ``Q`` (and the annotation dict) rather than
running queries against a DB, because the underlying handlers each have
their own integration tests.

The bug this guards against: until 2026-06-04 the dispatcher only handled
``SPAN_ATTRIBUTE`` and silently dropped every other col_type, including the
``ANNOTATION`` annotator filter behind the prod task
``f8481965-cd74-44b1-9b6d-f4bb66ec2218``. See plan
``~/.claude/plans/shift-to-fix-nightly-dev-27-05-and-zazzy-eagle.md``.

The two top-level id keys (``trace_id`` / ``span_id``) are the exception to
the structural rule: a ``repr`` cannot tell a key that was translated from one
that was silently dropped, so those are asserted on the rows the ``Q`` selects
and on the kwargs the ClickHouse companion produces.
"""

import uuid
from datetime import timedelta
from unittest import mock

import pytest
from django.db.models import OuterRef, Q
from django.utils import timezone

from tracer.models.eval_task import RowType
from tracer.models.observation_span import ObservationSpan
from tracer.models.trace import Trace
from tracer.services.clickhouse.v2.span_reader import CHSpanReader
from tracer.utils.eval_tasks import (
    annotation_source_q_for_row_type,
    parsing_evaltask_filters,
)
from tracer.utils.filters import FilterEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap(items, key="filters"):
    """Wrap a list of filter items in the eval-task top-level filters dict."""
    return {key: items}


def _span_attr_item(column_id="ended_reason", value="completed"):
    return {
        "column_id": column_id,
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": "text",
            "filter_op": "equals",
            "filter_value": value,
        },
    }


def _annotator_item(user_uuid="c65a0f3c-8a72-432a-987f-ddbd8391df29"):
    """Global `annotator` filter in the canonical snake_case contract.

    `normalize_filter_item` intentionally rejects camelCase aliases, so the
    filter must use `column_id`/`filter_config` to reach the annotator handler.
    """
    return {
        "column_id": "annotator",
        "filter_config": {
            "col_type": "ANNOTATION",
            "filter_op": "equals",
            "filter_type": "text",
            "filter_value": user_uuid,
        },
    }


def _label_value_item(label_uuid, value):
    return {
        "column_id": label_uuid,
        "filter_config": {
            "col_type": "ANNOTATION",
            "filter_type": "number",
            "filter_op": "greater_than",
            "filter_value": value,
        },
    }


def _trace(project, name):
    return Trace.objects.create(project=project, name=name)


def _span(project, trace, prefix="s"):
    return ObservationSpan.objects.create(
        id=f"{prefix}-{uuid.uuid4().hex[:12]}",
        project=project,
        trace=trace,
        name="span",
        observation_type="llm",
        start_time=timezone.now() - timedelta(seconds=5),
    )


class _RecordingClient:
    """Captures the SQL + params of the last query; returns no rows."""

    def __init__(self):
        self.sql = None
        self.params = None

    def query(self, sql, parameters=None, settings=None):
        self.sql = sql
        self.params = parameters or {}

        class _Result:
            result_rows = []

        return _Result()


def _reader_with(client) -> CHSpanReader:
    # Bypass __init__ (which opens a real CH connection) and inject the fake.
    reader = CHSpanReader.__new__(CHSpanReader)
    reader._client = client
    return reader


def _system_metric_item(column_id="cost", op="greater_than", value=0.5):
    return {
        "column_id": column_id,
        "filter_config": {
            "col_type": "SYSTEM_METRIC",
            "filter_type": "number",
            "filter_op": op,
            "filter_value": value,
        },
    }


def _eval_metric_item(eval_template_id, op="greater_than", value=0.8):
    return {
        "column_id": eval_template_id,
        "filter_config": {
            "col_type": "EVAL_METRIC",
            "filter_type": "number",
            "filter_op": op,
            "filter_value": value,
        },
    }


# ---------------------------------------------------------------------------
# Empty / None handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmpty:
    def test_none_returns_empty_tuple(self):
        q, anns = parsing_evaltask_filters(None, row_type=RowType.SPANS)
        assert q == Q()
        assert anns == {}

    def test_empty_dict_returns_empty_tuple(self):
        q, anns = parsing_evaltask_filters({}, row_type=RowType.SPANS)
        assert q == Q()
        assert anns == {}

    def test_empty_filters_list_returns_empty_tuple(self):
        q, anns = parsing_evaltask_filters({"filters": []}, row_type=RowType.SPANS)
        assert q == Q()
        assert anns == {}


# ---------------------------------------------------------------------------
# Legacy `span_attributes_filters` key still parsed (transition)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyKey:
    def test_legacy_key_with_span_attribute_filter(self):
        items = [_span_attr_item()]
        q, anns = parsing_evaltask_filters(
            _wrap(items, key="span_attributes_filters"), row_type=RowType.SPANS
        )
        # Legacy key produces the same Q as the canonical key.
        q_canonical, _ = parsing_evaltask_filters(
            _wrap(items, key="filters"), row_type=RowType.SPANS
        )
        assert repr(q) == repr(q_canonical)
        assert anns == {}

    def test_legacy_key_with_annotator_filter(self):
        # The specific shape that motivated the fix: prod task f8481965.
        items = [_annotator_item()]
        q, anns = parsing_evaltask_filters(
            _wrap(items, key="span_attributes_filters"), row_type=RowType.SPANS
        )
        assert q != Q()
        assert "Exists" in repr(q)  # routed through the voice annotation handler


# ---------------------------------------------------------------------------
# SPAN_ATTRIBUTE — regression guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpanAttribute:
    def test_span_attribute_only(self):
        q, anns = parsing_evaltask_filters(
            _wrap([_span_attr_item()]), row_type=RowType.SPANS
        )
        assert q != Q()
        # SPAN_ATTRIBUTE filters resolve via the span_attributes JSONB
        # (`has_key` + `contains`) — verify by string match on the Q repr.
        assert "span_attributes" in repr(q)
        assert anns == {}


# ---------------------------------------------------------------------------
# ANNOTATION — the original bug
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnnotation:
    def test_annotator_filter_produces_exists_clause(self):
        items = [_annotator_item()]
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        assert q != Q()
        # The annotator branch builds an Exists(Score.objects...) subquery.
        assert "Exists" in repr(q)
        # The annotator path is pure-Q — no annotate() kwargs.
        assert anns == {}

    def test_per_label_score_filter_builds_annotation_field_q(self):
        # A per-label ANNOTATION filter (e.g. "quality > 3") builds a Q
        # referencing the `annotation_<uuid>__score` field. The actual
        # `.annotate(annotation_<uuid>=...)` step is contributed by
        # `build_annotation_subqueries`, which `process_eval_task` applies
        # with the row_type's source Q. So `extra_anns` remains empty here;
        # this test guards the Q shape only — row-level behaviour lives in
        # the integration suite (test_eval_task_annotation_scope.py).
        label_uuid = str(uuid.uuid4())
        items = [_label_value_item(label_uuid, 3.0)]
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        assert q != Q()
        assert f"annotation_{label_uuid}__score" in repr(q)
        assert anns == {}

    def test_annotation_does_not_leak_into_non_system_handler(self):
        # Per list_spans_observe (observation_span.py:1769-1777), ANNOTATION
        # items and the annotation-special column_ids must be excluded from
        # the eval-metrics handler. Mixing both should still produce a single
        # well-formed Q without raising.
        items = [
            _annotator_item(),
            _eval_metric_item(str(uuid.uuid4())),
        ]
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        assert q != Q()


# ---------------------------------------------------------------------------
# SYSTEM_METRIC
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSystemMetric:
    def test_system_metric_filter(self):
        items = [_system_metric_item("cost", "greater_than", 0.1)]
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        # The system-metrics handler maps `cost` to `row_avg_cost`
        # (FilterEngine.DEFAULT_FIELD_MAP) and emits a Q referencing it.
        assert q != Q()
        assert anns == {}


# ---------------------------------------------------------------------------
# EVAL_METRIC
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvalMetric:
    def test_eval_metric_filter(self):
        items = [_eval_metric_item(str(uuid.uuid4()), "greater_than", 0.8)]
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        assert q != Q()


# ---------------------------------------------------------------------------
# Mixed — all four colTypes at once
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMixed:
    def test_all_four_col_types_combine(self):
        items = [
            _span_attr_item(),
            _system_metric_item("cost", "greater_than", 0.1),
            _eval_metric_item(str(uuid.uuid4())),
            _annotator_item(),
        ]
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        assert q != Q()
        # Each handler contributes its own clause; the combined Q should
        # reference all four underlying mechanisms.
        rep = repr(q)
        assert "span_attributes" in rep       # SPAN_ATTRIBUTE
        assert "row_avg_cost" in rep          # SYSTEM_METRIC (cost)
        assert "__score__gt" in rep           # EVAL_METRIC (per-eval score)
        assert "Exists" in rep                # ANNOTATION (annotator) Exists


# ---------------------------------------------------------------------------
# Sibling keys at the top level
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSiblingKeys:
    def test_date_range_still_applied(self):
        q, _ = parsing_evaltask_filters(
            {"date_range": ["2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"]},
            row_type=RowType.SPANS,
        )
        assert q != Q()
        assert "created_at" in repr(q)

    def test_project_id_still_applied(self):
        q, _ = parsing_evaltask_filters(
            {"project_id": str(uuid.uuid4())}, row_type=RowType.SPANS
        )
        assert q != Q()
        assert "project_id" in repr(q)

    def test_observation_type_still_applied(self):
        q, _ = parsing_evaltask_filters(
            {"observation_type": ["llm", "tool"]}, row_type=RowType.SPANS
        )
        assert q != Q()
        assert "observation_type" in repr(q)


# ---------------------------------------------------------------------------
# RowType → annotation source Q mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRowTypeSourceMapping:
    def test_spans_is_span_scoped(self):
        assert annotation_source_q_for_row_type(RowType.SPANS) == Q(
            observation_span_id=OuterRef("id")
        )

    def test_voice_calls_bridges_trace_and_span_scopes(self):
        expected = Q(trace_id=OuterRef("trace_id")) | Q(
            observation_span__trace_id=OuterRef("trace_id")
        )
        assert annotation_source_q_for_row_type(RowType.VOICE_CALLS) == expected

    def test_traces_matches_voice_calls_mapping(self):
        assert annotation_source_q_for_row_type(
            RowType.TRACES
        ) == annotation_source_q_for_row_type(RowType.VOICE_CALLS)

    def test_sessions_is_session_scoped(self):
        assert annotation_source_q_for_row_type(RowType.SESSIONS) == Q(
            trace_session_id=OuterRef("trace__session_id")
        )

    def test_accepts_stored_string_values(self):
        assert annotation_source_q_for_row_type(
            "voiceCalls"
        ) == annotation_source_q_for_row_type(RowType.VOICE_CALLS)

    def test_unknown_row_type_raises(self):
        with pytest.raises(ValueError):
            annotation_source_q_for_row_type("bogus")


# ---------------------------------------------------------------------------
# Dispatcher threads the row_type source Q into the annotation handlers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRowTypeDispatch:
    @pytest.mark.parametrize(
        "row_type",
        [RowType.SPANS, RowType.TRACES, RowType.SESSIONS, RowType.VOICE_CALLS],
    )
    def test_annotation_handlers_receive_row_type_source_q(self, row_type):
        items = [
            _annotator_item(),
            {
                "column_id": "has_annotation",
                "filter_config": {
                    "filter_type": "boolean",
                    "filter_op": "equals",
                    "filter_value": True,
                },
            },
        ]
        with mock.patch.object(
            FilterEngine,
            "get_filter_conditions_for_voice_call_annotations",
            return_value=(Q(), {}),
        ) as anno_mock, mock.patch.object(
            FilterEngine,
            "get_filter_conditions_for_has_annotation",
            return_value=Q(),
        ) as has_anno_mock:
            parsing_evaltask_filters(_wrap(items), row_type=row_type)

        expected_source_q = annotation_source_q_for_row_type(row_type)
        assert anno_mock.call_args.kwargs["source_q"] == expected_source_q
        assert anno_mock.call_args.kwargs["user_id"] is None
        assert has_anno_mock.call_args.kwargs["source_q"] == expected_source_q

    @pytest.mark.parametrize(
        "row_type",
        [RowType.SPANS, RowType.TRACES, RowType.SESSIONS, RowType.VOICE_CALLS],
    )
    def test_has_eval_stays_span_scoped(self, row_type):
        items = [
            {
                "column_id": "has_eval",
                "filter_config": {
                    "filter_type": "boolean",
                    "filter_op": "equals",
                    "filter_value": True,
                },
            }
        ]
        with mock.patch.object(
            FilterEngine,
            "get_filter_conditions_for_has_eval",
            return_value=Q(),
        ) as has_eval_mock:
            parsing_evaltask_filters(_wrap(items), row_type=row_type)

        assert has_eval_mock.call_args.kwargs["observe_type"] == "span"


# ---------------------------------------------------------------------------
# Unrecognised col_type — silent skip (no raise)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnrecognisedColType:
    def test_unknown_col_type_does_not_raise(self):
        items = [
            {
                "column_id": "some_id",
                "filter_config": {
                    "col_type": "TOTALLY_MADE_UP",
                    "filter_type": "text",
                    "filter_op": "equals",
                    "filter_value": "x",
                },
            }
        ]
        # Must not raise. The handlers' own col_type gates skip items they
        # don't recognise; the dispatcher itself doesn't enumerate types.
        q, anns = parsing_evaltask_filters(_wrap(items), row_type=RowType.SPANS)
        assert isinstance(q, Q)
        assert anns == {}


# ---------------------------------------------------------------------------
# trace_id / span_id — the rows the Q actually selects
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.django_db
class TestLinkedSourceIdRows:
    """The behavioural half of the two top-level id keys: the Q they build must
    narrow the queryset to the linked rows, and an id matching nothing must
    select no rows rather than the whole table."""

    @staticmethod
    def _ids(filters):
        q, _ = parsing_evaltask_filters(filters, row_type=RowType.SPANS)
        return set(ObservationSpan.objects.filter(q).values_list("id", flat=True))

    def test_trace_id_selects_only_that_traces_spans(self, project):
        picked = _trace(project, "picked")
        wanted = [_span(project, picked), _span(project, picked)]
        _span(project, _trace(project, "rest"))
        assert self._ids({"trace_id": [str(picked.id)]}) == {s.id for s in wanted}

    def test_span_id_selects_only_that_span(self, project):
        trace = _trace(project, "t")
        picked = _span(project, trace)
        _span(project, trace)
        assert self._ids({"span_id": [picked.id]}) == {picked.id}

    def test_scalar_trace_id_selects_that_traces_spans(self, project):
        picked = _trace(project, "picked")
        wanted = _span(project, picked)
        _span(project, _trace(project, "rest"))
        assert self._ids({"trace_id": str(picked.id)}) == {wanted.id}

    def test_unknown_trace_id_selects_no_rows(self, project):
        _span(project, _trace(project, "t"))
        assert self._ids({"trace_id": [str(uuid.uuid4())]}) == set()

    def test_unknown_span_id_selects_no_rows(self, project):
        _span(project, _trace(project, "t"))
        assert self._ids({"span_id": ["absent-span-id"]}) == set()

    def test_empty_trace_id_list_selects_no_rows(self, project):
        _span(project, _trace(project, "t"))
        assert self._ids({"trace_id": []}) == set()

    def test_empty_span_id_list_selects_no_rows(self, project):
        _span(project, _trace(project, "t"))
        assert self._ids({"span_id": []}) == set()

    def test_null_id_keys_leave_the_queryset_unnarrowed(self, project):
        span = _span(project, _trace(project, "t"))
        selected = self._ids(
            {"project_id": str(project.id), "trace_id": None, "span_id": None}
        )
        assert selected == {span.id}


# ---------------------------------------------------------------------------
# ClickHouse companion — the same two keys as reader kwargs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClickHouseIdKwargs:
    def test_trace_id_and_span_id_become_id_kwargs(self):
        out = CHSpanReader.parsing_evaltask_filters_for_ch(
            {"trace_id": ["t-1", "t-2"], "span_id": ["s-1"]}
        )
        assert out["trace_ids"] == ["t-1", "t-2"]
        assert out["span_ids"] == ["s-1"]

    def test_scalar_ids_are_normalised_to_lists(self):
        out = CHSpanReader.parsing_evaltask_filters_for_ch(
            {"trace_id": "t-1", "span_id": "s-1"}
        )
        assert out["trace_ids"] == ["t-1"]
        assert out["span_ids"] == ["s-1"]

    def test_id_kwargs_follow_key_presence_not_truthiness(self):
        # An emptied id list is "match nothing", so it has to reach the reader;
        # an absent key is "no constraint" and must not.
        emptied = CHSpanReader.parsing_evaltask_filters_for_ch(
            {"trace_id": [], "span_id": []}
        )
        assert emptied["trace_ids"] == []
        assert emptied["span_ids"] == []

        absent = CHSpanReader.parsing_evaltask_filters_for_ch(
            {"observation_type": ["llm"]}
        )
        assert "trace_ids" not in absent
        assert "span_ids" not in absent

    def test_null_id_keys_carry_no_constraint(self):
        out = CHSpanReader.parsing_evaltask_filters_for_ch(
            {"trace_id": None, "span_id": None}
        )
        assert "trace_ids" not in out
        assert "span_ids" not in out


@pytest.mark.unit
class TestClickHouseSpanIdPredicate:
    """``span_ids`` mirrors the ``trace_ids`` contract on both readers the
    eval-task filter kwargs feed: None is no filter, [] is match-nothing."""

    def test_session_ids_binds_the_span_id_predicate(self):
        client = _RecordingClient()
        _reader_with(client).distinct_session_ids_with_filters(
            project_id="p", span_ids=["s-1", "s-2"]
        )
        assert "id IN %(spids)s" in client.sql
        assert client.params["spids"] == ("s-1", "s-2")

    def test_session_ids_empty_span_ids_returns_nothing_without_querying(self):
        client = _RecordingClient()
        result = _reader_with(client).distinct_session_ids_with_filters(
            project_id="p", span_ids=[]
        )
        assert result == []
        assert client.sql is None

    def test_session_ids_none_span_ids_adds_no_predicate(self):
        client = _RecordingClient()
        _reader_with(client).distinct_session_ids_with_filters(
            project_id="p", span_ids=None
        )
        assert "%(spids)s" not in client.sql
        assert "spids" not in client.params

    def test_count_binds_the_span_id_predicate(self):
        client = _RecordingClient()
        _reader_with(client).count_with_filters(project_id="p", span_ids=["s-1", "s-2"])
        assert "id IN %(spids)s" in client.sql
        assert client.params["spids"] == ("s-1", "s-2")

    def test_count_empty_span_ids_returns_zero_without_querying(self):
        client = _RecordingClient()
        assert _reader_with(client).count_with_filters(project_id="p", span_ids=[]) == 0
        assert client.sql is None

    def test_count_none_span_ids_adds_no_predicate(self):
        client = _RecordingClient()
        _reader_with(client).count_with_filters(project_id="p", span_ids=None)
        assert "%(spids)s" not in client.sql
        assert "spids" not in client.params
