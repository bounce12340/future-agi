"""An empty bounded checkpoint must not be published as a complete answer."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest import mock

from tracer.selectors.trace_filter_reads import BoundedFilterPage
from tracer.services.clickhouse.list_cursor import bounded_chunk_complete

PROJECT_ID = "00000000-0000-4000-8000-000000000001"
WINDOW_END = datetime(2026, 1, 31)
WINDOW_START = WINDOW_END - timedelta(days=30)
CHECKPOINT = WINDOW_END - timedelta(days=4)


def _time_filter() -> dict[str, Any]:
    return {
        "column_id": "created_at",
        "filter_config": {
            "filter_type": "datetime",
            "filter_op": "between",
            "filter_value": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        },
    }


def _absent_attribute_filter() -> dict[str, Any]:
    """One negative leaf on a scalar span attribute, as the defect shape uses."""

    return {
        "column_id": "outcome_label",
        "filter_config": {
            "col_type": "SPAN_ATTRIBUTE",
            "filter_type": "text",
            "filter_op": "is_null",
        },
    }


def _checkpointed_page(
    *,
    rows: list[dict[str, Any]],
    before_id: Any,
) -> BoundedFilterPage:
    """A budget-exceeded read that covered only a prefix of the window."""

    return BoundedFilterPage(
        rows=list(rows),
        has_more=False,
        complete=False,
        status="degraded",
        error_code="scan_budget_exceeded",
        total_rows_lower_bound=len(rows),
        elapsed_ms=4_011.0,
        query_count=46,
        rows_returned=len(rows),
        result_payload_bytes=0,
        attempts=(),
        continuation_slice_start=CHECKPOINT,
        continuation_slice_end=CHECKPOINT + timedelta(hours=8),
        continuation_before_start_time=None if before_id is None else CHECKPOINT,
        continuation_before_id=before_id,
    )


def test_bounded_chunk_complete_publishes_only_a_proven_chunk() -> None:
    assert (
        bounded_chunk_complete(
            read_complete=True, cursor_has_more=False, published_rows=0
        )
        is True
    )
    assert (
        bounded_chunk_complete(
            read_complete=False, cursor_has_more=True, published_rows=1
        )
        is True
    )
    assert (
        bounded_chunk_complete(
            read_complete=False, cursor_has_more=True, published_rows=0
        )
        is False
    )
    assert (
        bounded_chunk_complete(
            read_complete=False, cursor_has_more=False, published_rows=1
        )
        is False
    )


def _read_span_page(bounded: BoundedFilterPage) -> dict[str, Any]:
    from tracer.views.observation_span import ObservationSpanView

    view = ObservationSpanView.__new__(ObservationSpanView)
    view._gm = SimpleNamespace(
        success_response=lambda payload: ("ok", payload),
        custom_error_response=lambda *args, **kwargs: ("error", args, kwargs),
    )
    organization = SimpleNamespace(id=uuid.uuid4())
    request = SimpleNamespace(
        query_params={},
        organization=organization,
        user=SimpleNamespace(organization=organization),
    )
    analytics = mock.MagicMock()
    analytics.execute_ch_query.side_effect = lambda *_args, **_kwargs: SimpleNamespace(
        data=[]
    )

    with (
        mock.patch("tracer.views.observation_span.CustomEvalConfig") as eval_config,
        mock.patch(
            "tracer.views.observation_span.get_annotation_labels_for_project",
            return_value=[],
        ),
        mock.patch(
            "tracer.selectors.trace_filter_reads.read_bounded_filter_page",
            return_value=bounded,
        ),
    ):
        eval_config.objects.filter.return_value.select_related.return_value = []
        outcome = view._list_spans_clickhouse(
            request,
            project_id=PROJECT_ID,
            validated_data={
                "filters": [_time_filter(), _absent_attribute_filter()],
                "page_number": 0,
                "page_size": 25,
                "cursor_mode": True,
            },
            analytics=analytics,
            org_project_ids=None,
            org=organization,
        )

    assert outcome[0] == "ok", outcome
    return outcome[1]


def test_empty_span_checkpoint_reports_the_degraded_read_it_actually_made() -> None:
    payload = _read_span_page(_checkpointed_page(rows=[], before_id=None))

    assert payload["table"] == []
    metadata = payload["metadata"]
    assert metadata["query_complete"] is False
    assert metadata["query_status"] == "degraded"
    assert metadata["query_error_code"] == "scan_budget_exceeded"
    # The caller must still be able to resume the unfinished scan.
    assert metadata["has_more"] is True
    assert isinstance(metadata["next_cursor"], str)
    assert metadata["total_rows_is_lower_bound"] is True


def _session_builder_class(instance: Any) -> Any:
    """Stub construction while retaining the real class-level admission policy."""

    from tracer.services.clickhouse.v2.query_builders.session_list import (
        SessionListQueryBuilderV2,
    )

    factory = mock.MagicMock(wraps=SessionListQueryBuilderV2)

    def construct(**params):
        instance.page_number = params.get("page_number", 0)
        instance.page_size = params.get("page_size", 30)
        return instance

    factory.side_effect = construct
    return factory


def _read_session_page(bounded: BoundedFilterPage) -> dict[str, Any]:
    from tracer.views.trace_session import TraceSessionView

    view = TraceSessionView.__new__(TraceSessionView)
    view._gm = SimpleNamespace(
        success_response=lambda payload: ("ok", payload),
        custom_error_response=lambda *args, **kwargs: ("error", args, kwargs),
        bad_request=lambda message: ("bad_request", message),
    )
    organization = SimpleNamespace(id=uuid.uuid4())
    request = SimpleNamespace(
        query_params={},
        organization=organization,
        user=SimpleNamespace(organization=organization),
    )

    builder = mock.MagicMock()
    builder.supports_candidate_first_page.return_value = False
    builder.supports_candidate_cursor_page.return_value = False
    builder.prefers_bounded_filter_page.return_value = True
    builder.supports_bounded_filter_scan.return_value = True
    builder.recommended_filter_classify_batch_size.return_value = 50
    builder.filter_candidate_seed_is_sampled.return_value = False
    builder.parse_time_range.return_value = (WINDOW_START, WINDOW_END)
    builder.build_page_metrics_query.return_value = ("page metrics", {})
    builder.build_content_query.return_value = ("page content", {})
    builder.build_span_attributes_query.return_value = ("page attributes", {})
    builder.format_sessions.side_effect = lambda rows, columns: [
        dict(zip(columns, row, strict=True)) for row in rows
    ]

    analytics = mock.MagicMock()
    analytics.execute_ch_query.side_effect = lambda *_args, **_kwargs: SimpleNamespace(
        data=[
            {
                "session_id": str(row["session_id"]),
                "session_start": row["start_time"],
                "session_end": row["start_time"],
                "duration": 0,
                "total_cost": 0,
                "total_tokens": 0,
                "traces_count": 1,
            }
            for row in bounded.rows
        ]
    )
    view._fetch_session_names = mock.MagicMock(return_value={})
    view._fetch_end_user_info = mock.MagicMock(return_value={})

    with (
        mock.patch(
            "tracer.views.trace_session.SessionListQueryBuilderV2",
            _session_builder_class(builder),
        ),
        mock.patch(
            "tracer.views.trace_session.read_bounded_filter_page",
            return_value=bounded,
        ),
        mock.patch(
            "tracer.views.trace_session.AnnotationsLabels.objects.filter",
            return_value=[],
        ),
    ):
        outcome = TraceSessionView._list_sessions_clickhouse(
            view,
            request,
            project_id=PROJECT_ID,
            project=None,
            analytics=analytics,
            validated_data={
                "filters": [_time_filter(), _absent_attribute_filter()],
                "sort_params": [],
                "page_number": 0,
                "page_size": 25,
                "cursor_mode": True,
            },
        )

    assert outcome[0] == "ok", outcome
    return outcome[1]


def test_empty_session_checkpoint_reports_the_degraded_read_it_actually_made() -> None:
    payload = _read_session_page(_checkpointed_page(rows=[], before_id=None))

    metadata = payload["metadata"]
    assert metadata["query_complete"] is False
    assert metadata["query_status"] == "degraded"
    assert metadata["query_error_code"] == "scan_budget_exceeded"
    assert metadata["has_more"] is True
    assert isinstance(metadata["next_cursor"], str)


def test_session_checkpoint_with_classified_rows_stays_a_complete_chunk() -> None:
    session_id = str(uuid.uuid4())
    payload = _read_session_page(
        _checkpointed_page(
            rows=[{"session_id": session_id, "start_time": CHECKPOINT}],
            before_id=session_id,
        )
    )

    metadata = payload["metadata"]
    assert metadata["query_complete"] is True
    assert metadata["query_status"] == "complete"
    assert metadata["query_error_code"] is None
    assert metadata["has_more"] is True
