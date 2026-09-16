"""Cost the filtered graph's raw span scan before the statement is issued.

A filtered Observe graph reads every physical span in its window and folds
trace membership in ClickHouse. On the highest-volume reference tenant that
window is hundreds of millions of rows, and the read learns it only by
running: the interactive wall expires, the dispatcher returns a degraded
payload, and the background worker then re-runs the identical SQL. The wall is
spent and nothing is published any sooner.

This module answers the same question from part metadata. ``EXPLAIN ESTIMATE``
over the statement's own key condition reports the rows the part index says
that scan will touch without reading a single part; a measured scan rate turns
those rows into milliseconds, compared against the deadline the request
actually has left. Nothing here decides membership, prunes a candidate or
reaches a published point: it decides which lane runs the statement, and the
statement is byte-identical either way, so the answer a user finally sees is
the same value an ungated read would have produced.

The estimate is an UPPER BOUND, twice over - whole granules, and every
physical ``ReplacingMergeTree`` version inside them - and the scan window it
costs is the widest one the builder can emit. Both errors point the same way,
at scheduling a read that might have fitted, never at running one that cannot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

import structlog
from django.conf import settings

logger = structlog.get_logger(__name__)

# The column names ClickHouse returns for ``EXPLAIN ESTIMATE``. They are the
# discriminator that tells a genuine estimate from a transport that answered
# something else entirely; the row count cannot do that job.
_ESTIMATE_COLUMNS = frozenset({"database", "table", "parts", "rows", "marks"})
_ESTIMATE_TABLE = "spans"
# The raw graph statement scans ``[start - 1 day, end + 1 day)`` whenever it
# carries row predicates, so that - not the request window - is what a cost
# estimate must cover. A span graph scans the narrower request window; costing
# it at the wider one keeps a single statement honest for both families and
# errs toward scheduling.
_SCAN_WINDOW_MARGIN = timedelta(days=1)
# Measured on production against the highest-volume reference tenant, running
# this surface's own unseeded 30-day filtered trace statement at
# DASHBOARD_TRACE_READ_MAX_THREADS workers: 5,572,506 physical spans and
# 12.04 GiB in 5,756 ms of server time, i.e. 968 rows/ms at 2.25 GB/s and
# 2,323 bytes per row. Faster rates exist for the same surface on the same
# tenant (74.2M rows in 28.5 s, 2,605 rows/ms, on rows roughly half as wide),
# and this is deliberately the slower end: predicting with the fastest rate
# under-predicts the time a read needs, which is the very mistake that spends
# the wall and publishes nothing.
#
# This is a throughput calibration, not a window: the affordable row count is
# always this rate multiplied by the milliseconds the request still has, so a
# longer wall admits a proportionally larger scan and no window length is
# written down anywhere.
_RAW_SCAN_ROWS_PER_MS = 968


def raw_graph_scan_window(
    start_date: datetime | None,
    end_date: datetime | None,
) -> tuple[datetime, datetime] | None:
    """Return the widest window the raw graph statement can scan."""

    if start_date is None or end_date is None or start_date >= end_date:
        return None
    return start_date - _SCAN_WINDOW_MARGIN, end_date + _SCAN_WINDOW_MARGIN


def _reduce_estimate(
    rows: Iterable[Mapping[str, Any]] | None,
    columns: Iterable[str] | None,
) -> int | None:
    """Reduce one ``EXPLAIN ESTIMATE`` result to the rows the scan would read.

    Three shapes have to be told apart, and the ``columns`` the transport
    reports are what separate the last one - not the row count:

    * the estimate table with part rows is their summed ``rows``;
    * the estimate table with NO rows is zero. For this statement that reading
      is unambiguous: the key condition is ``project_id`` and a half-open
      ``start_time`` range over a table partitioned by ``toDate(start_time)``,
      there is no subquery and no step that could vanish, so "no part
      selected" means "nothing to read" and the scan is affordable;
    * anything else - a transport that answered something other than this
      statement, or a server whose estimate table changed shape - is ``None``,
      meaning unknown.
    """

    try:
        names = {str(name) for name in (columns or ())}
        candidate_rows = list(rows or ())
    except TypeError:
        # A transport that answered something other than a result set at all.
        # "Unknown" covers that too; raising here would turn a routing
        # optimisation into a failed request.
        return None
    if not _ESTIMATE_COLUMNS.issubset(names):
        return None
    estimate = 0
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            return None
        if str(row.get("table") or "") != _ESTIMATE_TABLE:
            return None
        counted = row.get("rows")
        if isinstance(counted, bool) or not isinstance(counted, (int, float)):
            return None
        estimate += max(0, int(counted))
    return estimate


def estimate_raw_graph_scan_rows(
    *,
    analytics: Any,
    project_id: str,
    scan_start: datetime,
    scan_end: datetime,
    timeout_ms: int,
) -> int | None:
    """Estimate the physical spans one filtered graph statement would read.

    Deliberately the cheapest statement that answers the question, and for the
    same reasons the trace list's own density probe gives:

    * the key condition is ``project_id`` plus the half-open ``start_time``
      range the statement itself scans, and nothing else. ``is_deleted`` is not
      in the primary key, so it could not narrow an index estimate; leaving it
      out keeps the statement honest about what it measures - every physical
      row in the granules the scan touches, tombstones and stale versions
      included, which is what the graph statement has to walk past;
    * no attribute predicate and no ``indexHint``. Measured on production, the
      attribute witness this surface's seed probe carries prunes 0.08% of the
      estimated rows (87,413,176 to 87,346,495) and 109 of 14,532 granules,
      and costs 831 ms of single-worker skip-index analysis at 30 days and
      2,777 ms at twelve months - over that probe's own 1,500 ms budget. The
      pruning is not worth the money at any window;
    * no ``IN`` subquery: ``EXPLAIN`` executes a scalar/``IN`` subquery to plan
      around it, which would put a real read back inside a statement whose
      whole purpose is not to have one;
    * ``optimize_use_projections`` is pinned OFF. ``spans`` carries projections
      the optimizer will route a bare ``count()`` to, and ``rows`` would then
      describe that projection rather than the base table the graph statement
      reads. Measured on the same 30-day window, leaving projections on
      reports 10,846 marks where the base table needs 14,532.

    It answers a COST question only: which lane runs the statement, never what
    the statement returns.
    """

    probe_settings = {
        "max_threads": settings.DASHBOARD_TRACE_READ_MAX_THREADS,
        "optimize_use_projections": 0,
    }
    query = """
        EXPLAIN ESTIMATE
        SELECT count()
        FROM spans
        WHERE project_id = toUUID(%(graph_cost_project_id)s)
          AND start_time >= %(graph_cost_scan_start)s
          AND start_time < %(graph_cost_scan_end)s
    """
    params = {
        "graph_cost_project_id": str(project_id),
        "graph_cost_scan_start": scan_start,
        "graph_cost_scan_end": scan_end,
    }
    try:
        result = analytics.execute_ch_query(
            query,
            params,
            timeout_ms=int(timeout_ms),
            settings=probe_settings,
        )
    except Exception:
        # The cost probe is an optimisation of WHICH lane runs the statement.
        # Its absence restores exactly the routing this surface had before it
        # existed, which the post-hoc degrade-and-schedule path still covers.
        logger.info("graph_raw_scan_estimate_unavailable", exc_info=True)
        return None
    return _reduce_estimate(
        getattr(result, "data", None), getattr(result, "columns", None)
    )


def raw_graph_scan_fits_wall(estimated_rows: int | None, *, remaining_ms: int) -> bool:
    """Whether a scan of *estimated_rows* can complete in *remaining_ms*.

    An unknown estimate is not evidence of a large scan, so it fits: the gate
    fires on proof, never on the absence of it.
    """

    if estimated_rows is None:
        return True
    affordable_rows = max(0, int(remaining_ms)) * _RAW_SCAN_ROWS_PER_MS
    if estimated_rows <= affordable_rows:
        return True
    logger.info(
        "graph_raw_scan_predicted_over_wall",
        estimated_rows=int(estimated_rows),
        affordable_rows=int(affordable_rows),
        remaining_ms=int(remaining_ms),
    )
    return False


__all__ = [
    "estimate_raw_graph_scan_rows",
    "raw_graph_scan_fits_wall",
    "raw_graph_scan_window",
]
