"""Route a typed span-attribute Session page by the cost of its witness scan.

WHY THIS EXISTS. A Session-list page filtered by one typed span-attribute leaf
(a string, number or boolean picker value) can be served by two exact lanes.
The candidate cursor statement is seeded by ONE whole-window any-span witness -
a scalar subquery naming every session that owns a physical row carrying the
value - and ClickHouse materialises that scalar, and every ``IN`` set built
from it, while PLANNING. On the high-volume tenant at twelve months the
witness scan alone is 238 M rows, so the statement dies at the 30 s wall
before reading a byte. The bounded walk is exact for the same predicate and
bounded per statement - root-ordered seeds by slice, then a classifier over
only the seeded sessions - and the walk is where every typed leaf now goes.
But on a sparse tenant every forty-eight-hour slice is empty, and an empty
twelve-month answer that was one complete 2 s statement became thirty-two
seed statements with a cursor: about four cursor hops to learn that nothing
matches.

WHAT THIS MODULE DECIDES. The lane, and nothing else. Before the page's first
statement it asks the index what the witness scan would read: an ``EXPLAIN
ESTIMATE`` of exactly that scan - rendered by the builder from the same
helper the statement itself uses - reports, per part, the rows the key
condition and the skip indexes leave, without reading column data. Under the
runtime row target the page takes the seeded lane, which on the sparse tenant
is its single statement back; over it, the walk. Membership, coverage and
order are never touched by this decision: both lanes are exact on the list's
own cursor order, and set-valued session predicates stay whole-window per
candidate on either.

WHICH PAGES IT GOVERNS. Number and boolean leaves - the shapes the walk
routing demoted from the candidate lane, and the shapes whose witness rows
price their statement (57 B per estimated row on the sparse tenant's numeric
case). A string leaf is never probed and keeps the walk it has always had: the
seeded statement reads the wide string map in every scan, and on the same
tenant a string witness estimated at 0.49 M rows ran a 4.9 s statement over
7.3 GB, about 15 KB per estimated row. A row estimate is not a cost there.

WHAT A FAILED PROBE MEANS. The walk, always. A probe that raises, a result
that is not an estimate table, an estimate that names no part (ambiguous
between "nothing to read" and "no readable step" - the reading the other seed
lanes refuse to make without a same-request proof), a probe that returns after
its budget, and a request with no budget left for a probe all route to the
walk. Not knowing is never a licence for the unbounded statement; that is the
lesson of every ``EXPLAIN ESTIMATE`` consumer before this one, and the shape
this gate exists to remove is a probe that cannot answer followed by the
statement spending the whole wall.

THE PROBE RUNS AT THE COSTED STATEMENT'S SETTINGS. Same ``read_settings``
callable, so the same ``max_threads``, block size, memory ceiling; the one
deliberate difference is ``optimize_use_projections = 0``, because ``spans``
carries projections the optimizer may route a bare ``count()`` to, and an
estimate of a projection is not an estimate of the base table the statement
reads. A second round of a sibling gate ran its estimate at one thread and
paid ten times the statement it replaced; this one does not.

THE BUDGET BINDS ON THE CLIENT CLOCK ONLY. Production sends no server
``max_execution_time`` on any statement (``application_read_settings`` zeroes
it and is re-applied over every caller) and the application transport runs
the statement with no socket deadline either, so nothing here can cut a probe
short. The budget bounds what the gate ACCEPTS: a probe that came back after
it routes to the walk, however small its estimate, and the time it took was
spent. The probe is therefore only worth its place where it is cheap, which is
what the measurements in the pull request establish per index shape.

HOP CONSISTENCY. The first page of a pagination runs the probe and the lane
it decided is written into the cursor; every later hop reads the lane from
the cursor and runs no probe. A page and its hops therefore take one lane
even if the index estimate drifts between hops, and a token minted before the
field existed resolves to the walk, which is the lane every typed-leaf
pagination was on before this gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol

import structlog
from django.conf import settings

from tracer.selectors.filter_seed_width import EMPTY_DENSITY_ESTIMATE
from tracer.services.clickhouse.list_cursor import (
    TYPED_WITNESS_LANE_SEEDED,
    TYPED_WITNESS_LANE_WALK,
    TYPED_WITNESS_LANES,
)
from tracer.services.clickhouse.read_budget import ReadDeadline, ReadDeadlineExceeded

logger = structlog.get_logger(__name__)

# The estimate table is one row per part-set the plan reads; a page-sized
# result cap keeps the probe's own transport shape identical to the other
# ``EXPLAIN ESTIMATE`` consumers.
_PROBE_MAX_RESULT_ROWS = 256


class _QueryExecutor(Protocol):
    def execute_ch_query(
        self,
        query: str,
        params: dict[str, Any],
        *,
        timeout_ms: int,
        settings: dict[str, Any],
    ) -> Any: ...


@dataclass(frozen=True)
class WitnessLaneDecision:
    """What the gate decided, and what deciding it cost.

    ``lane`` is ``None`` for a page the typed-witness policy does not govern
    (no probe ran, nothing was pinned). ``probe_statements`` is one when a
    probe was issued, whether or not it answered.
    """

    lane: str | None
    reason: str
    estimated_rows: int | None = None
    probe_ms: float | None = None
    probe_statements: int = 0
    target_rows: int | None = None
    budget_ms: int | None = None


def typed_witness_seeded_target_rows() -> int:
    """The estimated witness rows under which the seeded lane is admitted.

    Read at call time so an operator's flip and a test's override both reach
    the next request. Zero turns the gate off: every governed page walks.
    """

    return max(0, int(settings.SESSION_LIST_TYPED_WITNESS_SEEDED_MAX_ESTIMATED_ROWS))


def typed_witness_probe_budget_ms() -> int:
    """The client-clock budget one probe may take and still be believed."""

    return max(1, int(settings.SESSION_LIST_TYPED_WITNESS_PROBE_BUDGET_MS))


def decide_typed_witness_lane(
    *,
    builder: Any,
    analytics: _QueryExecutor,
    deadline: ReadDeadline,
    read_settings: Callable[[int], dict[str, Any]],
    pinned_lane: str | None = None,
) -> WitnessLaneDecision:
    """Pin the lane of one typed-leaf page on ``builder`` and say why.

    ``pinned_lane`` is the lane the pagination's cursor carries, when it does;
    it is taken as-is and no probe runs. Otherwise the probe is the builder's
    own ``build_typed_witness_cost_probe_query`` at ``read_settings`` (the
    costed statement's settings) with projections pinned off, and its answer
    is read by the builder's ``typed_witness_cost_estimate``.
    """

    if pinned_lane is not None:
        if pinned_lane not in TYPED_WITNESS_LANES:
            raise ValueError("unknown typed witness lane")
        builder.pin_typed_witness_lane(pinned_lane)
        return WitnessLaneDecision(lane=pinned_lane, reason="cursor_pinned")

    probe = builder.build_typed_witness_cost_probe_query()
    if not (
        isinstance(probe, tuple)
        and len(probe) == 2
        and isinstance(probe[0], str)
        and probe[0].strip()
        and isinstance(probe[1], dict)
    ):
        # ``None`` is the builder saying the policy does not govern this page.
        # Anything else that is not a statement is treated the same way: no
        # pin, so a governed shape walks and an ungoverned one keeps its lane.
        builder.pin_typed_witness_lane(None)
        return WitnessLaneDecision(lane=None, reason="not_governed")

    target_rows = typed_witness_seeded_target_rows()
    budget_ms = typed_witness_probe_budget_ms()
    if target_rows <= 0:
        return _walk(builder, "gate_off", target_rows=target_rows, budget_ms=budget_ms)
    try:
        remaining_ms = deadline.remaining_ms(budget_ms)
    except ReadDeadlineExceeded:
        remaining_ms = 0
    if remaining_ms < budget_ms:
        # No budget left for a probe is no budget left to be wrong in.
        return _walk(builder, "no_budget", target_rows=target_rows, budget_ms=budget_ms)

    query, params = probe
    probe_settings = {
        **read_settings(_PROBE_MAX_RESULT_ROWS),
        "optimize_use_projections": 0,
    }
    started = monotonic()
    try:
        result = analytics.execute_ch_query(
            query,
            params,
            timeout_ms=deadline.remaining_ms(budget_ms),
            settings=probe_settings,
        )
    except Exception:
        probe_ms = (monotonic() - started) * 1000
        logger.info(
            "session_typed_witness_probe_failed",
            probe_ms=round(probe_ms, 1),
            exc_info=True,
        )
        return _walk(
            builder,
            "probe_failed",
            probe_ms=probe_ms,
            probe_statements=1,
            target_rows=target_rows,
            budget_ms=budget_ms,
        )
    probe_ms = (monotonic() - started) * 1000
    estimate = builder.typed_witness_cost_estimate(
        getattr(result, "data", None), getattr(result, "columns", None)
    )
    common = {
        "probe_ms": probe_ms,
        "probe_statements": 1,
        "target_rows": target_rows,
        "budget_ms": budget_ms,
    }
    if estimate is EMPTY_DENSITY_ESTIMATE:
        return _walk(builder, "empty_estimate", **common)
    if not isinstance(estimate, int):
        return _walk(builder, "unreadable", **common)
    if probe_ms > budget_ms:
        return _walk(builder, "probe_over_budget", estimated_rows=estimate, **common)
    if estimate > target_rows:
        return _walk(builder, "over_target", estimated_rows=estimate, **common)
    builder.pin_typed_witness_lane(TYPED_WITNESS_LANE_SEEDED)
    decision = WitnessLaneDecision(
        lane=TYPED_WITNESS_LANE_SEEDED,
        reason="under_target",
        estimated_rows=estimate,
        **common,
    )
    _log(decision)
    return decision


def _walk(builder: Any, reason: str, **fields: Any) -> WitnessLaneDecision:
    builder.pin_typed_witness_lane(TYPED_WITNESS_LANE_WALK)
    decision = WitnessLaneDecision(
        lane=TYPED_WITNESS_LANE_WALK, reason=reason, **fields
    )
    _log(decision)
    return decision


def _log(decision: WitnessLaneDecision) -> None:
    logger.info(
        "session_typed_witness_lane",
        lane=decision.lane,
        reason=decision.reason,
        estimated_rows=decision.estimated_rows,
        probe_ms=(None if decision.probe_ms is None else round(decision.probe_ms, 1)),
        target_rows=decision.target_rows,
        budget_ms=decision.budget_ms,
    )


__all__ = [
    "WitnessLaneDecision",
    "decide_typed_witness_lane",
    "typed_witness_probe_budget_ms",
    "typed_witness_seeded_target_rows",
]
