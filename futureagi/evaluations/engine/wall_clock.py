"""A wall clock around one evaluation's execution.

Nothing bounded the eval itself. The Temporal activity that runs an eval-task
entry declares a heartbeat timeout, but the heartbeat is emitted by a
background timer while the work runs in a thread, so it keeps beating for an
evaluation that has wedged — a hung model call, an unbounded retry inside a
provider SDK — and the only effective bound was the activity's start-to-close
ceiling, twelve hours. One wedged entry blocks its task's whole batch, because
the drain gathers the batch before claiming the next one.

A Temporal timeout cannot help even when it fires: it abandons the activity but
cannot kill the Python thread underneath. So the bound has to live here, at the
call, and it is a wall clock rather than a cancellation: the work runs in a
daemon thread and the caller stops waiting.

That means a timed-out evaluation's thread keeps running until it returns on
its own — it is unblocked, not killed. It is a daemon so it cannot hold the
process open, it closes its own database connections when it finishes, and its
result write is fenced on the entry still being RUNNING so it cannot land on a
row that has since been re-claimed. Bounding a thread we cannot kill is the
best available guarantee; the alternative on the table is twelve hours.

``EVAL_RUN_WALL_SECONDS`` sets the per-evaluation bound. 0 disables it, which
is the escape hatch for a deployment whose evaluations legitimately run longer.

A scope may also carry a **budget**: a deadline shared by every bounded call
inside it. One eval-task entry is one such scope, and it needs one because a
composite entry is not one evaluation — it fans out across its children, each
of which is a separate bounded call. Without a shared deadline N children cost
N walls, and the activity ceiling that has to contain them is a fixed 30
minutes, so a composite wide enough to outlive it used to time out, be retried
from scratch twice more, and end ERRORED having paid for every child three
times. With the budget the entry errors once, inside the ceiling, carrying the
real reason.
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from django.conf import settings
from django.db import connections

DEFAULT_WALL_SECONDS = 300

# Set for the duration of one entry's run so a caller can tell that the wall
# fired even when the evaluation core swallowed the exception — which it does:
# it catches everything around the model call and records a generic error on
# the entry instead of re-raising.
_timeouts: ContextVar[list[str] | None] = ContextVar(
    "eval_wall_clock_timeouts", default=None
)

# The monotonic instant every bounded call in this scope shares, or None when
# the scope carries no budget. Set once per scope, never extended: a budget
# that each call could reset would bound nothing.
_deadline: ContextVar[float | None] = ContextVar(
    "eval_wall_clock_deadline", default=None
)


class EvalWallClockExceeded(RuntimeError):
    """One evaluation outlived its wall clock."""


def configured_wall_seconds() -> int:
    return int(getattr(settings, "EVAL_RUN_WALL_SECONDS", DEFAULT_WALL_SECONDS))


@contextmanager
def eval_wall_clock_scope(*, budget_seconds: int | None = None):
    """Collect the wall-clock timeouts that fire inside this block.

    Yields the list the timeouts are recorded into, so a caller that cannot see
    the exception can still read that one happened.

    ``budget_seconds`` additionally gives every bounded call in the block one
    shared deadline, so the block as a whole is bounded however many
    evaluations it runs. A caller that passes none keeps the per-call bound
    only, which is what a single evaluation wants.
    """
    record: list[str] = []
    token = _timeouts.set(record)
    deadline_token = _deadline.set(
        time.monotonic() + budget_seconds
        if budget_seconds is not None and budget_seconds > 0
        else None
    )
    try:
        yield record
    finally:
        _deadline.reset(deadline_token)
        _timeouts.reset(token)


def remaining_budget_seconds() -> float | None:
    """Seconds left in the enclosing scope's shared budget, or None if none."""
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()


def record_timeout(message: str) -> None:
    record = _timeouts.get()
    if record is not None:
        record.append(message)


def run_bounded(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    timeout_seconds: int,
    label: str,
) -> Any:
    """Run ``func(**kwargs)`` and stop waiting after ``timeout_seconds``.

    The call runs in a daemon thread under a copy of the caller's context, so
    every ContextVar the evaluation reads — the engine's write target, the
    read-source override, the OpenTelemetry span — has the value it would have
    had inline. A non-positive timeout runs the call inline instead, keeping
    the disabled path free of the extra thread entirely.

    An enclosing scope's budget (``eval_wall_clock_scope(budget_seconds=...)``)
    is applied first and always binds: it is what is left of the deadline the
    whole block shares, so it caps this call even when the per-evaluation wall
    is longer or disabled, and a call that starts with none left is refused
    before it costs anything.
    """
    timeout_seconds = _within_budget(timeout_seconds, label)
    if timeout_seconds <= 0:
        return func(**kwargs)

    context = contextvars.copy_context()
    outcome: dict[str, Any] = {}

    def _work() -> None:
        try:
            outcome["value"] = context.run(func, **kwargs)
        except BaseException as exc:  # re-raised on the caller's thread below
            outcome["error"] = exc
        finally:
            # Thread-local, so this closes only this thread's connections —
            # including for a run whose caller has already given up on it.
            connections.close_all()

    worker = threading.Thread(target=_work, name="eval-wall-clock", daemon=True)
    worker.start()
    worker.join(timeout_seconds)

    if worker.is_alive():
        message = (
            f"Evaluation '{label}' exceeded its {timeout_seconds:.0f}s wall "
            "clock and was abandoned"
        )
        record_timeout(message)
        raise EvalWallClockExceeded(message)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _within_budget(timeout_seconds: float, label: str) -> float:
    """Clamp a per-call bound to what the enclosing scope's budget allows."""
    remaining = remaining_budget_seconds()
    if remaining is None:
        return timeout_seconds
    if remaining <= 0:
        message = (
            f"Evaluation '{label}' was not started: its entry's evaluation "
            "budget was already spent"
        )
        record_timeout(message)
        raise EvalWallClockExceeded(message)
    if timeout_seconds <= 0:
        return remaining
    return min(timeout_seconds, remaining)


__all__ = [
    "DEFAULT_WALL_SECONDS",
    "EvalWallClockExceeded",
    "configured_wall_seconds",
    "eval_wall_clock_scope",
    "record_timeout",
    "remaining_budget_seconds",
    "run_bounded",
]
