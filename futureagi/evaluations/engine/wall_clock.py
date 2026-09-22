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

``EVAL_RUN_WALL_SECONDS`` sets the bound. 0 disables it, which is the escape
hatch for a deployment whose evaluations legitimately run longer.
"""

from __future__ import annotations

import contextvars
import threading
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


class EvalWallClockExceeded(RuntimeError):
    """One evaluation outlived its wall clock."""


def configured_wall_seconds() -> int:
    return int(getattr(settings, "EVAL_RUN_WALL_SECONDS", DEFAULT_WALL_SECONDS))


@contextmanager
def eval_wall_clock_scope():
    """Collect the wall-clock timeouts that fire inside this block.

    Yields the list the timeouts are recorded into, so a caller that cannot see
    the exception can still read that one happened.
    """
    record: list[str] = []
    token = _timeouts.set(record)
    try:
        yield record
    finally:
        _timeouts.reset(token)


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
    """
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
            f"Evaluation '{label}' exceeded its {timeout_seconds}s wall clock "
            "and was abandoned"
        )
        record_timeout(message)
        raise EvalWallClockExceeded(message)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


__all__ = [
    "DEFAULT_WALL_SECONDS",
    "EvalWallClockExceeded",
    "configured_wall_seconds",
    "eval_wall_clock_scope",
    "record_timeout",
    "run_bounded",
]
