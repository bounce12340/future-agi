"""The wall clock around one evaluation's execution.

Nothing bounded ``eval_instance.run``: the activity's heartbeat is emitted by a
background timer while the work runs in a thread, so it keeps beating for a
wedged evaluation, and the only effective ceiling was the activity's twelve
hours. These tests pin the bound, the disabled path, and the context copy that
keeps the engine's ContextVars readable inside the bounded call.
"""

import threading
from contextvars import ContextVar

import pytest

from evaluations.engine.wall_clock import (
    EvalWallClockExceeded,
    eval_wall_clock_scope,
    run_bounded,
)

_probe: ContextVar[str] = ContextVar("wall_clock_test_probe", default="unset")


def test_a_call_that_finishes_returns_its_value():
    assert run_bounded(lambda x: x * 2, {"x": 21}, timeout_seconds=30, label="t") == 42


def test_a_call_that_raises_re_raises_on_the_callers_thread():
    def _boom():
        raise ValueError("engine failure")

    with pytest.raises(ValueError, match="engine failure"):
        run_bounded(_boom, {}, timeout_seconds=30, label="t")


def test_a_call_that_overruns_is_abandoned_and_recorded():
    """The caller stops waiting. The thread is not killed — it cannot be — so
    the guarantee is that the drain is unblocked, not that the work stops."""
    release = threading.Event()

    def _wedge():
        release.wait(30)

    with eval_wall_clock_scope() as timeouts:
        with pytest.raises(EvalWallClockExceeded) as captured:
            run_bounded(_wedge, {}, timeout_seconds=1, label="Slow Eval")

    release.set()
    assert "Slow Eval" in str(captured.value)
    assert "1s wall clock" in str(captured.value)
    assert timeouts == [str(captured.value)]


def test_the_wall_can_be_disabled():
    """0 is the escape hatch for a deployment whose evaluations legitimately
    run longer than any bound worth defaulting to."""
    seen = {}

    def _record():
        seen["thread"] = threading.current_thread().name
        return "done"

    assert run_bounded(_record, {}, timeout_seconds=0, label="t") == "done"
    assert seen["thread"] == threading.current_thread().name


def test_the_bounded_call_sees_the_callers_context_vars():
    """The engine reads its write target, its read-source override and the
    OpenTelemetry span from ContextVars. A bare thread would read the defaults
    and the evaluation would write to the wrong place."""
    token = _probe.set("caller-value")
    try:
        assert run_bounded(_probe.get, {}, timeout_seconds=30, label="t") == (
            "caller-value"
        )
    finally:
        _probe.reset(token)


def test_a_timeout_outside_a_scope_is_still_raised():
    """Every caller of ``run_eval`` gets the bound, not just the eval-task
    drain that opens a scope to read the flag back. Callers that build an
    instance and run it themselves bypass ``run_eval`` and are not bounded;
    the engine's module docstring names them."""
    release = threading.Event()

    with pytest.raises(EvalWallClockExceeded):
        run_bounded(lambda: release.wait(30), {}, timeout_seconds=1, label="t")

    release.set()
