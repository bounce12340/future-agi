"""The wall clock around one evaluation's execution.

Nothing bounded ``eval_instance.run``: the activity's heartbeat is emitted by a
background timer while the work runs in a thread, so it keeps beating for a
wedged evaluation, and the only effective ceiling was the activity's twelve
hours. These tests pin the bound, the disabled path, and the context copy that
keeps the engine's ContextVars readable inside the bounded call.
"""

import threading
import time
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


@pytest.mark.django_db(transaction=True)
def test_run_eval_bounds_the_evaluators_own_run(monkeypatch, organization, workspace):
    """The bound reaches the evaluation through ``run_eval``, not only through
    ``run_bounded`` called directly.

    Every test above drives the primitive. The change that matters is the one
    line in ``run_eval`` that routes ``eval_instance.run`` through it, and the
    value it passes: ``configured_wall_seconds()``, read from settings at call
    time. This drives the composed path -- registry, instance, params,
    preprocessing, the bounded call -- with a real template row and a stub
    evaluator that never returns.
    """
    from django.conf import settings

    from evaluations.engine import runner as runner_module
    from evaluations.engine.runner import EvalRequest, run_eval
    from model_hub.models.evals_metric import EvalTemplate

    template = EvalTemplate.objects.create(
        name="Wall Clock Template",
        description="t",
        organization=organization,
        workspace=workspace,
        config={"eval_type_id": "wall_clock_stub"},
    )
    release = threading.Event()
    started = threading.Event()

    class _NeverReturns:
        def run(self, **_kwargs):
            started.set()
            release.wait(30)
            return {"output": "too late"}

    monkeypatch.setattr(runner_module, "get_eval_class", lambda _id: _NeverReturns)
    monkeypatch.setattr(
        runner_module,
        "create_eval_instance",
        lambda **_kwargs: (_NeverReturns(), "criteria"),
    )
    monkeypatch.setattr(settings, "EVAL_RUN_WALL_SECONDS", 1, raising=False)

    try:
        with pytest.raises(EvalWallClockExceeded) as raised:
            run_eval(
                EvalRequest(
                    eval_template=template,
                    inputs={"input": "x"},
                    skip_params_preparation=True,
                )
            )
    finally:
        release.set()

    assert started.is_set()
    assert "Wall Clock Template" in str(raised.value)
    assert "1s wall clock" in str(raised.value)


@pytest.mark.django_db(transaction=True)
def test_an_abandoned_run_closes_the_database_connection_it_opened():
    """The abandoned thread is not killed, so it has to clean up after itself.

    An evaluation reads and writes through the ORM, and the thread that runs it
    opens its own connection (Django's connection handler is thread-local). If
    that connection were left open on every wall-clock timeout the pool would
    leak one per abandoned eval. The module claims the thread closes its own;
    nothing exercised it against a database.
    """
    from django.db import connections

    from model_hub.models.evals_metric import EvalTemplate

    release = threading.Event()
    opened = threading.Event()
    captured = {}

    def _query_then_block():
        EvalTemplate.objects.count()  # opens this thread's connection
        captured["wrapper"] = connections["default"]
        captured["connected"] = connections["default"].connection is not None
        opened.set()
        release.wait(30)

    with pytest.raises(EvalWallClockExceeded):
        run_bounded(_query_then_block, {}, timeout_seconds=1, label="db")

    assert opened.wait(5)
    release.set()
    for _ in range(100):
        if captured["wrapper"].connection is None:
            break
        time.sleep(0.05)

    assert captured["connected"] is True
    assert captured["wrapper"].connection is None
