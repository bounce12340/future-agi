"""Recovery of eval tasks whose drain stopped: reap, then restart or defer.

The scheduled sweep (``tracer.tasks.eval_task_sweeper``) is the adapter; this
is the policy. Candidates come from ``tracer.selectors.eval_tasks.stranded``.
"""

from __future__ import annotations

import structlog

from tfc.temporal.eval_tasks.types import ReapInput
from tracer.models.eval_task import EvalTask, EvalTaskStatus
from tracer.selectors.eval_tasks.stranded import (
    find_stranded_tasks,
    only_unreclaimable_claims_remain,
)
from tracer.services.eval_tasks.reaper import (
    effective_stale_seconds,
    reap_stale_running,
)

logger = structlog.get_logger(__name__)

# Same poison cap the workflow's own reap uses (``ReapInput.max_attempts``), so
# a row that keeps dying is failed after the same number of reclaims however it
# was reaped.
_MAX_ENTRY_ATTEMPTS = ReapInput.max_attempts

# What the first reap of a workflow this sweep starts applies. The sweep has
# just been told nothing owns the task, and the starter describes again and
# carries that into the run, so the run reaps with ``ReapInput``'s own
# threshold rather than the blind floor.
RESTART_REAP_SECONDS = effective_stale_seconds(
    ReapInput.older_than_seconds, workflow_confirmed_stopped=True
)


class RecoveryInterrupted(Exception):
    """A recovery that had already written rows when it failed.

    The describe is a gate and its failure propagates bare: nothing is written
    before it answers, so a tick that loses it loses nothing to report. Every
    later step can have reaped rows already, and those requeues and the
    attempts they spent are real whether or not the restart that follows them
    succeeds — so the failure has to carry them out to the tick rather than
    discard them. ``entries_requeued`` is the line the runbook tells an
    operator to watch; it must not read 0 for a reap that happened.
    """

    def __init__(self, outcome: dict, cause: Exception):
        super().__init__(str(cause))
        self.outcome = outcome
        self.cause = cause


def recover_task(task: EvalTask, *, stale_running_seconds: int) -> dict:
    """Ask whether anything is draining the task; if nothing is, reap its stale
    ``running`` entries and restart it — or defer it to a later tick.

    The describe comes first and is a gate, not a hint. An entry is ``RUNNING``
    from the moment its batch is claimed, so a task that is draining normally
    holds a tail of claimed-but-unstarted entries whose stamp is as old as the
    claim — reaping those requeues work a queued activity is about to run, and
    that activity then takes the re-claim and pays for the evaluation twice. By
    asking first the sweep only ever retires a claim no execution owns any
    more, and a healthy task really does cost one describe and nothing else.

    A describe that cannot answer (Temporal unreachable) propagates before
    anything is written. A failure after the reap — the deferral check, the
    status flip, or the start — raises ``RecoveryInterrupted`` carrying the
    reap already performed, so either way the tick's counts describe what it
    actually did.

    The reap still reaches what the workflow-start reaper cannot: a workflow
    that stopped mid-drain leaves entries abandoned in ``running``, and nothing
    else looks at them until something starts a workflow. The one run that can
    still be in flight here belongs to an execution that has since closed —
    bounded by a single activity attempt, because a closed execution dispatches
    no retries — which is what ``EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS`` is
    floored to exceed. Its writes are fenced on the claim stamp it took, not
    merely on ``RUNNING``, so a requeue and a re-claim refuse them.

    **Deferral.** A workflow that stopped a few minutes ago leaves claims
    younger than both this reap and the ``RESTART_REAP_SECONDS`` the restarted
    run's own reap applies. If those are all the task holds, a restart claims
    nothing and cannot finalize. So the task is left exactly as it is — still
    sweepable, nothing written — and a later tick restarts it once a claim is
    reclaimable. ``deferred`` reports it.

    The restart coalesces (``replace_existing=False``) rather than terminating:
    the workflow id is per task, so the Temporal server decides atomically
    whether a live execution already owns it. A describe is a read taken a
    moment earlier, so terminating on it would kill a workflow a user started
    in between — the exact failure this whole change exists to prevent. For the
    same reason ``restarted`` counts starts *issued*: one that coalesced onto an
    execution started in the gap is counted too, because the server resolves
    that and does not report which way it went.
    """
    from tfc.temporal.eval_tasks.client import (
        WF_PROGRESSING,
        describe_eval_task_workflow_sync,
        start_eval_task_workflow_sync,
    )

    outcome = {
        "requeued": 0,
        "failed": 0,
        "restarted": False,
        "progressing": False,
        "deferred": False,
    }
    if describe_eval_task_workflow_sync(task.id) == WF_PROGRESSING:
        # Reported, not merely skipped: a healthy draining task is a candidate
        # that costs one describe and nothing else, so without this the tick
        # has no field that separates the fleet's ordinary working set from
        # the tasks it acted on.
        outcome["progressing"] = True
        return outcome

    outcome["requeued"], outcome["failed"] = reap_stale_running(
        task,
        # Through the same helper as the workflow's own reap, carrying the
        # same evidence: the describe above just answered that no execution
        # owns this task. ``SWEEP_STALE_RUNNING_SECONDS`` declares its minimum
        # as ``LONGEST_RUNNING_ENTRY_SECONDS + 1`` — the blind floor — so both
        # branches of ``effective_stale_seconds`` agree at every value an
        # operator can set; the flag records the evidence, it does not lower
        # the threshold.
        older_than_seconds=effective_stale_seconds(
            stale_running_seconds, workflow_confirmed_stopped=True
        ),
        max_attempts=_MAX_ENTRY_ATTEMPTS,
    )

    try:
        # Checked before the FAILED flip below, so an opted-in failed task that
        # has to wait is left failed rather than made pending with nothing to
        # run.
        if only_unreclaimable_claims_remain(
            task, reclaimable_after_seconds=RESTART_REAP_SECONDS
        ):
            outcome["deferred"] = True
            return outcome

        if task.status == EvalTaskStatus.FAILED:
            # A failed row makes ``get_eval_task_state_activity`` report the
            # task inactive, so a restart would exit on its first state check.
            # Clear it the way Resume does, guarded so a concurrent pause or
            # delete wins.
            changed = EvalTask.objects.filter(
                id=task.id, status=EvalTaskStatus.FAILED
            ).update(status=EvalTaskStatus.PENDING)
            if not changed:
                return outcome
            task.status = EvalTaskStatus.PENDING

        start_eval_task_workflow_sync(task, replace_existing=False)
    except Exception as exc:
        raise RecoveryInterrupted(outcome, exc) from exc
    outcome["restarted"] = True
    return outcome


def recover_stranded_tasks(*, limit: int, stale_running_seconds: int) -> dict:
    """One sweep tick: recover every candidate, isolate each one's failure.

    One task's failure (a Temporal describe against an unreachable service,
    say) must not cost the others their recovery; the next tick retries it.
    Each absorbed failure is logged with its traceback, because the counts
    alone say that something failed and never why.
    """
    tasks = find_stranded_tasks(limit=limit)
    counts = {
        "progressing": 0,
        "restarted": 0,
        "deferred": 0,
        "entries_requeued": 0,
        "entries_poisoned": 0,
        "errors": 0,
    }
    for task in tasks:
        try:
            outcome = recover_task(task, stale_running_seconds=stale_running_seconds)
        except RecoveryInterrupted as exc:
            # The restart is gone, but the reap before it is not: those rows
            # are pending again and have spent an attempt, and the next tick
            # recovers them as ordinary stranded work.
            counts["errors"] += 1
            logger.warning(
                "eval_task_sweep_task_failed",
                task_id=str(task.id),
                error_type=type(exc.cause).__name__,
                exc_info=exc.cause,
            )
            outcome = exc.outcome
        except Exception as exc:
            counts["errors"] += 1
            logger.warning(
                "eval_task_sweep_task_failed",
                task_id=str(task.id),
                error_type=type(exc).__name__,
                exc_info=exc,
            )
            continue
        counts["progressing"] += int(outcome["progressing"])
        counts["restarted"] += int(outcome["restarted"])
        counts["deferred"] += int(outcome["deferred"])
        counts["entries_requeued"] += outcome["requeued"]
        counts["entries_poisoned"] += outcome["failed"]
    return {"candidates": len(tasks), **counts}


__all__ = [
    "RESTART_REAP_SECONDS",
    "RecoveryInterrupted",
    "recover_stranded_tasks",
    "recover_task",
]
