"""Scheduled recovery sweep for eval tasks whose drain stopped.

An eval task drains through exactly one Temporal workflow, and every start of
one is user-initiated (create, resume, edit). The reaper that reclaims entries
abandoned in ``running`` only runs at workflow start — deliberately skipped
across continue-as-new — so a workflow that stops (failed, terminated, lost
with its worker) leaves every remaining entry frozen with nothing watching.
``has_undrained_work`` exists but is only ever read from inside a running
workflow. This sweep is the missing out-of-band recovery, and the only thing
that makes a stranded task visible at all.

Per tick, bounded and idempotent:

* take tasks in a sweepable status that still have undrained entries, oldest
  activity first, capped at ``EVAL_TASK_SWEEP_MAX_TASKS``;
* ask Temporal whether a workflow is progressing, and leave that task alone if
  one is — a healthy task costs one describe and nothing else;
* for the rest, reclaim entries stuck ``running`` past
  ``EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS`` and restart the workflow.

Asking before reaping is what makes the reap safe to run on a timer. An entry
is ``RUNNING`` from the moment its batch is claimed, not from the moment its
run starts: ``claim_pending_batch`` stamps a whole batch at once and the drain
runs ``max_concurrent`` of them at a time, so a batch tail waits several waves
under a frozen claim stamp. Reaping such a task would requeue an entry whose
own activity is still queued, and that activity would then take the re-claim
and pay for the evaluation a second time. The sweep therefore only ever
retires claims belonging to an execution that is no longer running.

``paused`` and ``deleted`` tasks are never touched: restarting them spends
evaluation calls on work their owner stopped on purpose. ``failed`` is out of
scope for the same reason and is opt-in through
``EVAL_TASK_SWEEP_RECOVER_FAILED``; the Resume button recovers one explicitly.

``EVAL_TASK_SWEEP_MAX_TASKS=0`` disables the sweep. That is the rollback that
survives a restart: pausing the schedule in Temporal takes effect at once but
is undone by the next backend container start, which re-registers every
schedule with its state rebuilt from config.
"""

from __future__ import annotations

import structlog
from django.conf import settings
from django.db.models import Max, Subquery, TextField
from django.db.models.functions import Cast

from tfc.temporal.drop_in import temporal_activity
from tracer.models.eval_task import EvalTask, EvalTaskStatus
from tracer.models.observation_span import EvalEntryStatus, EvalLogger
from tracer.services.eval_tasks.reaper import (
    effective_stale_seconds,
    reap_stale_running,
)

logger = structlog.get_logger(__name__)

# Statuses whose undrained work the sweep may re-enter without asking anyone.
_SWEEPABLE = (EvalTaskStatus.RUNNING, EvalTaskStatus.PENDING)

# Same poison cap the workflow's own reap uses (``ReapInput.max_attempts``), so
# a row that keeps dying is failed after the same number of reclaims however it
# was reaped.
_MAX_ENTRY_ATTEMPTS = 3

_UNDRAINED = (EvalEntryStatus.PENDING, EvalEntryStatus.RUNNING)


def sweepable_statuses() -> list[str]:
    statuses = list(_SWEEPABLE)
    if getattr(settings, "EVAL_TASK_SWEEP_RECOVER_FAILED", False):
        statuses.append(EvalTaskStatus.FAILED)
    return statuses


def find_stranded_tasks(*, limit: int | None = None) -> list[EvalTask]:
    """Tasks in a sweepable status that still hold undrained entries.

    Ordered by the oldest last-touched entry first, so a task frozen for days
    is always served before tasks that are draining normally — otherwise a busy
    fleet would spend every tick's budget on healthy tasks and never reach the
    stranded one. A healthy task selected anyway costs one describe and nothing
    else: ``recover_task`` asks before it writes, so a progressing workflow's
    entries are neither reaped nor restarted.

    ``candidates`` therefore counts stranded tasks *in a sweepable status*, and
    never the rest: a paused, delete-status or failed task holding undrained
    entries, or an entry whose task row is gone, contributes nothing to the
    count and produces no event. ``candidates: 0`` means nothing sweepable is
    stranded, not that nothing is.

    The per-tick cap is applied **after** the sweepable-status filter, not
    before it. A task the sweep refuses to act on — paused, delete-status,
    finished with leftovers, or an entry pointing at a task row that no longer
    exists — never drains, so its entries' ``updated_at`` never advances and it
    sorts oldest on every tick for ever. Costing such a task against the cap
    would hand it a head slot permanently: fill the cap with them and every
    tick returns nothing, silently, for as long as they exist. The runbook
    tells operators not to sweep paused tasks, so paused tasks holding
    undrained entries are the expected steady state, not an edge case.

    ``no_workspace_objects``: this is a system-wide job and must not inherit a
    leaked workspace scope. Both managers exclude soft-deleted rows, so the
    entries a Delete & rerun wiped cannot read as undrained work, and a
    soft-deleted task cannot be swept.
    """
    limit = limit if limit is not None else int(settings.EVAL_TASK_SWEEP_MAX_TASKS)
    # ``eval_task_id`` is a CharField holding the task uuid's text form (every
    # writer stamps ``str(task.id)``), so the sweepable set is cast to text to
    # join against it. As a subquery rather than a materialized id list: the
    # candidate read has to be narrowed to sweepable tasks *inside* the query
    # the cap slices, which is the whole point of the ordering above.
    sweepable_task_ids = (
        EvalTask.no_workspace_objects.filter(status__in=sweepable_statuses())
        .annotate(id_text=Cast("id", output_field=TextField()))
        .values("id_text")
        # The model's Meta ordering would otherwise ride along inside the
        # semi-join, sorting a set nothing reads in order.
        .order_by()
    )
    # ``isnull`` / ``exclude("")`` are redundant beside the semi-join — neither
    # value can be in a set of rendered uuids — and are kept only to drop those
    # rows before the join. They are no longer what stops an empty string
    # reaching a UUID column: nothing compares ``eval_task_id`` to a uuid any
    # more.
    stranded_ids = [
        row["eval_task_id"]
        for row in EvalLogger.no_workspace_objects.filter(
            status__in=_UNDRAINED,
            eval_task_id__isnull=False,
            eval_task_id__in=Subquery(sweepable_task_ids),
        )
        .exclude(eval_task_id="")
        .values("eval_task_id")
        .annotate(last_touched=Max("updated_at"))
        .order_by("last_touched")[:limit]
    ]
    if not stranded_ids:
        return []
    # Re-read as model instances. The status filter is repeated so a task
    # paused or deleted between the two reads is dropped rather than swept.
    by_id = {
        str(task.id): task
        for task in EvalTask.no_workspace_objects.filter(
            id__in=stranded_ids, status__in=sweepable_statuses()
        )
    }
    return [by_id[task_id] for task_id in stranded_ids if task_id in by_id]


def recover_task(task: EvalTask, *, stale_running_seconds: int) -> dict:
    """Ask whether anything is draining the task; if nothing is, reap its stale
    ``running`` entries and restart it.

    The describe comes first and is a gate, not a hint. An entry is ``RUNNING``
    from the moment its batch is claimed, so a task that is draining normally
    holds a tail of claimed-but-unstarted entries whose stamp is as old as the
    claim — reaping those requeues work a queued activity is about to run, and
    that activity then takes the re-claim and pays for the evaluation twice. By
    asking first the sweep only ever retires a claim no execution owns any
    more, and a healthy task really does cost one describe and nothing else.

    A describe that cannot answer (Temporal unreachable) propagates before
    anything is written, so the tick's counts describe what it actually did.

    The reap still reaches what the workflow-start reaper cannot: a workflow
    that stopped mid-drain leaves entries abandoned in ``running``, and nothing
    else looks at them until something starts a workflow. The one run that can
    still be in flight here belongs to an execution that has since closed —
    bounded by a single activity attempt, because a closed execution dispatches
    no retries — which is what ``EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS`` is
    floored to exceed. Its writes are fenced on the claim stamp it took, not
    merely on ``RUNNING``, so a requeue and a re-claim refuse them.

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

    if describe_eval_task_workflow_sync(task.id) == WF_PROGRESSING:
        return {"requeued": 0, "failed": 0, "restarted": False}

    requeued, failed = reap_stale_running(
        task,
        # A validated setting cannot be below the floor, but going through the
        # same helper as the workflow's reap is what makes "no reap retires a
        # claim a live run can still own" hold by construction rather than by a
        # spec bound a later edit could loosen on its own.
        older_than_seconds=effective_stale_seconds(stale_running_seconds),
        max_attempts=_MAX_ENTRY_ATTEMPTS,
    )
    outcome = {"requeued": requeued, "failed": failed, "restarted": False}

    if task.status == EvalTaskStatus.FAILED:
        # A failed row makes ``get_eval_task_state_activity`` report the task
        # inactive, so a restart would exit on its first state check. Clear it
        # the way Resume does, guarded so a concurrent pause or delete wins.
        changed = EvalTask.objects.filter(
            id=task.id, status=EvalTaskStatus.FAILED
        ).update(status=EvalTaskStatus.PENDING)
        if not changed:
            return outcome
        task.status = EvalTaskStatus.PENDING

    start_eval_task_workflow_sync(task, replace_existing=False)
    outcome["restarted"] = True
    return outcome


@temporal_activity(time_limit=600, queue="tasks_s", max_retries=0)
def sweep_stranded_eval_tasks():
    """Restart the workflow of every task whose drain stopped. Counts only.

    ``max_retries=0``: the next tick recovers a sweep-level failure, and one
    task's failure (a Temporal describe against an unreachable service, say)
    must not cost the others their recovery.

    ``EVAL_TASK_SWEEP_MAX_TASKS=0`` turns the sweep off. It reports that as its
    own event rather than as an empty tick, because "disabled" and "nothing is
    stranded" are the two readings an operator has to tell apart, and this job
    only ever speaks in counts.
    """
    limit = int(settings.EVAL_TASK_SWEEP_MAX_TASKS)
    if limit <= 0:
        logger.info("eval_task_sweep_disabled")
        return {
            "candidates": 0,
            "restarted": 0,
            "entries_requeued": 0,
            "entries_poisoned": 0,
            "errors": 0,
            "disabled": True,
        }
    stale_running_seconds = int(settings.EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS)
    tasks = find_stranded_tasks(limit=limit)
    restarted = requeued = poisoned = errors = 0
    for task in tasks:
        try:
            outcome = recover_task(task, stale_running_seconds=stale_running_seconds)
        except Exception as exc:
            errors += 1
            logger.warning("eval_task_sweep_task_failed", error_type=type(exc).__name__)
            continue
        restarted += int(outcome["restarted"])
        requeued += outcome["requeued"]
        poisoned += outcome["failed"]
    result = {
        "candidates": len(tasks),
        "restarted": restarted,
        "entries_requeued": requeued,
        "entries_poisoned": poisoned,
        "errors": errors,
        "disabled": False,
    }
    logger.info("eval_task_sweep_completed", **result)
    return result


__all__ = [
    "find_stranded_tasks",
    "recover_task",
    "sweep_stranded_eval_tasks",
    "sweepable_statuses",
]
