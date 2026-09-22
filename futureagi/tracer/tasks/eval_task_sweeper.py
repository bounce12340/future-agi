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
* reclaim entries stuck ``running`` past ``EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS``
  — this is what lets the reaper reach a task whose workflow is alive but no
  longer draining it, and the threshold is why it cannot race one that is;
* ask Temporal whether a workflow is progressing, and restart only those where
  none is.

``paused`` and ``deleted`` tasks are never touched: restarting them spends
evaluation calls on work their owner stopped on purpose. ``failed`` is out of
scope for the same reason and is opt-in through
``EVAL_TASK_SWEEP_RECOVER_FAILED``; the Resume button recovers one explicitly.
"""

from __future__ import annotations

import structlog
from django.conf import settings
from django.db.models import Max

from tfc.temporal.drop_in import temporal_activity
from tracer.models.eval_task import EvalTask, EvalTaskStatus
from tracer.models.observation_span import EvalEntryStatus, EvalLogger
from tracer.services.eval_tasks.reaper import reap_stale_running

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
    else: its workflow is progressing, so it is left alone.

    ``no_workspace_objects``: this is a system-wide job and must not inherit a
    leaked workspace scope. Both managers exclude soft-deleted rows, so the
    entries a Delete & rerun wiped cannot read as undrained work.
    """
    limit = limit if limit is not None else int(settings.EVAL_TASK_SWEEP_MAX_TASKS)
    # ``eval_task_id`` is a CharField, not a relation, so the id set is
    # materialized here rather than left as a subquery against a uuid column.
    stranded_ids = [
        row["eval_task_id"]
        for row in EvalLogger.no_workspace_objects.filter(
            status__in=_UNDRAINED,
            eval_task_id__isnull=False,
        )
        .values("eval_task_id")
        .annotate(last_touched=Max("updated_at"))
        .order_by("last_touched")[:limit]
    ]
    if not stranded_ids:
        return []
    by_id = {
        str(task.id): task
        for task in EvalTask.no_workspace_objects.filter(
            id__in=stranded_ids, status__in=sweepable_statuses()
        )
    }
    return [by_id[task_id] for task_id in stranded_ids if task_id in by_id]


def recover_task(task: EvalTask, *, stale_running_seconds: int) -> dict:
    """Reap the task's stale ``running`` entries, then restart it if nothing is
    draining it.

    The reap runs whatever the workflow is doing — that is the point: the
    workflow-start reaper can never reach a task whose workflow is still alive
    but no longer draining. It is safe beside a live worker because the
    threshold is longer than a running entry's longest legitimate life, and
    because ``persist_eval_result`` / ``mark_terminal`` fence their writes on
    ``RUNNING``, so a late result from a requeued entry no-ops instead of
    landing on the re-claimed row.

    The restart coalesces (``replace_existing=False``) rather than terminating:
    the workflow id is per task, so the Temporal server decides atomically
    whether a live execution already owns it. A describe is a read taken a
    moment earlier, so terminating on it would kill a workflow a user started
    in between — the exact failure this whole change exists to prevent.
    """
    from tfc.temporal.eval_tasks.client import (
        WF_PROGRESSING,
        describe_eval_task_workflow_sync,
        start_eval_task_workflow_sync,
    )

    requeued, failed = reap_stale_running(
        task,
        older_than_seconds=stale_running_seconds,
        max_attempts=_MAX_ENTRY_ATTEMPTS,
    )
    outcome = {"requeued": requeued, "failed": failed, "restarted": False}
    if describe_eval_task_workflow_sync(task.id) == WF_PROGRESSING:
        return outcome

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
    """
    stale_running_seconds = int(settings.EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS)
    tasks = find_stranded_tasks()
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
    }
    logger.info("eval_task_sweep_completed", **result)
    return result


__all__ = [
    "find_stranded_tasks",
    "recover_task",
    "sweep_stranded_eval_tasks",
    "sweepable_statuses",
]
