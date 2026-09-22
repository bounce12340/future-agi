"""Workflow starters for eval tasks.

``start_eval_task_workflow`` picks the historical or continuous workflow by the
task's ``run_type`` and starts it under the per-task id ``eval-task-{id}`` so at
most one workflow runs per task. Wired into the views at cutover (PR 9).
"""

from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

from tfc.temporal.common.client import (
    signal_workflow_sync,
    start_workflow_async,
    start_workflow_sync,
)


def _workflow_id(task_id: str) -> str:
    return f"eval-task-{task_id}"


def _select(task, task_queue):
    """Return (workflow_class, workflow_input) for the task's run_type."""
    from tfc.temporal.eval_tasks.types import (
        ContinuousDrainState,
        EvalTaskWorkflowInput,
    )
    from tfc.temporal.eval_tasks.workflows import (
        ContinuousEvalTaskWorkflow,
        HistoricalEvalTaskWorkflow,
    )
    from tracer.models.eval_task import RunType

    if task.run_type == RunType.CONTINUOUS:
        return ContinuousEvalTaskWorkflow, ContinuousDrainState(
            task_id=str(task.id), task_queue=task_queue
        )
    return HistoricalEvalTaskWorkflow, EvalTaskWorkflowInput(
        task_id=str(task.id), task_queue=task_queue
    )


def start_eval_task_workflow_sync(
    task,
    task_queue: str = "tasks_s",
    *,
    replace_existing: bool = False,
) -> str:
    """Start the workflow for ``task`` from synchronous Django code.

    New-task/legacy callers coalesce with an already-active workflow. A rerun
    or resume passes ``replace_existing=True`` after its PENDING state commits,
    so a stale or closing execution cannot absorb the start and strand the row.
    """
    workflow_class, workflow_input = _select(task, task_queue)
    conflict_policy = (
        WorkflowIDConflictPolicy.TERMINATE_EXISTING
        if replace_existing
        else WorkflowIDConflictPolicy.USE_EXISTING
    )
    handle = start_workflow_sync(
        workflow_class=workflow_class,
        workflow_input=workflow_input,
        workflow_id=_workflow_id(str(task.id)),
        task_queue=task_queue,
        cancel_existing=False,
        # Closed executions may be followed by a new run under the per-task ID.
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=conflict_policy,
    )
    return handle.id


# What a describe of the per-task workflow id told us about the task's drain.
WF_PROGRESSING = "progressing"
WF_ABSENT = "absent"
WF_CLOSED = "closed"


async def describe_eval_task_workflow_async(task_id) -> str:
    """Classify the task's workflow: progressing, closed, or never/no longer there.

    ``WF_PROGRESSING`` means a RUNNING execution owns the task and must be left
    alone. ``WF_ABSENT`` (NOT_FOUND — retention expired, or one was never
    started) and ``WF_CLOSED`` (completed / failed / terminated / timed out)
    both mean nothing is draining the task. Any other RPC failure is the
    Temporal service being unreachable, not an answer, so it propagates: the
    caller must skip that task rather than restart it blind.
    """
    from temporalio.client import WorkflowExecutionStatus
    from temporalio.service import RPCError, RPCStatusCode

    from tfc.temporal.common.client import get_client

    client = await get_client()
    handle = client.get_workflow_handle(_workflow_id(str(task_id)))
    try:
        description = await handle.describe()
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return WF_ABSENT
        raise
    if description.status == WorkflowExecutionStatus.RUNNING:
        return WF_PROGRESSING
    return WF_CLOSED


def describe_eval_task_workflow_sync(task_id) -> str:
    from tfc.temporal.common.client import _run_async_in_sync_context

    return _run_async_in_sync_context(
        lambda: describe_eval_task_workflow_async(task_id)
    )


def signal_pause_eval_task_workflow(task_id) -> bool:
    """Tell the running workflow to stop launching new evals at once. Best-effort
    — the paused DB status the caller already wrote is the durable source of
    truth the workflow also checks at each batch boundary."""
    return signal_workflow_sync(_workflow_id(str(task_id)), "pause")


async def start_eval_task_workflow_async(
    task,
    task_queue: str = "tasks_s",
    *,
    replace_existing: bool = False,
) -> str:
    workflow_class, workflow_input = _select(task, task_queue)
    conflict_policy = (
        WorkflowIDConflictPolicy.TERMINATE_EXISTING
        if replace_existing
        else WorkflowIDConflictPolicy.USE_EXISTING
    )
    handle = await start_workflow_async(
        workflow_class=workflow_class,
        workflow_input=workflow_input,
        workflow_id=_workflow_id(str(task.id)),
        task_queue=task_queue,
        cancel_existing=False,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=conflict_policy,
    )
    return handle.id


__all__ = [
    "WF_ABSENT",
    "WF_CLOSED",
    "WF_PROGRESSING",
    "describe_eval_task_workflow_async",
    "describe_eval_task_workflow_sync",
    "start_eval_task_workflow_sync",
    "start_eval_task_workflow_async",
    "signal_pause_eval_task_workflow",
]
