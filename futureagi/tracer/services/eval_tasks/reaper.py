"""Reaper — resets stale ``running`` entries back to ``pending`` so a worker
that died mid-run can't strand them. A poison cap fails an item that
keeps dying after ``max_attempts`` reclaims, so it can't block task completion.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from django.db.models import F
from django.utils import timezone

from tfc.settings.runtime_setting_specs import LONGEST_RUNNING_ENTRY_SECONDS
from tracer.models.observation_span import EvalEntryStatus, EvalLogger

if TYPE_CHECKING:
    from tracer.models.eval_task import EvalTask

# The shortest staleness a *production* reap may act on. A reap always races
# one run it cannot see: an activity a since-closed execution left in flight on
# a worker, bounded by the run-entry start-to-close ceiling. Retiring a claim
# inside that window requeues an entry whose run is still executing — it is
# re-claimed and evaluated a second time, one of its three reclaims is spent,
# and the first run's paid result is then refused by the write fence.
#
# ``LONGEST_RUNNING_ENTRY_SECONDS`` is the same bound
# ``validate_eval_execution_settings`` holds the sweep's own threshold above.
# Applying it here is what makes it bind on *every* reap rather than only on
# the one caller that reads a validated setting: the reap the workflow runs at
# start asks for ``ReapInput.older_than_seconds`` (600), which is well inside
# that window, and it is reached by Resume, by Edit → Save and by the restart
# the sweep issues.
MIN_STALE_RUNNING_SECONDS = LONGEST_RUNNING_ENTRY_SECONDS + 1


def effective_stale_seconds(requested: int) -> int:
    """The staleness a production reap really applies.

    Callers name the threshold they want and this raises it to the floor above
    when it is shorter. ``reap_stale_running`` itself stays exact — it is the
    primitive, and a caller that has established there is no live run (a test,
    a future caller that owns the execution) states its own threshold.
    """
    return max(int(requested), MIN_STALE_RUNNING_SECONDS)


def reap_stale_running(
    task: EvalTask, *, older_than_seconds: int, max_attempts: int
) -> tuple[int, int]:
    """Reclaim entries stuck in ``running`` longer than ``older_than_seconds``.

    Returns ``(requeued, failed)``: under the cap → back to ``pending``
    (attempts incremented); at/over the cap → ``errored`` permanently.
    """
    now = timezone.now()
    cutoff = now - timedelta(seconds=older_than_seconds)
    stale = EvalLogger.objects.filter(
        eval_task_id=str(task.id),
        status=EvalEntryStatus.RUNNING,
        updated_at__lt=cutoff,
    )
    failed = stale.filter(attempts__gte=max_attempts).update(
        status=EvalEntryStatus.ERRORED,
        error=True,
        error_message="reaper: max attempts exceeded",
        updated_at=now,
    )
    requeued = stale.filter(attempts__lt=max_attempts).update(
        status=EvalEntryStatus.PENDING,
        attempts=F("attempts") + 1,
        updated_at=now,
    )
    return requeued, failed
