"""Unit tests for the scheduled eval-task recovery sweep.

Before this sweep existed a task whose workflow stopped kept its pending
entries forever: the reaper only runs at workflow start, every start is
user-initiated, and ``has_undrained_work`` is only read from inside a running
workflow. These tests pin the recovery contract — what is swept, what is left
alone, and the threshold that keeps the reap from racing a live worker.
"""

import uuid
from datetime import timedelta

import pytest
from django.utils import timezone

from model_hub.models.ai_model import AIModel
from model_hub.models.evals_metric import EvalTemplate
from tracer.models.custom_eval_config import CustomEvalConfig
from tracer.models.eval_task import EvalTask, EvalTaskStatus, RowType, RunType
from tracer.models.observation_span import (
    EvalEntryStatus,
    EvalLogger,
    EvalTargetType,
    ObservationSpan,
)
from tracer.models.project import Project
from tracer.models.trace import Trace
from tracer.tasks import eval_task_sweeper as sweeper

_UNDRAINED = (EvalEntryStatus.PENDING, EvalEntryStatus.RUNNING)


@pytest.fixture(autouse=True)
def only_this_modules_entries(db):
    """Start every test from an empty undrained set.

    The sweep is a system-wide job: it reads every task in the database, so any
    entry another module left behind changes its answer. ``--reuse-db`` is in
    this repo's pytest addopts and a ``transaction=True`` test commits its rows,
    so leftovers survive between sessions. The delete runs inside the test's own
    transaction and rolls back with it, leaving nothing changed for anyone else.
    """
    EvalLogger.all_objects.filter(status__in=_UNDRAINED).delete()


@pytest.fixture
def sweep_project(db, organization, workspace):
    return Project.objects.create(
        name="Sweep Project",
        organization=organization,
        workspace=workspace,
        model_type=AIModel.ModelTypes.GENERATIVE_LLM,
        trace_type="experiment",
        config=[],
    )


@pytest.fixture
def sweep_config(db, sweep_project, organization, workspace):
    template = EvalTemplate.objects.create(
        name="Sweep Template",
        description="t",
        organization=organization,
        workspace=workspace,
        config={"type": "pass_fail", "criteria": "c"},
    )
    return CustomEvalConfig.objects.create(
        name="Sweep Eval",
        project=sweep_project,
        eval_template=template,
        config={},
        mapping={"input": "input"},
        filters={},
    )


@pytest.fixture
def make_task(db, sweep_project, sweep_config):
    def _make(status=EvalTaskStatus.RUNNING):
        task = EvalTask.objects.create(
            project=sweep_project,
            name="sweep task",
            filters={},
            sampling_rate=100.0,
            spans_limit=100,
            run_type=RunType.HISTORICAL,
            status=status,
            row_type=RowType.SPANS,
        )
        task.evals.add(sweep_config)
        return task

    return _make


@pytest.fixture
def make_entry(db, sweep_config):
    def _make(task, status=EvalEntryStatus.PENDING, age_seconds=0, deleted=False):
        trace = Trace.objects.create(project=task.project, name="sw")
        span = ObservationSpan.objects.create(
            id=f"sw-{uuid.uuid4().hex[:12]}",
            project=task.project,
            trace=trace,
            name="s",
            observation_type="llm",
        )
        entry = EvalLogger.objects.create(
            target_type=EvalTargetType.SPAN,
            observation_span=span,
            trace=trace,
            custom_eval_config=sweep_config,
            eval_task_id=str(task.id),
            status=status,
            deleted=deleted,
        )
        # ``updated_at`` is auto_now, so age it with an explicit update.
        EvalLogger.all_objects.filter(id=entry.id).update(
            updated_at=timezone.now() - timedelta(seconds=age_seconds)
        )
        entry.refresh_from_db()
        return entry

    return _make


@pytest.fixture
def temporal(monkeypatch):
    """Stand in for the Temporal client: record describes and starts."""
    from tfc.temporal.eval_tasks import client

    calls = {"described": [], "started": [], "verdict": client.WF_CLOSED}

    def _describe(task_id):
        calls["described"].append(str(task_id))
        verdict = calls["verdict"]
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    def _start(task, **kwargs):
        calls["started"].append((str(task.id), kwargs))
        return "wf"

    monkeypatch.setattr(client, "describe_eval_task_workflow_sync", _describe)
    monkeypatch.setattr(client, "start_eval_task_workflow_sync", _start)
    return calls


@pytest.mark.django_db
class TestFindStrandedTasks:
    def test_a_task_with_pending_work_is_a_candidate(self, make_task, make_entry):
        task = make_task()
        make_entry(task)

        assert [str(t.id) for t in sweeper.find_stranded_tasks()] == [str(task.id)]

    def test_a_fully_drained_task_is_not_a_candidate(self, make_task, make_entry):
        task = make_task()
        make_entry(task, status=EvalEntryStatus.COMPLETED)
        make_entry(task, status=EvalEntryStatus.SKIPPED)

        assert sweeper.find_stranded_tasks() == []

    def test_soft_deleted_entries_are_not_undrained_work(self, make_task, make_entry):
        """A Delete & rerun soft-deletes every live entry. Those rows keep
        ``status=pending``, so counting them would restart that task forever."""
        task = make_task()
        make_entry(task, deleted=True)

        assert sweeper.find_stranded_tasks() == []

    def test_an_entry_with_a_blank_task_id_cannot_kill_the_tick(
        self, make_task, make_entry
    ):
        """``eval_task_id`` is a CharField, and rows carrying an empty string
        exist — ``queries/eval_clustering.py`` filters them out by name. An empty
        string reaching the ``id__in`` lookup against a UUID column raises before
        the per-task error handling, and the sweep runs with ``max_retries=0``,
        so one such row would kill every tick forever."""
        task = make_task()
        make_entry(task)
        orphan = make_entry(task)
        EvalLogger.all_objects.filter(id=orphan.id).update(eval_task_id="")

        assert [str(t.id) for t in sweeper.find_stranded_tasks()] == [str(task.id)]

    @pytest.mark.parametrize(
        "status", [EvalTaskStatus.PAUSED, EvalTaskStatus.DELETED, EvalTaskStatus.FAILED]
    )
    def test_paused_deleted_and_failed_tasks_are_left_alone(
        self, make_task, make_entry, status
    ):
        task = make_task(status=status)
        make_entry(task)

        assert sweeper.find_stranded_tasks() == []

    def test_failed_tasks_join_the_sweep_only_when_opted_in(
        self, make_task, make_entry, settings
    ):
        task = make_task(status=EvalTaskStatus.FAILED)
        make_entry(task)

        settings.EVAL_TASK_SWEEP_RECOVER_FAILED = True

        assert [str(t.id) for t in sweeper.find_stranded_tasks()] == [str(task.id)]

    def test_the_longest_stranded_task_is_served_first_and_the_tick_is_capped(
        self, make_task, make_entry
    ):
        oldest = make_task()
        make_entry(oldest, age_seconds=86_400)
        newest = make_task()
        make_entry(newest, age_seconds=1)

        found = sweeper.find_stranded_tasks(limit=1)

        assert [str(t.id) for t in found] == [str(oldest.id)]

    def test_a_non_sweepable_task_cannot_hold_the_tick_slot(
        self, make_task, make_entry
    ):
        """A paused task never drains, so its entries' ``updated_at`` never
        advances and it sorts oldest on every tick for ever. If the per-tick cap
        were applied before the task-status filter it would own a slot
        permanently and the sweepable task behind it would never be reached."""
        parked = make_task(status=EvalTaskStatus.PAUSED)
        make_entry(parked, age_seconds=864_000)
        sweepable = make_task()
        make_entry(sweepable, age_seconds=1)

        found = sweeper.find_stranded_tasks(limit=1)

        assert [str(t.id) for t in found] == [str(sweepable.id)]

    def test_entries_whose_task_is_gone_cannot_hold_the_tick_slot(
        self, make_task, make_entry
    ):
        """Same shape with no task row at all: an entry pointing at an id the
        task table does not carry is never sweepable, so it must not be costed
        against the cap either."""
        orphan_host = make_task()
        orphan = make_entry(orphan_host, age_seconds=864_000)
        EvalLogger.all_objects.filter(id=orphan.id).update(
            eval_task_id=str(uuid.uuid4())
        )
        sweepable = make_task()
        make_entry(sweepable, age_seconds=1)

        found = sweeper.find_stranded_tasks(limit=1)

        assert [str(t.id) for t in found] == [str(sweepable.id)]

    def test_non_sweepable_tasks_cannot_fill_the_whole_cap(self, make_task, make_entry):
        """The starvation is permanent, not a one-slot rounding error: fill the
        cap with tasks the sweep refuses to act on and every tick returns an
        empty candidate list, silently, for as long as they exist."""
        for _ in range(3):
            make_entry(make_task(status=EvalTaskStatus.PAUSED), age_seconds=864_000)
        sweepable = make_task()
        make_entry(sweepable, age_seconds=432_000)

        found = sweeper.find_stranded_tasks(limit=3)

        assert [str(t.id) for t in found] == [str(sweepable.id)]


@pytest.mark.django_db
class TestRecoverTask:
    def test_a_task_with_no_live_workflow_is_restarted(
        self, make_task, make_entry, temporal
    ):
        task = make_task()
        make_entry(task)

        outcome = sweeper.recover_task(task, stale_running_seconds=7_200)

        assert outcome["restarted"] is True
        assert temporal["started"] == [(str(task.id), {"replace_existing": False})]

    def test_a_progressing_workflow_is_never_restarted(
        self, make_task, make_entry, temporal
    ):
        from tfc.temporal.eval_tasks.client import WF_PROGRESSING

        task = make_task()
        make_entry(task)
        temporal["verdict"] = WF_PROGRESSING

        outcome = sweeper.recover_task(task, stale_running_seconds=7_200)

        assert outcome["restarted"] is False
        assert temporal["started"] == []

    def test_a_stale_running_entry_is_reclaimed_even_under_a_live_workflow(
        self, make_task, make_entry, temporal
    ):
        """The whole point of putting the reap on a schedule: the workflow-start
        reaper can never reach a task whose workflow is still alive."""
        from tfc.temporal.eval_tasks.client import WF_PROGRESSING

        task = make_task()
        entry = make_entry(task, status=EvalEntryStatus.RUNNING, age_seconds=10_000)
        temporal["verdict"] = WF_PROGRESSING

        outcome = sweeper.recover_task(task, stale_running_seconds=7_200)

        entry.refresh_from_db()
        assert outcome["requeued"] == 1
        assert entry.status == EvalEntryStatus.PENDING
        assert entry.attempts == 1

    def test_a_freshly_claimed_entry_is_not_reclaimed(
        self, make_task, make_entry, temporal
    ):
        task = make_task()
        entry = make_entry(task, status=EvalEntryStatus.RUNNING, age_seconds=60)

        outcome = sweeper.recover_task(task, stale_running_seconds=7_200)

        entry.refresh_from_db()
        assert outcome["requeued"] == 0
        assert entry.status == EvalEntryStatus.RUNNING
        assert entry.attempts == 0

    def test_a_failed_task_is_made_pending_before_its_workflow_restarts(
        self, make_task, make_entry, temporal
    ):
        """``get_eval_task_state_activity`` reports a FAILED task inactive, so a
        restart that left the status alone would exit on its first check."""
        task = make_task(status=EvalTaskStatus.FAILED)
        make_entry(task)

        outcome = sweeper.recover_task(task, stale_running_seconds=7_200)

        task.refresh_from_db()
        assert outcome["restarted"] is True
        assert task.status == EvalTaskStatus.PENDING


@pytest.mark.django_db
class TestSweepActivity:
    def test_the_sweep_reports_counts_and_restarts_what_it_found(
        self, make_task, make_entry, temporal
    ):
        task = make_task()
        make_entry(task)

        result = sweeper.sweep_stranded_eval_tasks._original_func()

        assert result["candidates"] == 1
        assert result["restarted"] == 1
        assert result["errors"] == 0
        assert temporal["started"] == [(str(task.id), {"replace_existing": False})]

    def test_an_unreachable_temporal_does_not_restart_anything(
        self, make_task, make_entry, temporal
    ):
        """A describe that fails for any reason other than NOT_FOUND is the
        service being unreachable, not an answer. Restarting blind on it would
        terminate healthy workflows across the fleet."""
        task = make_task()
        make_entry(task)
        temporal["verdict"] = RuntimeError("temporal unreachable")

        result = sweeper.sweep_stranded_eval_tasks._original_func()

        assert result["errors"] == 1
        assert result["restarted"] == 0
        assert temporal["started"] == []

    def test_the_sweep_can_be_turned_off_without_a_deploy_of_its_own(
        self, make_task, make_entry, temporal, settings
    ):
        """A Temporal pause is the immediate lever but not a durable one:
        ``register_temporal_schedules`` runs on every backend container start
        and rebuilds each schedule's state from config, so the pause is undone
        by the next deploy, restart or scale-up. The setting is the rollback
        that survives one, and it has to report itself — an operator must be
        able to tell a disabled sweep from a fleet with nothing stranded."""
        task = make_task()
        make_entry(task)

        settings.EVAL_TASK_SWEEP_MAX_TASKS = 0
        result = sweeper.sweep_stranded_eval_tasks._original_func()

        assert result["disabled"] is True
        assert result["candidates"] == 0
        assert temporal["described"] == []
        assert temporal["started"] == []

    def test_one_tasks_failure_does_not_cost_the_others_their_recovery(
        self, make_task, make_entry, temporal, monkeypatch
    ):
        from tfc.temporal.eval_tasks import client

        first = make_task()
        make_entry(first, age_seconds=86_400)
        second = make_task()
        make_entry(second, age_seconds=1)

        def _describe(task_id):
            if str(task_id) == str(first.id):
                raise RuntimeError("boom")
            return client.WF_CLOSED

        monkeypatch.setattr(client, "describe_eval_task_workflow_sync", _describe)

        result = sweeper.sweep_stranded_eval_tasks._original_func()

        assert result["errors"] == 1
        assert result["restarted"] == 1
        assert [started[0] for started in temporal["started"]] == [str(second.id)]


def test_the_sweep_is_actually_scheduled_and_its_activity_is_registered():
    """A recovery job nobody runs is the defect it is meant to fix. The schedule
    and the activity registration are separate wires — pin both."""
    from tfc.temporal.common.registry import TEMPORAL_ACTIVITY_MODULES
    from tfc.temporal.schedules.tracer import TRACER_SCHEDULES

    scheduled = {
        config.schedule_id: config
        for config in TRACER_SCHEDULES
        if config.schedule_id == "sweep-stranded-eval-tasks"
    }
    config = scheduled["sweep-stranded-eval-tasks"]

    assert config.activity_name == sweeper.sweep_stranded_eval_tasks._activity_name
    assert config.queue == "tasks_s"
    assert 0 < config.interval_seconds <= 600
    assert "tracer.tasks.eval_task_sweeper" in TEMPORAL_ACTIVITY_MODULES


def test_the_mirrored_run_entry_ceiling_matches_the_workflow():
    """The settings module declares the workflow's run-entry ceiling and retry
    count by value, because the workflow module cannot be imported at
    settings-load time. Every bound below is derived from those two constants,
    so this is the test that makes the copy honest: change the workflow and
    this fails rather than the bounds silently describing a ceiling that moved.
    """
    from tfc.settings.runtime_setting_specs import (
        RUN_ENTRY_CEILING_SECONDS,
        RUN_ENTRY_MAX_ATTEMPTS,
    )
    from tfc.temporal.eval_tasks.workflows import (
        _RUN_ENTRY_TIMEOUT,
        RUN_ENTRY_RETRY_POLICY,
    )

    assert RUN_ENTRY_CEILING_SECONDS == _RUN_ENTRY_TIMEOUT.total_seconds()
    assert RUN_ENTRY_MAX_ATTEMPTS == RUN_ENTRY_RETRY_POLICY.maximum_attempts


def test_sweep_stale_threshold_exceeds_a_live_entrys_longest_run():
    """The scheduled reap runs beside live workflows, so its threshold must stay
    above the longest a legitimately running entry can live: the workflow's
    run-entry start-to-close ceiling times its retry attempts. Below that bound
    the sweep requeues an entry a worker is still evaluating and spends one of
    that entry's three reclaims on it.

    Asserted against the spec's **minimum**, not against the live setting. The
    setting resolves from the process environment through ``load_numeric_settings``
    and is bounded only by the spec, so a test that reads the running value
    passes on the default and says nothing about the range an operator can
    actually configure.
    """
    from tfc.settings.runtime_setting_specs import RUNTIME_NUMERIC_SETTING_SPECS
    from tfc.temporal.eval_tasks.workflows import (
        _RUN_ENTRY_TIMEOUT,
        RUN_ENTRY_RETRY_POLICY,
    )

    longest_run = (
        _RUN_ENTRY_TIMEOUT.total_seconds() * RUN_ENTRY_RETRY_POLICY.maximum_attempts
    )
    spec = RUNTIME_NUMERIC_SETTING_SPECS["EVAL_TASK_SWEEP_STALE_RUNNING_SECONDS"]

    assert spec.minimum > longest_run
    assert spec.default > longest_run


def test_the_activity_ceiling_leaves_room_for_the_engines_own_wall():
    """The engine's wall bounds one evaluation; the activity ceiling has to
    cover a whole entry — telemetry loads, media, a composite's sub-evals — so
    it must be the looser of the two, or the activity would abandon runs the
    wall was still willing to allow. Asserted against the spec's **maximum**,
    for the same reason as above: the declared range is what an operator gets
    to choose from."""
    from tfc.settings.runtime_setting_specs import RUNTIME_NUMERIC_SETTING_SPECS
    from tfc.temporal.eval_tasks.workflows import _RUN_ENTRY_TIMEOUT

    spec = RUNTIME_NUMERIC_SETTING_SPECS["EVAL_RUN_WALL_SECONDS"]

    assert _RUN_ENTRY_TIMEOUT.total_seconds() >= spec.maximum
    assert _RUN_ENTRY_TIMEOUT.total_seconds() > spec.default
