"""Every sequence of Users walk hops ends, and publishes each user once, in order.

The walk's cursor promises that every user with a matching row at or after
its coverage is decided. Two ways of breaking that promise stalled the list:

* a user whose finishing replay always outlasts the server cap was retried,
  and stopped, on every hop, so the cursor never moved (B1);
* a user rejected by its replay above the coverage, re-witnessed by a lower
  row, was admitted again, and when the budget refused its replay the
  coverage moved back UP to it, so hops cycled (B2).

These tests drive the real walk and manager through the scripted ``World`` /
``Engine`` of ``test_users_matching_walk`` (and, for the server cap, through
the real ``V2AnalyticsQueryService`` and ``ClickHouseClient`` over a recording
native driver). The property test generates worlds from fixed seeds and
follows every cursor to the end.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from tracer.services import users_matching_walk as walk
from tracer.services.clickhouse import read_budget
from tracer.services.clickhouse.read_budget import ReadDeadline, ReadDeadlineExceeded
from tracer.tests.test_users_matching_walk import (
    PROJECT,
    SERVICE,
    WINDOW_END,
    WINDOW_START,
    Engine,
    World,
    _from_us,
    _manager,
    _names,
    _NativeDriver,
    _never_seed,
    _page,
    _real_service_page,
    _signed_cursor,
    _tied_world,
    kind_of,
    minutes_before_end,
)

pytestmark = pytest.mark.unit

TICK = timedelta(microseconds=1)


def _uid(n: int) -> str:
    return str(uuid.UUID(int=n))


def _add(
    world: World,
    uid_int: int,
    alias_ints: list[int],
    key: datetime | None,
    raw: list[tuple[datetime, int]],
    *,
    curated: bool = True,
) -> str:
    """A user with explicit ids, so survivors and aliases interleave in id order.

    ``raw`` is ``((moment, identity index), ...)``: which of the user's ids
    carries each witnessed row.
    """
    uid = _uid(uid_int)
    ids = (uid, *(_uid(alias) for alias in alias_ints))
    world.users[uid] = {
        "key": key,
        "cost": 1.0,
        "aliases": ids,
        "curated": curated,
        "name": uid,
        "typed_values": [("string", '"Gold"')],
    }
    for identity in ids:
        world.canonical[identity] = uid
    for moment, index in raw:
        world.raw.append((moment, ids[index % len(ids)]))
    return uid


def _expected(world: World) -> list[str]:
    """Members in page order: newest matching activity, then id, descending."""

    members = [
        (user["key"], uid)
        for uid, user in world.users.items()
        if user["key"] is not None and user["curated"]
    ]
    return [uid for _key, uid in sorted(members, reverse=True)]


def _check_uncapped(
    replays: list[tuple[tuple[str, ...], bool, bool]],
    reasons: list[str],
    *,
    replay_info: list[tuple[float, int]] | None = None,
    finish_wall_ms: float | None = None,
) -> None:
    """At most one replay without a cap a request, for one user, and why.

    ``replays`` is one request's ``(ids, capped, stopped)``, in order, and
    ``reasons`` what the walk logged for its uncapped replay:
    ``own_stopped``, right after that user's own capped replay was stopped;
    ``batch_stopped``, right after a stopped batch the user led, when the
    budget could not afford the user's own attempt; ``refused``, when the
    search had spent the analytics wall and its capped attempt was never
    sent. With ``replay_info`` (when each replay was sent, and its rows)
    it also holds that no replay before it returned a row: an uncapped
    replay comes only while the page has published nothing.
    """
    uncapped = [i for i, (_ids, capped, _stopped) in enumerate(replays) if not capped]
    assert len(uncapped) == len(reasons) <= 1, (replays, reasons)
    for index, reason in zip(uncapped, reasons, strict=True):
        (user,) = replays[index][0]
        if replay_info is not None:
            assert not any(rows for _at, rows in replay_info[:index]), replay_info
        if reason == "own_stopped":
            assert index > 0 and replays[index - 1] == ((user,), True, True), replays
        elif reason == "batch_stopped":
            before, capped, stopped = replays[index - 1]
            assert capped and stopped and len(before) > 1, replays
            assert before[0] == user, replays
        else:
            assert reason == "refused", reason
            if replay_info is not None:
                assert replay_info[index][0] >= finish_wall_ms - 25, replay_info


def _check_singly(finishing: list[tuple[str, tuple[str, ...], bool, bool]]) -> None:
    """Once a finishing statement is stopped, every later replay is one user."""

    stops = [
        i for i, (_kind, _ids, _capped, stopped) in enumerate(finishing) if stopped
    ]
    if stops:
        later = finishing[stops[0] + 1 :]
        assert all(len(ids) == 1 for kind, ids, _c, _s in later if kind == "replay"), (
            finishing
        )


class _Clock:
    """The monotonic clock of the walls, advanced by each scripted statement."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now

    def spend(self, ms: float) -> None:
        self.now += ms / 1000.0


@contextmanager
def _scripted_clock(clock: _Clock):
    """Run the page wall, the finish deadline and statement timing on ``clock``."""

    fake = SimpleNamespace(monotonic=clock.monotonic)
    with patch.object(read_budget, "time", fake), patch.object(walk, "time", fake):
        yield clock


@contextmanager
def _shipped_walls():
    with (
        patch.object(walk, "USER_LIST_PAGE_WALL_MS", 5_000),
        patch.object(walk, "USER_LIST_WALK_FINISH_WALL_MS", 30_000),
    ):
        yield


# The statements that finish a materialisation after the replay; each names
# the users it reads in ``candidate_end_user_ids``.
FINISHING = frozenset(
    {"replay", "relation", "session_metrics", "span_metrics", "evals"}
)


def _statement_kind(query: str, params: dict | None) -> str:
    """``kind_of``, told apart for every finishing statement."""

    params = params or {}
    if "latest_relation_candidate_spans AS" in query:
        return "relation"
    if "eval_eu_ids" in params:
        return "evals"
    if "session_rows AS" in query:
        return "session_metrics"
    if "candidate_users AS" in query:
        return "replay"
    if "candidate_end_user_ids" in params:
        return "span_metrics"
    return kind_of(query)


class _CappedEngine(Engine):
    """A scripted server on which a heavy user's replay outlasts any cap.

    A replay that carries a heavy user, or more than ``batch_limit`` users,
    and asks the server to stop at a cap is stopped there
    (``ReadDeadlineExceeded``, as the service maps code 159 under a cap); the
    same replay without a cap runs to completion; so do the relation,
    metrics and evals statements that finish it. Records every replay as
    ``(ids, capped, stopped)``, and every finishing statement as ``(kind,
    ids, capped, stopped)``. With a ``clock``, every statement spends its
    time on it: 1 ms, a stopped statement its cap, and a slice what
    ``slice_ms(width, returned_rows)`` says. A slice that costs more than the
    cap it was sent with is stopped there; one sent without a cap runs to
    the end whatever it costs. Records every slice as ``(width, cap,
    stopped)``.
    """

    def __init__(
        self,
        world: World,
        heavy: frozenset[str] = frozenset(),
        batch_limit: int | None = None,
        clock: _Clock | None = None,
        slice_ms: Callable[[timedelta, bool], float] | None = None,
    ) -> None:
        super().__init__(world)
        self.heavy = heavy
        self.batch_limit = batch_limit
        self.clock = clock
        self.slice_ms = slice_ms
        self.replays: list[tuple[tuple[str, ...], bool, bool]] = []
        # Per replay: ms into the request when it was sent, rows it returned.
        self.replay_info: list[tuple[float, int]] = []
        self.finishing: list[tuple[str, tuple[str, ...], bool, bool]] = []
        self.slices: list[tuple[timedelta, float | None, bool]] = []
        # Per slice: ms into the request when it was sent.
        self.slice_at: list[float] = []
        # Every statement fails in the transport: the server is unreachable.
        self.outage = False
        self.start = clock.now if clock is not None else 0.0

    def _elapsed_ms(self) -> float:
        return (self.clock.now - self.start) * 1000.0 if self.clock else 0.0

    def execute_ch_query(
        self,
        query,
        params=None,
        timeout_ms=None,
        settings=None,
        *,
        server_execution_cap_ms=None,
    ):
        kind = _statement_kind(query, params)
        if self.outage:
            self.calls.append(query)
            raise ReadDeadlineExceeded("the server is unreachable")
        if kind in FINISHING:
            ids = tuple((params or {})["candidate_end_user_ids"])
            capped = server_execution_cap_ms is not None
            stopped = capped and (
                bool(self.heavy.intersection(ids))
                or (self.batch_limit is not None and len(ids) > self.batch_limit)
            )
            self.finishing.append((kind, ids, capped, stopped))
            if kind == "replay":
                self.replays.append((ids, capped, stopped))
                self.replay_info.append((self._elapsed_ms(), 0))
            if stopped:
                self.calls.append(query)
                self.settings.append(settings)
                self.timeouts.append(timeout_ms)
                self.caps.append(server_execution_cap_ms)
                if self.clock is not None:
                    self.clock.spend(server_execution_cap_ms)
                raise ReadDeadlineExceeded("ClickHouse statement exceeded its cap")
            if kind != "replay":
                self.calls.append(query)
                self.settings.append(settings)
                self.timeouts.append(timeout_ms)
                self.caps.append(server_execution_cap_ms)
                if self.clock is not None:
                    self.clock.spend(1.0)
                # Every user matches the relation filter; metrics and evals
                # carry nothing a filter reads.
                data = (
                    [{"end_user_id": uid} for uid in ids] if kind == "relation" else []
                )
                return SimpleNamespace(data=data, query_time_ms=1.0)
        if kind == "slice":
            self.slice_at.append(self._elapsed_ms())
        result = super().execute_ch_query(
            query,
            params,
            timeout_ms,
            settings,
            server_execution_cap_ms=server_execution_cap_ms,
        )
        if kind == "replay":
            self.replay_info[-1] = (self.replay_info[-1][0], len(result.data or ()))
        cost = 1.0
        if kind == "slice":
            width = TICK * (params["slice_end_us"] - params["slice_start_us"])
            if self.slice_ms is not None:
                cost = self.slice_ms(width, bool(result.data))
            cap = server_execution_cap_ms
            stopped = cap is not None and cost > cap
            self.slices.append((width, cap, stopped))
            if stopped:
                if self.clock is not None:
                    self.clock.spend(cap)
                raise ReadDeadlineExceeded("ClickHouse statement exceeded its cap")
        if self.clock is not None:
            result.query_time_ms = cost
            self.clock.spend(cost)
        return result


def _every_slice(ms: float) -> Callable[[timedelta, bool], float]:
    """Every slice that returns rows costs ``ms``; an empty one, 1 ms."""

    return lambda _width, rows: ms if rows else 1.0


def _per_hour(ms: float) -> Callable[[timedelta, bool], float]:
    """A slice that returns rows costs ``ms`` per hour of its width."""

    return lambda width, rows: (
        max(1.0, ms * width / timedelta(hours=1)) if rows else 1.0
    )


def _follow(
    world: World,
    *,
    page_size: int,
    max_hops: int,
    max_statements: int | None = None,
    heavy: frozenset[str] = frozenset(),
    mutate=None,
    slice_ms: Callable[[timedelta, bool], float] | None = None,
    keys: int = 0,
    finish: int = 1,
    outage_every: int | None = None,
) -> tuple[list[str], int]:
    """Follow the cursor to the end; returns the published names and the hops.

    Every hop must keep the coverage where it was or lower it, send no more
    statements than the walk's own budget for the page (every finishing
    statement included), and send at most one replay without a server cap,
    for one user, right after a capped replay it led was stopped. In a
    static world a repeated ``(cursor, seen rows)`` is a livelock, and no two
    hops in a row may both make no progress: publish a user, lower the
    coverage, or lower the decided position. The walls run on a scripted
    clock (``_CappedEngine``); the page shows ``keys`` attribute columns
    besides the filtered one, and the columns and filters that make
    ``finish`` finishing statements (``FINISH_SHAPES``). Every uncapped
    replay and slice must have the reason the walk logs for it
    (``_check_uncapped``, ``_check_uncapped_slices``), and once a finishing
    statement is stopped every later replay carries one user
    (``_check_singly``). With ``outage_every``, every such hop finds the
    server unreachable: it must not raise the coverage, and it is left out of
    the progress and livelock checks. ``mutate(world, published, cursor)``
    runs between hops.
    """
    with _scripted_clock(_Clock()) as clock:
        return _follow_on(
            world,
            clock=clock,
            page_size=page_size,
            max_hops=max_hops,
            max_statements=max_statements,
            heavy=heavy,
            mutate=mutate,
            slice_ms=slice_ms,
            keys=keys,
            finish=finish,
            outage_every=outage_every,
        )


def _follow_on(
    world: World,
    *,
    clock: _Clock,
    page_size: int,
    max_hops: int,
    max_statements: int | None,
    heavy: frozenset[str],
    mutate,
    slice_ms: Callable[[timedelta, bool], float] | None,
    keys: int,
    finish: int,
    outage_every: int | None,
) -> tuple[list[str], int]:
    budget = walk._statement_budget(_keyed_manager(keys, finish))
    assert budget >= (max_statements or walk.USER_LIST_WALK_MAX_STATEMENTS)
    names: list[str] = []
    seen_states: set = set()
    coverage = WINDOW_END
    position: tuple = (None, None)
    stalled = False
    cursor = None
    finish_wall = walk.USER_LIST_WALK_FINISH_WALL_MS
    for hop in range(1, max_hops + 1):
        engine = _CappedEngine(world, heavy, clock=clock, slice_ms=slice_ms)
        engine.outage = outage_every is not None and hop % outage_every == 0
        with capture_logs() as logs:
            read = _keyed_page(
                page_size=page_size,
                cursor=cursor,
                engine=engine,
                keys=keys,
                finish=finish,
            )
        names.extend(_names(read))
        assert len(engine.calls) <= budget, (hop, len(engine.calls))
        if not engine.outage:
            _check_uncapped(
                engine.replays,
                _logged(logs, "users_matching_walk_uncapped_finish"),
                replay_info=engine.replay_info,
                finish_wall_ms=finish_wall,
            )
            _check_singly(engine.finishing)
            _check_uncapped_slices(
                engine.slices,
                _logged(logs, "users_matching_walk_uncapped_slice"),
                slice_at=engine.slice_at,
                finish_wall_ms=finish_wall,
            )
        if not read.has_more:
            return names, hop
        order = tuple(read.checkpoint_order)
        assert order[3] <= coverage, f"coverage moved up at hop {hop}: {order}"
        if engine.outage:
            assert _names(read) == [], hop
            cursor = _signed_cursor(read)
            continue
        if mutate is None:
            state = (order, read.seen_rows)
            assert state not in seen_states, f"livelock at hop {hop}: {state}"
            seen_states.add(state)
            progressed = (
                bool(_names(read))
                or order[3] < coverage
                or _lower_position(order[1:3], position)
            )
            assert progressed or not stalled, f"two hops without progress at {hop}"
            stalled = not progressed
        coverage, position = order[3], order[1:3]
        cursor = _signed_cursor(read)
        if mutate is not None:
            mutate(world, set(names), order)
    raise AssertionError(f"no end after {max_hops} hops; published {len(names)}")


ANNOTATION_FILTER = {
    "column_id": str(uuid.UUID(int=77)),
    "filter_config": {
        "filter_type": "number",
        "filter_op": "greater_than",
        "filter_value": 3,
        "col_type": "ANNOTATION",
    },
}
# Finishing statements per materialisation: the columns and relation filter
# that make them. A session metric, a span metric, a relation filter and an
# eval column each add one statement after the replay.
FINISH_SHAPES = {
    1: ((), False),
    2: (("num_sessions",), False),
    3: (("avg_session_duration", "avg_trace_latency"), False),
    4: (("avg_session_duration", "avg_trace_latency"), True),
    5: (("avg_session_duration", "avg_trace_latency", "eval_score"), True),
}


def _keyed_manager(keys: int, finish: int = 1):
    """The page's manager: ``keys`` attribute columns besides ``tag``, and the
    columns and filters that make ``finish`` finishing statements."""

    from tracer.services.users_list_manager import UsersListManager

    base = _manager()
    columns, relation = FINISH_SHAPES[finish]
    manager = UsersListManager(
        organization_id=base.organization_id,
        allowed_project_ids=list(base.scoped_project_ids),
        project_id=base.project_id,
        filters=[*base.filters, *([ANNOTATION_FILTER] if relation else [])],
        requested_columns=list(columns),
        attribute_keys=[f"k{n:03d}" for n in range(keys)],
    )
    assert walk._materialisation_statement_count(manager) == finish
    return manager


@contextmanager
def _eval_configs():
    """One eval config in the project, so an eval column sends its statement."""

    from unittest.mock import MagicMock

    configs = MagicMock()
    configs.filter.return_value.values_list.return_value = [
        (PROJECT, str(uuid.UUID(int=88)))
    ]
    with patch(
        "tracer.models.custom_eval_config.CustomEvalConfig.no_workspace_objects",
        configs,
    ):
        yield


def _keyed_page(*, page_size: int, cursor, engine: Engine, keys: int, finish: int = 1):
    manager = _keyed_manager(keys, finish)
    with (
        patch(SERVICE, return_value=engine),
        patch.object(manager, "_read_dimension_candidates", side_effect=_never_seed),
        _eval_configs(),
    ):
        return manager.list_cursor_payload(page_size=page_size, cursor=cursor)


def _check_uncapped_slices(
    slices: list[tuple[timedelta, float | None, bool]],
    reasons: list[str],
    *,
    slice_at: list[float] | None = None,
    finish_wall_ms: float | None = None,
) -> None:
    """At most one slice without a cap a request, no wider than the least, and why.

    ``reasons`` is what the walk logged: ``narrowest_stopped``, right after
    a capped slice of the least width was stopped; ``no_budget``, right
    after a wider one was stopped when the budget could not afford a
    narrower retry; ``wall_spent``, once the analytics wall had nothing
    left (``slice_at`` says when each slice was sent).
    """
    uncapped = [i for i, (_width, cap, _stopped) in enumerate(slices) if cap is None]
    assert len(uncapped) == len(reasons) <= 1, (slices, reasons)
    least = walk.USER_LIST_WALK_MIN_SLICE
    for index, reason in zip(uncapped, reasons, strict=True):
        assert slices[index][0] <= least, slices
        if reason == "wall_spent":
            if slice_at is not None:
                assert slice_at[index] >= finish_wall_ms - 25, slice_at
            continue
        width, cap, stopped = slices[index - 1]
        assert index > 0 and cap is not None and stopped, slices
        if reason == "narrowest_stopped":
            assert width <= least, slices
        else:
            assert reason == "no_budget" and width > least, (reason, slices)


def _logged(logs: list[dict], event: str) -> list[str]:
    return [entry["reason"] for entry in logs if entry["event"] == event]


def _lower_position(new: tuple, old: tuple) -> bool:
    """Whether cursor position ``(last_key, last_id)`` ``new`` is below ``old``."""

    if new[0] is None:
        return False
    return old[0] is None or tuple(new) < tuple(old)


# --------------------------------------------------------------------------
# B1: a user whose finishing replay always outlasts the server cap.
# --------------------------------------------------------------------------


class _SlowUserDriver(_NativeDriver):
    """The server stops a replay carrying a heavy user whenever it is told to.

    Heavy, not broken: the same replay sent without ``max_execution_time``
    completes. Also stops, under a cap, any replay of more than
    ``batch_limit`` users. Records every replay as ``(ids, capped, stopped)``
    and each cap it was sent with.
    """

    def __init__(
        self, engine: Engine, heavy: set[str], batch_limit: int | None = None
    ) -> None:
        super().__init__(engine)
        self.heavy = frozenset(heavy)
        self.batch_limit = batch_limit
        self.replays: list[tuple[tuple[str, ...], bool, bool]] = []
        self.caps: list[float] = []

    def execute(self, query, params=None, *, with_column_types=False, settings=None):
        kind = _statement_kind(query, params)
        if kind == "replay":
            ids = tuple((params or {}).get("candidate_end_user_ids", ()))
            cap = float((settings or {}).get("max_execution_time") or 0)
            stopped = cap > 0 and (
                bool(self.heavy.intersection(ids))
                or (self.batch_limit is not None and len(ids) > self.batch_limit)
            )
            self.replays.append((ids, cap > 0, stopped))
            self.caps.append(cap)
            if stopped:
                from clickhouse_driver.errors import ErrorCodes, ServerException

                self.sent.append(("replay", settings))
                raise ServerException(
                    "Timeout exceeded: elapsed 8.0 seconds, maximum: 8",
                    code=ErrorCodes.TIMEOUT_EXCEEDED,
                )
        if kind not in FINISHING:
            return super().execute(
                query, params, with_column_types=with_column_types, settings=settings
            )
        self.sent.append((kind, settings))
        result = self.engine.execute_ch_query(query, params)
        data = list(result.data or ())
        columns = list(getattr(result, "columns", None) or (data[0] if data else ()))
        rows = [tuple(row.get(name) for name in columns) for row in data]
        return rows, [(name, "String") for name in columns]


def _real_service_hops(world: World, driver: _SlowUserDriver, max_hops: int):
    """Every page through the real service and client: ``(names, replays)``."""

    hops = []
    cursor = None
    coverage = WINDOW_END
    for _ in range(max_hops):
        before = len(driver.replays)
        with capture_logs() as logs:
            read = _real_service_page(world, driver, cursor=cursor)
        replays = driver.replays[before:]
        reasons = _logged(logs, "users_matching_walk_uncapped_finish")
        # The shipped budget always affords a user's own capped attempt.
        assert set(reasons) <= {"own_stopped", "refused"}, reasons
        _check_uncapped(replays, reasons)
        hops.append((_names(read), replays))
        if not read.has_more:
            return hops
        assert read.checkpoint_order[3] <= coverage, read.checkpoint_order
        coverage = read.checkpoint_order[3]
        cursor = _signed_cursor(read)
    raise AssertionError(f"no end after {max_hops} hops: {[h[0] for h in hops]}")


def test_a_sole_user_whose_replay_always_outlasts_the_cap_is_published():
    """The page decides its head-of-line user without the cap, once."""

    world = World()
    uid = world.user(1, key=minutes_before_end(3), raw=(minutes_before_end(3),))
    driver = _SlowUserDriver(_CappedEngine(world), {uid})

    hops = _real_service_hops(world, driver, max_hops=3)

    assert [names for names, _replays in hops] == [["user-1"]]
    # Stopped under the server cap, then decided with none.
    assert driver.replays == [((uid,), True, True), ((uid,), False, False)]
    assert 0 < driver.caps[0] <= 8.0 and driver.caps[1] == 0


@pytest.mark.parametrize(
    ("heavy_rank", "pages"),
    [
        (1, [[1, 2, 3, 4, 5]]),
        (3, [[1, 2], [3, 4, 5]]),
        (5, [[1, 2, 3, 4], [5]]),
    ],
)
def test_a_heavy_user_never_blocks_the_users_ranked_around_it(heavy_rank, pages):
    """The five users share one replay batch, and the server stops it.

    The page replays them one at a time, capped: the users ahead of the heavy
    user publish on the first page, the heavy user's own replay is stopped
    and ends that page, and the next page decides it without the cap (it has
    published nothing yet) and goes on to the users behind it.
    """
    world = World()
    ids = [
        world.user(n, key=minutes_before_end(n), raw=(minutes_before_end(n),))
        for n in range(1, 6)
    ]
    driver = _SlowUserDriver(_CappedEngine(world), {ids[heavy_rank - 1]})

    hops = _real_service_hops(world, driver, max_hops=4)

    assert [names for names, _replays in hops] == [
        [f"user-{n}" for n in page] for page in pages
    ]
    assert [ids for ids, capped, _stopped in driver.replays if not capped] == [
        (ids[heavy_rank - 1],)
    ]


def test_a_batch_that_outlasts_the_cap_publishes_its_users_one_at_a_time():
    """No user is heavy alone, but any replay of more than three users is.

    Every page's batch is stopped; its users then replay one at a time under
    the cap, until the statement budget ends the page. No statement is ever
    sent without the cap.
    """
    world = World()
    for n in range(1, 41):
        world.user(n, key=minutes_before_end(n), raw=(minutes_before_end(n),))
    driver = _SlowUserDriver(_CappedEngine(world), set(), batch_limit=3)

    hops = _real_service_hops(world, driver, max_hops=4)

    assert [name for names, _replays in hops for name in names] == [
        f"user-{n}" for n in range(1, 41)
    ]
    assert len(hops[0][0]) > 1
    assert all(capped for _ids, capped, _stopped in driver.replays)


def test_metric_columns_are_charged_one_statement_per_metric_group():
    """A session metric and a span metric column are two metrics statements.

    Through the real service and client. Every replay of more than three
    users is stopped, so the page materialises one user at a time, and each
    materialisation sends the replay and both metrics statements. They were
    charged as one, and a request sent 30-31 statements at a budget of 24;
    every request now stays within its budget.
    """
    from tracer.services.clickhouse.client import ClickHouseClient
    from tracer.services.users_list_manager import UsersListManager

    world, expected = _spread_world(30, 5)
    driver = _SlowUserDriver(_CappedEngine(world), set(), batch_limit=3)
    base = _manager()

    def manager():
        return UsersListManager(
            organization_id=base.organization_id,
            allowed_project_ids=list(base.scoped_project_ids),
            project_id=base.project_id,
            filters=base.filters,
            requested_columns=["avg_session_duration", "avg_trace_latency"],
            attribute_keys=[],
        )

    assert walk._materialisation_statement_count(manager()) == 3
    budget = walk._statement_budget(manager())
    names, cursor, per_request = [], None, []
    for _hop in range(12):
        client = ClickHouseClient(host="localhost", port=39999, pool_size=1)
        page = manager()
        before = len(driver.sent)
        with (
            patch.object(client, "_get_client", return_value=driver),
            patch.object(client, "_return_client"),
            patch(
                "tracer.services.clickhouse.v2.query_service.get_v2_query_client",
                return_value=client,
            ),
            patch.object(page, "_read_dimension_candidates", side_effect=_never_seed),
        ):
            read = page.list_cursor_payload(page_size=25, cursor=cursor)
        per_request.append(driver.sent[before:])
        names.extend(_names(read))
        if not read.has_more:
            break
        cursor = _signed_cursor(read)

    assert names == expected
    assert all(len(sent) <= budget for sent in per_request), [
        len(sent) for sent in per_request
    ]
    kinds = [kind for sent in per_request for kind, _settings in sent]
    assert kinds.count("session_metrics") == kinds.count("span_metrics") > 0


def test_a_heavy_user_inside_a_large_tie_does_not_stall_the_instant():
    world, expected = _tied_world(601)
    heavy = next(u for u, v in world.users.items() if v["name"] == expected[59])
    driver = _SlowUserDriver(_CappedEngine(world), {heavy})

    hops = _real_service_hops(world, driver, max_hops=30)

    assert [name for names, _replays in hops for name in names] == expected


def test_a_finish_the_server_stops_after_the_page_published_carries_the_rest():
    """Once the page has users to publish, a stopped finish ends it, bounded.

    User 2's replay always outlasts the cap. The batch of both is stopped;
    user 1 then replays alone, capped, and publishes; user 2's own capped
    replay is stopped, and the page is published as it stands, degraded,
    with user 2 carried in the cursor. Nothing is sent without the cap.
    """
    world = World()
    cheap = world.user(1, key=minutes_before_end(3), raw=(minutes_before_end(3),))
    heavy = world.user(2, key=minutes_before_end(5), raw=(minutes_before_end(5),))
    driver = _SlowUserDriver(_CappedEngine(world), {heavy})

    first = _real_service_page(world, driver)

    assert _names(first) == ["user-1"]
    assert first.payload["query_status"] == "degraded"
    assert first.has_more is True
    assert driver.replays == [
        ((cheap, heavy), True, True),
        ((cheap,), True, False),
        ((heavy,), True, True),
    ]
    assert first.checkpoint_order[3] > minutes_before_end(5)

    resumed = _real_service_page(world, driver, cursor=_signed_cursor(first))
    assert _names(resumed) == ["user-2"]
    assert driver.replays[3:] == [((heavy,), True, True), ((heavy,), False, False)]


def test_a_finish_the_analytics_wall_refuses_is_decided_without_the_cap():
    """The search spent the analytics wall, so the capped replay is never sent.

    Nothing is left of the finish deadline: the capped replay is refused at
    admission, on the client, before it reaches the server. That stalls the
    cursor just as a replay the server stops does, and it ends the same way:
    the page has published nothing, so it decides its head-of-line user
    without a cap.
    """
    world = World()
    uid = world.user(1, key=minutes_before_end(3), raw=(minutes_before_end(3),))
    driver = _SlowUserDriver(_CappedEngine(world), set())
    spent = ReadDeadline.start(1, enforce_on_server=True)

    with patch.object(walk, "_finish_deadline", return_value=spent):
        read = _real_service_page(world, driver)

    assert _names(read) == ["user-1"]
    assert driver.replays == [((uid,), False, False)]
    assert driver.caps == [0]
    assert read.has_more is False


# --------------------------------------------------------------------------
# B2: a replay-rejected user re-witnessed below the resume coverage.
# --------------------------------------------------------------------------


def _unique_time_world(rng: random.Random, n: int, reject_rate: float) -> World:
    """No ties: every witnessed row at its own microsecond."""

    world = World()
    span = int((WINDOW_END - WINDOW_START) / TICK)
    used: set[int] = set()

    def moment() -> datetime:
        while True:
            value = rng.randrange(span)
            if value not in used:
                used.add(value)
                return WINDOW_START + TICK * value

    for index in range(n):
        key = None if rng.random() < 0.15 else moment()
        raw = [key] if key is not None else []
        raw += [moment() for _ in range(rng.choice([0, 1, 2, 3]))]
        if not raw:
            raw = [moment()]
        _add(
            world,
            10 + index,
            [],
            key,
            [(m, 0) for m in raw],
            curated=rng.random() > reject_rate,
        )
    return world


def test_a_rejected_user_found_again_below_the_coverage_does_not_raise_it():
    """Small budget, no ties, 35% of members rejected by their replay."""

    rng = random.Random(36)
    world = _unique_time_world(rng, rng.choice([4, 8, 15, 30]), reject_rate=0.35)
    with (
        patch.object(walk, "USER_LIST_WALK_SLICE_USER_LIMIT", 2),
        patch.object(walk, "USER_LIST_WALK_MAX_STATEMENTS", 8),
        patch.object(walk, "USER_LIST_WALK_CERTIFY_BATCH_SIZE", 2),
    ):
        names, _hops = _follow(world, page_size=3, max_hops=60, max_statements=8)

    assert names == _expected(world)


def test_a_dense_rejecting_world_ends_at_shipped_limits():
    """2,000 users, no ties, most rejected by the replay, a metric column shown."""

    assert walk.USER_LIST_WALK_SLICE_USER_LIMIT == 200
    assert walk.USER_LIST_WALK_MAX_STATEMENTS == 24
    assert walk.USER_LIST_WALK_CERTIFY_BATCH_SIZE == 25
    rng = random.Random(1)
    n = rng.choice([2000, 4000])
    world = _unique_time_world(rng, n, reject_rate=rng.choice([0.7, 0.9, 0.97]))
    names, _hops = _follow(world, page_size=25, max_hops=400, finish=2)

    assert names == _expected(world)


# --------------------------------------------------------------------------
# P2: a populated slice whose own statement outlasts the page wall.
# --------------------------------------------------------------------------


def _spread_world(users: int, every_minutes: int) -> tuple[World, list[str]]:
    world = World()
    for n in range(1, users + 1):
        moment = minutes_before_end(every_minutes * n)
        world.user(n, key=moment, raw=(moment,))
    return world, [f"user-{n}" for n in range(1, users + 1)]


def test_a_slice_that_outlasts_the_page_wall_still_decides_its_first_batch():
    """Every slice that returns rows takes 6 s against the 5 s page wall.

    The slice statement is admitted inside the wall and runs past it. Its
    survivor statement was then refused, the slice discarded whole, and the
    next request read the same slice again, so the list never got past it.
    A request that has decided nothing yet finishes that slice's first batch
    against the analytics wall: every request publishes, and the list ends.
    """
    world, expected = _spread_world(60, 2)

    with _shipped_walls():
        names, hops = _follow(
            world, page_size=25, max_hops=8, slice_ms=_every_slice(6_000)
        )

    assert names == expected
    assert hops <= 6


def _stopped_slice_hops(world, slice_ms, max_hops):
    """Every request, on the scripted clock: ``(names, slices)`` per hop."""

    clock = _Clock()
    hops = []
    cursor = None
    with _shipped_walls(), _scripted_clock(clock):
        for _hop in range(max_hops):
            engine = _CappedEngine(world, clock=clock, slice_ms=slice_ms)
            with capture_logs() as logs:
                read, _engine = _page(world, page_size=25, cursor=cursor, engine=engine)
            _check_uncapped_slices(
                engine.slices,
                _logged(logs, "users_matching_walk_uncapped_slice"),
                slice_at=engine.slice_at,
                finish_wall_ms=walk.USER_LIST_WALK_FINISH_WALL_MS,
            )
            hops.append((_names(read), engine.slices))
            if not read.has_more:
                return hops
            cursor = _signed_cursor(read)
    raise AssertionError(f"no end after {max_hops} hops: {[h[0] for h in hops]}")


def test_a_slice_that_outlasts_every_cap_is_read_uncapped_at_its_narrowest():
    """Every slice that returns rows takes 40 s: longer than the whole request.

    Every capped attempt is stopped, however narrow, and the analytics wall
    (30 s) cannot hold even one of them. The request narrows while it can,
    then reads its head-of-line slice once more, at the least width, without
    a cap, and decides that slice's first batch with no deadline: the same
    minimal escape as a replay that outlasts every cap.
    """
    world, expected = _spread_world(12, 5)

    hops = _stopped_slice_hops(world, _every_slice(40_000), max_hops=14)

    # Only an uncapped slice can read rows here, and a request reads one:
    # every request decides one user (the last proves the rest empty).
    published = [names for names, _slices in hops if names]
    assert published == [[name] for name in expected]
    assert len(hops) <= len(expected) + 1
    first = hops[0][1]
    assert [w for w, _c, _s in first[:3]] == [
        walk.USER_LIST_WALK_INITIAL_SLICE / 4**n for n in range(3)
    ]
    assert [s for _w, _c, s in first[:2]] == [True, True]
    width, cap, stopped = first[-1]
    assert cap is None and not stopped and width == walk.USER_LIST_WALK_MIN_SLICE


def test_a_slice_that_outlasts_its_cap_only_when_wide_is_narrowed_not_uncapped():
    """A slice costs 40 s per hour of width: stopped when wide, fine when narrow.

    The request narrows the stopped slice a quarter at a time and reads the
    narrower one under a cap; no slice is ever sent without one.
    """
    world, expected = _spread_world(12, 5)

    hops = _stopped_slice_hops(world, _per_hour(40_000), max_hops=14)

    assert [name for names, _slices in hops for name in names] == expected
    assert all(names for names, _slices in hops[:-1])
    slices = [entry for _names_, hop in hops for entry in hop]
    assert any(stopped for _width, _cap, stopped in slices)
    assert all(cap is not None for _width, cap, _stopped in slices)


def test_a_stopped_slice_at_the_window_start_is_read_at_what_is_left():
    """The slice at the window's start is narrower than the least width.

    It cannot be narrowed; when it is stopped the request reads that slice,
    clipped at the window start, without a cap.
    """
    world = World()
    ids = []
    for n, seconds in enumerate((20, 10, 5), start=1):
        moment = WINDOW_START + timedelta(seconds=seconds)
        ids.append(world.user(n, key=moment, raw=(moment,)))
    expected = ["user-1", "user-2", "user-3"]

    hops = _stopped_slice_hops(world, _every_slice(40_000), max_hops=12)

    assert [name for names, _slices in hops for name in names] == expected
    uncapped = [w for _n, hop in hops for w, cap, _s in hop if cap is None]
    assert uncapped and all(w < walk.USER_LIST_WALK_MIN_SLICE for w in uncapped)


class _SlowSliceDriver(_NativeDriver):
    """The server stops a capped slice wider than ``limit``, as code 159."""

    def __init__(self, engine: Engine, limit: timedelta) -> None:
        super().__init__(engine)
        self.limit = limit
        self.slices: list[tuple[timedelta, float]] = []

    def execute(self, query, params=None, *, with_column_types=False, settings=None):
        if kind_of(query) == "slice":
            width = TICK * (params["slice_end_us"] - params["slice_start_us"])
            cap = float((settings or {}).get("max_execution_time") or 0)
            self.slices.append((width, cap))
            if cap > 0 and width > self.limit:
                from clickhouse_driver.errors import ErrorCodes, ServerException

                raise ServerException(
                    "Timeout exceeded: elapsed 15.0 seconds, maximum: 15",
                    code=ErrorCodes.TIMEOUT_EXCEEDED,
                )
        return super().execute(
            query, params, with_column_types=with_column_types, settings=settings
        )


def test_a_slice_the_server_stops_is_narrowed_through_the_real_client():
    """Through ``V2AnalyticsQueryService`` and ``ClickHouseClient`` unmocked.

    Every slice reaches the driver with a positive ``max_execution_time``;
    the server stops the wide ones (code 159 under that cap), and the walk
    narrows them and publishes every user.
    """
    world, expected = _spread_world(6, 7)
    driver = _SlowSliceDriver(Engine(world), timedelta(minutes=20))

    read = _real_service_page(world, driver)

    assert _names(read) == expected
    assert all(cap > 0 for _width, cap in driver.slices)
    widths = [width for width, _cap in driver.slices]
    assert widths[:2] == [timedelta(hours=1), timedelta(minutes=15)]


# --------------------------------------------------------------------------
# Certification that runs out of a read budget.
# --------------------------------------------------------------------------


class _MemoryEngine(_CappedEngine):
    """An enrichment runs out of memory (code 241) when it holds too much.

    It fails when its bucket is wider than ``width``, whatever it carries,
    or when its users times its bucket's hours exceed ``user_hours``.
    Records every enrichment as ``(users, bucket width, failed)``.
    """

    def __init__(
        self,
        world: World,
        *,
        clock: _Clock,
        width: timedelta | None = None,
        user_hours: float | None = None,
        slice_ms: Callable[[timedelta, bool], float] | None = None,
    ) -> None:
        super().__init__(world, clock=clock, slice_ms=slice_ms)
        self.width = width
        self.user_hours = user_hours
        self.enrichments: list[tuple[int, timedelta, bool]] = []

    def execute_ch_query(
        self,
        query,
        params=None,
        timeout_ms=None,
        settings=None,
        *,
        server_execution_cap_ms=None,
    ):
        if kind_of(query) == "enrich":
            bucket = _from_us(params["attr_end_us"]) - _from_us(params["attr_start_us"])
            users = len(params["eu_ids"])
            failed = (self.width is not None and bucket > self.width) or (
                self.user_hours is not None
                and users * (bucket / timedelta(hours=1)) > self.user_hours
            )
            self.enrichments.append((users, bucket, failed))
            if failed:
                from clickhouse_driver.errors import ErrorCodes, ServerException

                self.calls.append(query)
                self.clock.spend(50.0)
                raise ServerException(
                    "Memory limit exceeded", code=ErrorCodes.MEMORY_LIMIT_EXCEEDED
                )
        return super().execute_ch_query(
            query,
            params,
            timeout_ms,
            settings,
            server_execution_cap_ms=server_execution_cap_ms,
        )


def _memory_hops(world, *, max_hops, **engine):
    """Every request on the scripted clock: ``(names, statements, enrichments)``."""

    clock = _Clock()
    hops = []
    cursor = None
    with _shipped_walls(), _scripted_clock(clock):
        for _hop in range(max_hops):
            memory = _MemoryEngine(world, clock=clock, **engine)
            read, _engine = _page(world, page_size=25, cursor=cursor, engine=memory)
            hops.append((_names(read), len(memory.calls), memory.enrichments))
            if not read.has_more:
                return hops, read
            cursor = _signed_cursor(read)
    raise AssertionError(f"no end after {max_hops} hops: {[h[0] for h in hops]}")


WINDOW = WINDOW_END - WINDOW_START


def test_a_batch_whose_enrichment_runs_out_of_memory_certifies_its_head_alone():
    """Memory grows with users times bucket: five users do not fit, one does.

    The batch's enrichment is tried once over the whole window and not split
    in time; when it runs out of a read budget the head-of-line user is
    certified alone, and the rest of the request certifies one user at a
    time. No enrichment is ever narrowed in time, and every request stays
    within its statement budget.
    """
    world, expected = _spread_world(5, 3)

    hops, last = _memory_hops(world, max_hops=4, user_hours=30.0)

    assert [name for names, _s, _e in hops for name in names] == expected
    enrichments = [entry for _n, _s, hop in hops for entry in hop]
    assert all(bucket == WINDOW for _users, bucket, _failed in enrichments)
    assert [users for users, _b, failed in enrichments if failed] == [5]
    assert all(statements <= 24 for _n, statements, _e in hops)
    # Nothing was narrowed in time, so nothing is marked inexact.
    assert last.payload["query_exact"] is True


@pytest.mark.parametrize("slices", ["cheap", "every_capped_slice_stopped"])
def test_an_enrichment_that_fails_above_a_bucket_width_still_ends(slices):
    """Every enrichment wider than ten minutes runs out of memory, one user or five.

    Only a single user's enrichment is narrowed in time, down to buckets
    that fit: its statements are the documented remainder, at most two per
    bucket of the least width over the window. The list still ends, each
    user once and in order, whether the slices are cheap or every capped
    slice is stopped (the head-of-line path, where that enrichment has no
    deadline).
    """
    world, expected = _spread_world(5, 3)
    slice_ms = _every_slice(40_000) if slices != "cheap" else None

    hops, _last = _memory_hops(
        world, max_hops=12, width=timedelta(minutes=10), slice_ms=slice_ms
    )

    assert [name for names, _s, _e in hops for name in names] == expected
    for _, statements, enrichments in hops:
        # A batch of more than one user is never narrowed in time.
        assert all(
            bucket == WINDOW for users, bucket, _f in enrichments if users > 1
        ), enrichments
        bisection_bound = 2 * (WINDOW / timedelta(minutes=1))
        assert statements <= 43 + bisection_bound, statements
        assert len(enrichments) <= 1 + 2 * (WINDOW / timedelta(minutes=5)), len(
            enrichments
        )


# --------------------------------------------------------------------------
# The view's largest attribute key count.
# --------------------------------------------------------------------------


def test_the_views_largest_key_count_still_decides_every_user():
    """100 attribute keys, the most the Users view accepts, one of them filtered.

    Certifying a batch reads every key, four ordinary keys a statement: 26
    statements, more than the 24-statement budget, so every certification
    was refused and the list returned empty, degraded pages forever. A
    request may always spend one batch's decision.
    """
    from tracer.serializers.trace import UsersQuerySerializer
    from tracer.services.clickhouse.client import ClickHouseClient
    from tracer.services.users_list_manager import UsersListManager

    keys = [f"k{n:02d}" for n in range(99)] + ["tag"]
    assert UsersQuerySerializer(data={"attribute_keys": json.dumps(keys)}).is_valid()
    too_many = json.dumps([*keys, "k99"])
    assert not UsersQuerySerializer(data={"attribute_keys": too_many}).is_valid()

    world, expected = _spread_world(30, 3)
    driver = _NativeDriver(Engine(world))

    def page(cursor):
        client = ClickHouseClient(host="localhost", port=39999, pool_size=1)
        base = _manager()
        manager = UsersListManager(
            organization_id=base.organization_id,
            allowed_project_ids=list(base.scoped_project_ids),
            project_id=base.project_id,
            filters=base.filters,
            requested_columns=[],
            attribute_keys=keys,
        )
        assert walk._enrichment_statement_count(manager) == 26
        with (
            patch.object(client, "_get_client", return_value=driver),
            patch.object(client, "_return_client"),
            patch(
                "tracer.services.clickhouse.v2.query_service.get_v2_query_client",
                return_value=client,
            ),
            patch.object(
                manager, "_read_dimension_candidates", side_effect=_never_seed
            ),
        ):
            before = len(driver.sent)
            read = manager.list_cursor_payload(page_size=25, cursor=cursor)
        return read, len(driver.sent) - before

    names = []
    cursor = None
    for _hop in range(6):
        read, statements = page(cursor)
        assert statements <= 9 + 26 + 2 * 1, statements
        names.extend(_names(read))
        if not read.has_more:
            break
        cursor = _signed_cursor(read)
    assert names == expected


# --------------------------------------------------------------------------
# Property: generated worlds, from tiny budgets to the shipped limits.
# --------------------------------------------------------------------------


def _instants() -> list[datetime]:
    """Where ties are likeliest to hurt: the window's edges and slice floors."""

    first_floor = WINDOW_END - walk.USER_LIST_WALK_INITIAL_SLICE
    return [
        WINDOW_START,
        WINDOW_START + TICK,
        WINDOW_END - TICK,
        first_floor,
        first_floor - TICK,
        first_floor + TICK,
        minutes_before_end(3),
        minutes_before_end(61),
        minutes_before_end(90),
        WINDOW_END - timedelta(hours=5),
    ]


def _world(rng: random.Random, n_users: int) -> World:
    world = World()
    pool = rng.sample(range(1, 1 << 16), n_users * 4)
    instants = _instants()
    span = int((WINDOW_END - WINDOW_START) / TICK)
    tie_bias = rng.choice([0.0, 0.3, 0.7, 0.95])
    reject_rate = rng.choice([0.0, 0.12, 0.35])

    def moment() -> datetime:
        if rng.random() < tie_bias:
            return rng.choice(instants)
        return WINDOW_START + TICK * rng.randrange(span)

    for n in range(n_users):
        aliases = pool[n * 4 + 1 : n * 4 + 1 + rng.choice([0, 0, 0, 1, 2, 3])]
        # Witnessed but never matching live: no key, never a member.
        key = None if rng.random() < 0.12 else moment()
        raw = [(key, rng.randrange(4))] if key is not None else []
        # Other witnessed rows, above the key (stale versions) or below it.
        raw += [(moment(), rng.randrange(4)) for _ in range(rng.choice([0, 0, 1, 2]))]
        if not raw:
            raw = [(moment(), 0)]
        _add(
            world,
            pool[n * 4],
            aliases,
            key,
            raw,
            curated=rng.random() >= reject_rate,
        )
    return world


# (slice user limit, statement budget, certify batch, finishing statements):
# tiny budgets up to the shipped limits. A request never has fewer
# statements than one decision (``walk._statement_budget``), so the tiniest
# budgets here are lifted to that. The finishing statements are real ones:
# the page shows the columns and filters that make them (``FINISH_SHAPES``).
LIMITS = [
    (2, 6, 1, 1),
    (2, 8, 2, 2),
    (3, 7, 3, 1),
    (4, 10, 2, 2),
    (5, 12, 5, 3),
    (8, 24, 25, 4),
    (200, 24, 25, 1),
    (200, 24, 25, 5),
]
PAGE_SIZES = [1, 2, 3, 7, 25, 100]
STATIC_WORLDS = 160
CHANGING_WORLDS = 40
# What a slice that returns rows costs, per run of the limit grid: nothing
# much; 6 s, past the page wall; 40 s, past the whole request; or 40 s an
# hour, past its cap only while it is wide.
SLICE_MODELS = [
    None,
    _every_slice(6_000),
    None,
    _every_slice(40_000),
    _per_hour(40_000),
]


def _slice_model(seed: int) -> Callable[[timedelta, bool], float] | None:
    return SLICE_MODELS[(seed // len(LIMITS)) % len(SLICE_MODELS)]


# Attribute columns the page shows besides the filtered key, up to the Users
# view's maximum of 100: one enrichment statement per four of them, so 1 to
# 26 enrichment statements per certified batch. None is listed twice: most
# pages show few columns, and every column costs statements to build.
KEY_COUNTS = [0, 0, 3, 12, 40, 100]


def _key_count(seed: int) -> int:
    # A generator of its own, so the key count varies independently of the
    # limit grid and the slice model, and the world's own draws stay put.
    return random.Random(7_919 * seed + 1).choice(KEY_COUNTS)


def _slice_hops(world: World, seed: int) -> int:
    """Extra hops a world may take when no capped slice can return rows.

    Then only the head-of-line slice read without a cap, one least width
    wide, returns rows, and a request reads one: every least-width window
    that holds a witnessed row may cost a request of its own.
    """
    if _slice_model(seed) is not SLICE_MODELS[3]:
        return 0
    return len({moment.replace(second=0, microsecond=0) for moment, _id in world.raw})


@contextmanager
def _limits(seed: int):
    """The walk's limits for ``seed``; yields its statement budget and the
    finishing statements its page makes."""

    slice_limit, max_statements, batch, finish = LIMITS[seed % len(LIMITS)]
    with (
        patch.object(walk, "USER_LIST_WALK_SLICE_USER_LIMIT", slice_limit),
        patch.object(walk, "USER_LIST_WALK_MAX_STATEMENTS", max_statements),
        patch.object(walk, "USER_LIST_WALK_CERTIFY_BATCH_SIZE", batch),
    ):
        yield max_statements, finish


def _hop_bound(n_users: int, page_size: int, heavy: int) -> int:
    """The hops a world may take.

    A hop publishes a user, decides one (rejected, or placed elsewhere), or
    lowers the coverage by at least a slice; a hop that does none of these
    opens the instant below the coverage, and the next one decides there.
    So two hops per user, the empty window's slices, and, for each heavy
    user, one-user pages for the users sharing its replay batch ahead of it.
    """

    return 30 + 2 * n_users + heavy * (min(page_size, n_users) + 2)


@pytest.mark.parametrize("seed", range(STATIC_WORLDS))
def test_every_hop_sequence_ends_exact_and_never_raises_the_coverage(seed):
    rng = random.Random(seed)
    n_users = rng.choice([1, 3, 8, 20, 45, 80])
    world = _world(rng, n_users)
    page_size = rng.choice(PAGE_SIZES)
    heavy = frozenset(
        rng.sample(sorted(world.users), min(n_users, rng.choice([0, 0, 0, 1, 2, 3])))
    )
    # A quarter of the worlds lose the server on every fifth request.
    outage = random.Random(31 * seed + 7).random() < 0.25
    bound = _hop_bound(n_users, page_size, len(heavy)) + _slice_hops(world, seed)
    with _limits(seed) as (max_statements, finish), _shipped_walls():
        names, _hops = _follow(
            world,
            page_size=page_size,
            max_hops=bound + (bound // 4 if outage else 0),
            max_statements=max_statements,
            heavy=heavy,
            slice_ms=_slice_model(seed),
            keys=_key_count(seed),
            finish=finish,
            outage_every=5 if outage else None,
        )

    assert len(names) == len(set(names)), "a user was published twice"
    assert names == _expected(world)


def _stop_matching(rng: random.Random, changed: set[str]):
    """Between hops, an unpublished member may stop matching or be rejected."""

    def mutate(world: World, published: set[str], _cursor: tuple) -> None:
        members = [
            uid
            for uid, user in world.users.items()
            if uid not in published and user["key"] is not None and user["curated"]
        ]
        if not members or rng.random() < 0.5:
            return
        uid = rng.choice(members)
        changed.add(uid)
        if rng.random() < 0.5:
            world.users[uid]["key"] = None
        else:
            world.users[uid]["curated"] = False

    return mutate


@pytest.mark.parametrize("seed", range(CHANGING_WORLDS))
def test_users_that_stop_matching_between_hops_never_stall_or_repeat(seed):
    rng = random.Random(10_000 + seed)
    n_users = rng.choice([8, 20, 45, 80])
    world = _world(rng, n_users)
    before = _expected(world)
    page_size = rng.choice([1, 3, 7, 25])
    heavy = frozenset(rng.sample(sorted(world.users), rng.choice([0, 0, 1, 2])))
    changed: set[str] = set()
    with _limits(seed) as (max_statements, finish), _shipped_walls():
        names, _hops = _follow(
            world,
            page_size=page_size,
            max_hops=_hop_bound(n_users, page_size, len(heavy))
            + _slice_hops(world, seed),
            max_statements=max_statements,
            heavy=heavy,
            mutate=_stop_matching(rng, changed),
            slice_ms=_slice_model(seed),
            keys=_key_count(seed),
            finish=finish,
        )

    assert len(names) == len(set(names)), "a user was published twice"
    # Users that never changed are all published, in order; a changed user
    # appears only if it was published before it changed.
    assert [uid for uid in names if uid not in changed] == [
        uid for uid in before if uid not in changed
    ]
    assert set(names) <= set(before)


def _passed(key: datetime, uid: str, cursor: tuple) -> bool:
    """Whether ``cursor`` has already walked past the position ``(key, uid)``."""

    _marker, last_key, last_id, coverage = cursor[:4]
    return key >= coverage or (
        last_key is not None and (key, uid) >= (last_key, last_id)
    )


def _change(rng: random.Random, kind: str, changed: dict[str, bool]):
    """Between hops, an unpublished user starts matching or its key moves.

    ``start`` makes a non-member a member (a new live match, or no longer
    rejected); ``up`` and ``down`` move a member's newest live match, its
    old row staying behind as a witness. Each user changes at most once;
    ``changed`` records whether the cursor had already passed its new key.
    """

    def mutate(world: World, published: set[str], cursor: tuple) -> None:
        if rng.random() < 0.5:
            return
        uid = rng.choice(sorted(world.users))
        user = world.users[uid]
        if uid in published or uid in changed:
            return
        member = user["key"] is not None and user["curated"]
        if kind == "start":
            if member:
                return
            user["curated"] = True
            if user["key"] is None:
                user["key"] = WINDOW_START + TICK * rng.randrange(WINDOW // TICK)
        elif not member:
            return
        elif kind == "up":
            user["key"] = user["key"] + (WINDOW_END - TICK - user["key"]) * rng.random()
        else:
            user["key"] = WINDOW_START + (user["key"] - WINDOW_START) * rng.random()
        world.raw.append((user["key"], uid))
        changed[uid] = _passed(user["key"], uid, cursor)

    return mutate


@pytest.mark.parametrize("kind", ["start", "up", "down"])
@pytest.mark.parametrize("seed", range(16))
def test_users_that_change_between_hops_follow_the_coverage_fence(kind, seed):
    """A change a cursor has walked past is not published by it; any other is.

    A user whose new key lies at or above the cursor's coverage, or behind
    its keyset, was decided before the change as the data stood then, so
    the cursor does not publish it (a new first page would). A user whose new
    key lies ahead of the cursor is published once, where it now sorts.
    Unchanged users are all published, once, in order.
    """
    rng = random.Random(20_000 + 97 * seed + len(kind))
    n_users = rng.choice([8, 20, 45])
    world = _world(rng, n_users)
    page_size = rng.choice([1, 3, 7, 25])
    changed: dict[str, bool] = {}
    with _limits(seed) as (max_statements, finish), _shipped_walls():
        names, _hops = _follow(
            world,
            page_size=page_size,
            max_hops=_hop_bound(n_users, page_size, 0),
            max_statements=max_statements,
            mutate=_change(rng, kind, changed),
            slice_ms=_slice_model(seed),
            finish=finish,
        )

    assert len(names) == len(set(names)), "a user was published twice"
    members = set(_expected(world))
    for uid, passed in changed.items():
        assert (uid in names) == (not passed and uid in members), (uid, passed)
    # Everything published is in newest-matching-activity order as it stands.
    positions = {uid: (world.users[uid]["key"], uid) for uid in names}
    assert names == sorted(names, key=positions.__getitem__, reverse=True)
    assert [uid for uid in names if uid not in changed] == [
        uid for uid in _expected(world) if uid not in changed
    ]
