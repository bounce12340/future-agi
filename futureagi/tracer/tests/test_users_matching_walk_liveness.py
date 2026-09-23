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
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tracer.services import users_matching_walk as walk
from tracer.services.clickhouse import read_budget
from tracer.services.clickhouse.read_budget import ReadDeadline, ReadDeadlineExceeded
from tracer.tests.test_users_matching_walk import (
    WINDOW_END,
    WINDOW_START,
    Engine,
    World,
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
    replays: list[tuple[tuple[str, ...], bool, bool]], *, proven: bool = True
) -> None:
    """At most one replay without a cap, for one user a capped replay stopped.

    ``replays`` is one request's ``(ids, capped, stopped)``, in order. The
    replay just before it was stopped under the cap and led with that user;
    with ``proven`` (a budget that affords the capped attempt first), it was
    that user's own.
    """
    uncapped = [i for i, (_ids, capped, _stopped) in enumerate(replays) if not capped]
    assert len(uncapped) <= 1, replays
    for index in uncapped:
        (user,) = replays[index][0]
        assert index > 0, replays
        before, capped, stopped = replays[index - 1]
        assert capped and stopped and before[0] == user, replays
        if proven:
            assert before == (user,), replays


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


class _CappedEngine(Engine):
    """A scripted server on which a heavy user's replay outlasts any cap.

    A replay that carries a heavy user, or more than ``batch_limit`` users,
    and asks the server to stop at a cap is stopped there
    (``ReadDeadlineExceeded``, as the service maps code 159 under a cap); the
    same replay without a cap runs to completion. Records every replay as
    ``(ids, capped, stopped)``. With a ``clock``, every statement spends its
    time on it: 1 ms, a stopped replay its cap, and a slice that returns rows
    ``slow_slice_ms`` when that is set (search statements carry no server
    cap, so a slow slice runs to the end whatever the page wall says).
    """

    def __init__(
        self,
        world: World,
        heavy: frozenset[str] = frozenset(),
        batch_limit: int | None = None,
        clock: _Clock | None = None,
        slow_slice_ms: float = 0.0,
    ) -> None:
        super().__init__(world)
        self.heavy = heavy
        self.batch_limit = batch_limit
        self.clock = clock
        self.slow_slice_ms = slow_slice_ms
        self.replays: list[tuple[tuple[str, ...], bool, bool]] = []

    def execute_ch_query(
        self,
        query,
        params=None,
        timeout_ms=None,
        settings=None,
        *,
        server_execution_cap_ms=None,
    ):
        if kind_of(query) == "replay":
            ids = tuple((params or {})["candidate_end_user_ids"])
            capped = server_execution_cap_ms is not None
            stopped = capped and (
                bool(self.heavy.intersection(ids))
                or (self.batch_limit is not None and len(ids) > self.batch_limit)
            )
            self.replays.append((ids, capped, stopped))
            if stopped:
                self.calls.append(query)
                self.settings.append(settings)
                self.timeouts.append(timeout_ms)
                self.caps.append(server_execution_cap_ms)
                if self.clock is not None:
                    self.clock.spend(server_execution_cap_ms)
                raise ReadDeadlineExceeded("ClickHouse statement exceeded its cap")
        result = super().execute_ch_query(
            query,
            params,
            timeout_ms,
            settings,
            server_execution_cap_ms=server_execution_cap_ms,
        )
        if self.clock is not None:
            slow = self.slow_slice_ms and kind_of(query) == "slice" and result.data
            result.query_time_ms = self.slow_slice_ms if slow else 1.0
            self.clock.spend(result.query_time_ms)
        return result


def _follow(
    world: World,
    *,
    page_size: int,
    max_hops: int,
    max_statements: int | None = None,
    heavy: frozenset[str] = frozenset(),
    mutate=None,
    slow_slice_ms: float = 0.0,
) -> tuple[list[str], int]:
    """Follow the cursor to the end; returns the published names and the hops.

    Every hop must keep the coverage where it was or lower it, stay inside
    the statement budget, and send at most one replay without a server cap,
    for one user, right after a capped replay it led was stopped. In a
    static world a repeated ``(cursor, seen rows)`` is a livelock, and no two
    hops in a row may both make no progress: publish a user, lower the
    coverage, or lower the decided position. The walls run on a scripted
    clock (``_CappedEngine``).
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
            slow_slice_ms=slow_slice_ms,
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
    slow_slice_ms: float,
) -> tuple[list[str], int]:
    budget = _decision_budget(max_statements or walk.USER_LIST_WALK_MAX_STATEMENTS)
    names: list[str] = []
    seen_states: set = set()
    coverage = WINDOW_END
    position: tuple = (None, None)
    stalled = False
    cursor = None
    for hop in range(1, max_hops + 1):
        engine = _CappedEngine(world, heavy, clock=clock, slow_slice_ms=slow_slice_ms)
        read, _engine = _page(world, page_size=page_size, cursor=cursor, engine=engine)
        names.extend(_names(read))
        assert len(engine.calls) <= budget, (hop, len(engine.calls))
        _check_uncapped(engine.replays, proven=False)
        if not read.has_more:
            return names, hop
        order = tuple(read.checkpoint_order)
        assert order[3] <= coverage, f"coverage moved up at hop {hop}: {order}"
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
            mutate(world, set(names))
    raise AssertionError(f"no end after {max_hops} hops; published {len(names)}")


def _decision_budget(configured: int) -> int:
    """The statements a request may spend: its budget, or one decision.

    One decision: the open instant, a slice, its survivors, one batch's
    enrichment, and a finish with its uncapped retry.
    """
    manager = _manager()
    return max(
        configured,
        3
        + walk._enrichment_statement_count(manager)
        + 2 * walk._materialisation_statement_count(manager),
    )


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
        if kind_of(query) == "replay":
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
        return super().execute(
            query, params, with_column_types=with_column_types, settings=settings
        )


def _real_service_hops(world: World, driver: _SlowUserDriver, max_hops: int):
    """Every page through the real service and client: ``(names, replays)``."""

    hops = []
    cursor = None
    coverage = WINDOW_END
    for _ in range(max_hops):
        before = len(driver.replays)
        read = _real_service_page(world, driver, cursor=cursor)
        replays = driver.replays[before:]
        _check_uncapped(replays)
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
    driver = _SlowUserDriver(Engine(world), {uid})

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
    driver = _SlowUserDriver(Engine(world), {ids[heavy_rank - 1]})

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
    driver = _SlowUserDriver(Engine(world), set(), batch_limit=3)

    hops = _real_service_hops(world, driver, max_hops=4)

    assert [name for names, _replays in hops for name in names] == [
        f"user-{n}" for n in range(1, 41)
    ]
    assert len(hops[0][0]) > 1
    assert all(capped for _ids, capped, _stopped in driver.replays)


def test_a_heavy_user_inside_a_large_tie_does_not_stall_the_instant():
    world, expected = _tied_world(601)
    heavy = next(u for u, v in world.users.items() if v["name"] == expected[59])
    driver = _SlowUserDriver(Engine(world), {heavy})

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
    driver = _SlowUserDriver(Engine(world), {heavy})

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
    driver = _SlowUserDriver(Engine(world), set())
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
    with patch.object(walk, "_materialisation_statement_count", return_value=2):
        names, _hops = _follow(world, page_size=25, max_hops=400)

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

    The slice statement is admitted inside the wall and runs past it (search
    statements carry no server cap). Its survivor statement was then refused,
    the slice discarded whole, and the next request read the same slice
    again, so the list never got past it. A request that has certified
    nothing yet now finishes that slice's first batch against the analytics
    wall: every request publishes, and the list ends.
    """
    world, expected = _spread_world(60, 2)

    with _shipped_walls():
        names, hops = _follow(world, page_size=25, max_hops=8, slow_slice_ms=6_000)

    assert names == expected
    assert hops <= 6


def test_a_slice_that_outlasts_the_analytics_wall_still_stalls_the_list():
    """Known residual, left to the owner: a slice slower than the whole request.

    Every slice that returns rows takes 31 s, past the analytics wall (30 s)
    that bounds a request's admissions. Nothing after it may start, so the
    request publishes nothing; the next request's open instant moves the
    coverage down one microsecond before the same slice is read again. The
    list shows empty, degraded pages, each taking over 30 s, and never gets
    past that slice. Ending it needs either a statement admitted with no
    deadline at all or a narrower slice carried in the cursor.
    """
    world, _expected_names = _spread_world(3, 5)
    clock = _Clock()
    cursor = None
    coverages = []
    with _shipped_walls(), _scripted_clock(clock):
        for _hop in range(4):
            engine = _CappedEngine(world, clock=clock, slow_slice_ms=31_000)
            read, _engine = _page(world, page_size=25, cursor=cursor, engine=engine)
            assert read.payload["table"] == []
            assert read.has_more is True
            assert read.payload["query_status"] == "degraded"
            coverages.append(read.checkpoint_order[3])
            cursor = _signed_cursor(read)

    assert WINDOW_END - coverages[-1] <= 2 * TICK


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
        assert statements <= 3 + 26 + 2 * 1, statements
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


# (slice user limit, statement budget, certify batch, finishing statements,
# enrichment statements): tiny budgets, up to the shipped limits. A request
# never has fewer statements than one decision (``walk._statement_budget``),
# so the tiniest budgets here are lifted to that.
LIMITS = [
    (2, 6, 1, 1, 1),
    (2, 8, 2, 2, 1),
    (3, 7, 3, 1, 2),
    (4, 10, 2, 2, 2),
    (5, 12, 5, 3, 1),
    (8, 24, 25, 4, 2),
    (200, 24, 25, 1, 1),
    (200, 24, 25, 4, 1),
]
PAGE_SIZES = [1, 2, 3, 7, 25, 100]
STATIC_WORLDS = 320
CHANGING_WORLDS = 96
# Every fourth run of the limit grid: every slice that returns rows takes
# longer than the page wall (and less than the analytics wall).
SLOW_SLICE_MS = 6_000.0


def _slow_slice_ms(seed: int) -> float:
    return SLOW_SLICE_MS if (seed // len(LIMITS)) % 4 == 1 else 0.0


@contextmanager
def _limits(seed: int):
    """The walk's limits for ``seed``; yields its statement budget."""

    slice_limit, max_statements, batch, finish, enrich = LIMITS[seed % len(LIMITS)]
    with (
        patch.object(walk, "USER_LIST_WALK_SLICE_USER_LIMIT", slice_limit),
        patch.object(walk, "USER_LIST_WALK_MAX_STATEMENTS", max_statements),
        patch.object(walk, "USER_LIST_WALK_CERTIFY_BATCH_SIZE", batch),
        patch.object(walk, "_materialisation_statement_count", return_value=finish),
        patch.object(walk, "_enrichment_statement_count", return_value=enrich),
    ):
        yield max_statements


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
    with _limits(seed) as max_statements, _shipped_walls():
        names, _hops = _follow(
            world,
            page_size=page_size,
            max_hops=_hop_bound(n_users, page_size, len(heavy)),
            max_statements=max_statements,
            heavy=heavy,
            slow_slice_ms=_slow_slice_ms(seed),
        )

    assert len(names) == len(set(names)), "a user was published twice"
    assert names == _expected(world)


def _stop_matching(rng: random.Random, changed: set[str]):
    """Between hops, an unpublished member may stop matching or be rejected."""

    def mutate(world: World, published: set[str]) -> None:
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
    with _limits(seed) as max_statements, _shipped_walls():
        names, _hops = _follow(
            world,
            page_size=page_size,
            max_hops=_hop_bound(n_users, page_size, len(heavy)),
            max_statements=max_statements,
            heavy=heavy,
            mutate=_stop_matching(rng, changed),
            slow_slice_ms=_slow_slice_ms(seed),
        )

    assert len(names) == len(set(names)), "a user was published twice"
    # Users that never changed are all published, in order; a changed user
    # appears only if it was published before it changed.
    assert [uid for uid in names if uid not in changed] == [
        uid for uid in before if uid not in changed
    ]
    assert set(names) <= set(before)
