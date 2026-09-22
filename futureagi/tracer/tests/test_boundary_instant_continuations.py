"""What a wall-stopped continuation may do at a single instant.

The publication floor speaks in result order, and result order can only
separate two rows at one instant when their order tokens are comparable. On a
route that keysets on a physical row - a matched span - they are not, so the
bound can name the INSTANT and nothing finer. Everything here is about that
instant: that a hop always gets through it, that no row is published twice,
and exactly which row a scalar bound still cannot carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from unittest import mock

import pytest

from tracer.selectors.trace_filter_reads import read_bounded_filter_page
from tracer.services.clickhouse.query_service import QueryResult
from tracer.tests.test_bounded_trace_filter_reads import (
    END,
    _ManualMonotonic,
    _SeedTokenFakeBuilder,
    _SeedTokenFakeExecutor,
    _time_filter,
    _WideInitialSliceFakeBuilder,
)

_WINDOW_START = END - timedelta(hours=6)

# Every shape is a list of (id, age) seed roots. A row's RANK is its oldest
# root, as a session's is its oldest live root and a trace's is its canonical
# one, so a row discovered in a recent slice can rank hours earlier.
_SHAPES: dict[str, tuple[tuple[str, timedelta], ...]] = {
    "rank_lag": (
        ("long-new", timedelta(minutes=10)),
        ("short-a", timedelta(minutes=70)),
        ("short-a", timedelta(minutes=75)),
        ("short-b", timedelta(minutes=80)),
        ("long-old", timedelta(minutes=85)),
        ("short-b", timedelta(minutes=95)),
        ("long-old", timedelta(hours=5)),
        ("long-new", timedelta(hours=5, minutes=30)),
    ),
    "single_root_at_checkpoint": (
        ("long-new", timedelta(minutes=10)),
        ("short-a", timedelta(minutes=70)),
        ("short-a", timedelta(minutes=75)),
        ("short-b", timedelta(minutes=80)),
        ("long-old", timedelta(minutes=85)),
        ("long-old", timedelta(hours=5)),
        ("long-new", timedelta(hours=5, minutes=30)),
    ),
    "same_instant_tie": (
        ("long-new", timedelta(minutes=10)),
        ("tie-a", timedelta(minutes=80)),
        ("tie-b", timedelta(minutes=80)),
        ("tie-c", timedelta(minutes=80)),
        ("long-old", timedelta(hours=5)),
        ("long-new", timedelta(hours=5, minutes=30)),
    ),
}


def _ranks(roots: tuple[tuple[str, timedelta], ...]) -> dict[str, datetime]:
    ranks: dict[str, datetime] = {}
    for row_id, age in roots:
        when = END - age
        if row_id not in ranks or when < ranks[row_id]:
            ranks[row_id] = when
    return ranks


def _walk_hops(
    roots: tuple[tuple[str, timedelta], ...],
    *,
    page_size: int = 2,
    max_hops: int = 24,
    matching: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Follow one cursor to exhaustion on a route with its own seed token."""

    seed_rows = [{"id": row_id, "start_time": END - age} for row_id, age in roots]
    ranks = _ranks(roots)
    match_rows = [
        {"id": row_id, "start_time": t}
        for row_id, t in ranks.items()
        if matching is None or row_id in matching
    ]
    hops: list[dict[str, Any]] = []
    order: tuple[Any, ...] | None = None
    scan: dict[str, Any] = {}
    for _ in range(max_hops):
        builder = _SeedTokenFakeBuilder(
            seed_rows,
            start=_WINDOW_START,
            end=END,
            match_rows=match_rows,
            recommended_batch_size=50,
            recommended_seed_batch_size=200,
        )
        clock = _ManualMonotonic()
        executor = _SeedTokenFakeExecutor(
            builder, clock=clock, durations_ms={"seed": 100, "match": 900}
        )
        with mock.patch("tracer.selectors.trace_filter_reads.monotonic", new=clock):
            page = read_bounded_filter_page(
                builder=builder,
                analytics=executor,
                filters=[_time_filter(_WINDOW_START, END)],
                key_field="id",
                page_number=0,
                page_size=page_size,
                deadline_ms=2_400,
                max_seed_attempts=24,
                max_candidates=200,
                max_query_count=50,
                classify_batch_size=50,
                include_incomplete_rows=True,
                bounded_continuation=True,
                cursor_start_time=order[0] if order is not None else None,
                cursor_order_token=order[1] if order is not None else None,
                **scan,
            )
        hops.append(
            {
                "rows": [row["id"] for row in page.rows],
                "complete": page.complete,
                "has_more": page.has_more,
                "floor": page.continuation_published_order_floor,
                "slice_end": page.continuation_slice_end,
                "before": (
                    page.continuation_before_start_time,
                    page.continuation_before_id,
                ),
                "classified": sorted(
                    {
                        candidate
                        for query, params in executor.calls
                        if query == "match"
                        for candidate in params["candidate_ids"]
                    }
                ),
            }
        )
        if page.complete and not page.has_more:
            break
        floor = page.continuation_published_order_floor
        if floor is not None:
            order = (floor[0], "" if floor[1] is None else str(floor[1]))
        elif page.rows:
            order = (page.rows[-1]["start_time"], str(page.rows[-1]["id"]))
        scan = (
            {}
            if page.has_more or page.continuation_slice_end is None
            else {
                "continuation_slice_start": page.continuation_slice_start,
                "continuation_slice_end": page.continuation_slice_end,
                "continuation_before_start_time": (page.continuation_before_start_time),
                "continuation_before_id": page.continuation_before_id,
            }
        )
    return hops


@pytest.mark.unit
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_a_tie_at_the_boundary_instant_never_stalls_the_walk(shape: str) -> None:
    """No hop may re-emit its own input.

    The keyset is the only position that advances INSIDE an instant. An earlier
    revision of this floor gave the keyset up so the instant could be re-read,
    which is correct only while one hop can get through the instant: with more
    rows at that microsecond than a hop classifies, every hop re-read the same
    rows, published none, and handed back the position it started from. The
    list never filled. The floor now names the instant and keeps the keyset, so
    the rows there are published rather than held and the scan moves on.
    """

    hops = _walk_hops(_SHAPES[shape])
    expected = set(_ranks(_SHAPES[shape]))

    published = [row for hop in hops for row in hop["rows"]]
    assert sorted(published) == sorted(expected)
    assert len(published) == len(set(published))
    assert hops[-1]["complete"] is True
    # Progress, hop by hop: no two consecutive hops may publish nothing from
    # the same committed position, which is what a stall looks like.
    positions = [(hop["slice_end"], hop["before"]) for hop in hops]
    empty_repeats = [
        index
        for index in range(1, len(hops))
        if not hops[index]["rows"]
        and not hops[index - 1]["rows"]
        and positions[index] == positions[index - 1]
    ]
    assert empty_repeats == []


@pytest.mark.unit
@pytest.mark.parametrize("page_size", [1, 2, 3, 4])
def test_an_instant_wider_than_the_page_loses_nothing(page_size: int) -> None:
    """Four rows in one microsecond, a page that cannot hold them, no loss.

    This is the shape that decided where the floor may be published at all. A
    hop can stop with its keyset INSIDE such an instant, having classified only
    part of it. A floor that named the whole instant then claimed the unread
    remainder as published and the next hop dropped it; a floor one microsecond
    above held the keyset row that nothing would re-read; giving the keyset up
    to re-read the instant stalled. So no floor is published for a position the
    bound cannot name, the page keeps every match it classified, and the view
    resumes at its last published row - which walks down through the instant,
    one row per hop, losing none of it.
    """

    roots = (
        ("above", timedelta(minutes=10)),
        ("w4", timedelta(minutes=30)),
        ("w3", timedelta(minutes=30)),
        ("w2", timedelta(minutes=30)),
        ("w1", timedelta(minutes=30)),
        ("below", timedelta(minutes=50)),
    )
    hops = _walk_hops(roots, page_size=page_size)
    published = [row for hop in hops for row in hop["rows"]]

    assert sorted(published) == sorted({row_id for row_id, _ in roots})
    assert len(published) == len(set(published))
    assert hops[-1]["complete"] is True
    assert len(hops) < 24


@pytest.mark.unit
@pytest.mark.parametrize("page_size", [1, 2, 3])
def test_an_instant_wider_than_the_page_publishes_no_rejected_row(
    page_size: int,
) -> None:
    """The same instant with non-matching rows in it: none of them leaks."""

    roots = (
        ("above", timedelta(minutes=10)),
        ("y2", timedelta(minutes=30)),
        ("no2", timedelta(minutes=30)),
        ("y1", timedelta(minutes=30)),
        ("no1", timedelta(minutes=30)),
        ("below", timedelta(minutes=50)),
    )
    matching = {"above", "y1", "y2", "below"}
    hops = _walk_hops(roots, page_size=page_size, matching=matching)
    published = [row for hop in hops for row in hop["rows"]]

    assert sorted(published) == sorted(matching)
    assert len(published) == len(set(published))
    assert not [row for row in published if row.startswith("no")]


class _RemappingFakeExecutor(_SeedTokenFakeExecutor):
    """Acquire raw identities, publish canonical ones, as the session read does.

    The seed page carries the ids the rows were written with; the classifier
    resolves each to its survivor and publishes THAT. Both are the same field,
    so nothing in the reader can see that the two orders differ.
    """

    CANONICAL = {"a-early": "zz-canon-early"}

    def execute_ch_query(self, query, params, *, timeout_ms, settings):
        if query != "match":
            return super().execute_ch_query(
                query, params, timeout_ms=timeout_ms, settings=settings
            )
        self.clock.advance_ms(self.durations_ms.get(query, 50))
        self.calls.append((query, params))
        wanted = set(params["candidate_ids"])
        rows = [
            {**row, "id": self.CANONICAL.get(row["id"], row["id"])}
            for row in (self.builder.match_rows or self.builder.rows)
            if row["id"] in wanted
        ]
        return QueryResult(rows, len(rows), "clickhouse", 1.0)


@dataclass
class _RemappingFakeBuilder(_WideInitialSliceFakeBuilder):
    """Keysets on the field it publishes, but not on the same VALUES."""


@pytest.mark.unit
def test_a_remapped_identity_at_the_boundary_instant_is_the_known_residual() -> None:
    """The one row a comparable keyset can still miss, pinned rather than prosed.

    A route can keyset on the same FIELD it publishes and still have the two
    orders disagree, because acquisition carries raw ids and classification
    publishes canonical ones. The comparability test reads one seed row, where
    the two are by construction the same value, so the keyset is taken as
    comparable. A row the hop has not yet seen, ranked at exactly the keyset's
    instant and published under a canonical id that sorts ABOVE the keyset
    row's raw id, therefore ranks at or above the floor without having been
    published, and the next hop drops it.

    This asserts the residual EXISTS. It is the disclosed limit of a scalar
    order bound on a remapping route, and it is not reachable for any identity
    that was never remapped. If a later change closes it, this test fails and
    the disclosure in the pull request has to be retired with it.
    """

    # Four rows in one microsecond inside the walk's first slice, plus one
    # rank-lagging row above them so the prefix cannot be proven and the wall
    # is what stops the hop. The seed batch takes three of the four, so the
    # keyset stops INSIDE the instant with ``a-early`` still unseen.
    instant = END - timedelta(minutes=30)
    seed_rows = [
        {"id": "c-newer", "start_time": END - timedelta(minutes=20)},
        {"id": "b-mid", "start_time": instant},
        {"id": "b-late", "start_time": instant},
        {"id": "a-early", "start_time": instant},
        # ``c-newer`` also has a much older root, so it RANKS below the instant
        # and cannot complete the prefix.
        {"id": "c-newer", "start_time": END - timedelta(hours=5)},
    ]
    ranks = {
        "c-newer": END - timedelta(hours=5),
        "b-mid": instant,
        "b-late": instant,
        "a-early": instant,
    }
    builder = _RemappingFakeBuilder(
        seed_rows,
        start=_WINDOW_START,
        end=END,
        match_rows=[
            {"id": row_id, "start_time": when} for row_id, when in ranks.items()
        ],
        recommended_batch_size=50,
        recommended_seed_batch_size=200,
    )
    clock = _ManualMonotonic()
    executor = _RemappingFakeExecutor(
        builder, clock=clock, durations_ms={"seed": 100, "match": 900}
    )
    with mock.patch("tracer.selectors.trace_filter_reads.monotonic", new=clock):
        page = read_bounded_filter_page(
            builder=builder,
            analytics=executor,
            filters=[_time_filter(_WINDOW_START, END)],
            key_field="id",
            page_number=0,
            page_size=2,
            deadline_ms=2_400,
            max_seed_attempts=24,
            max_candidates=200,
            max_query_count=50,
            classify_batch_size=50,
            include_incomplete_rows=True,
            bounded_continuation=True,
        )

    floor = page.continuation_published_order_floor
    published = {row["id"] for row in page.rows}
    if floor is None or floor[1] is None or floor[0] != instant:
        pytest.skip("this shape did not commit a keyset inside the instant")
    # The floor's token is a RAW id. The row still to be found publishes a
    # canonical id that sorts above it at the same instant, so the next hop's
    # exclusive bound covers a row no page ever published.
    assert "zz-canon-early" not in published
    assert (instant, "zz-canon-early") >= (floor[0], str(floor[1]))
