"""Functional Python Programming 3e

Chapter 7, Example Set 4
"""

# Even more generic Rank-Order processing.

from typing import NamedTuple, Any


class RankData(NamedTuple):
    rank_seq: tuple[float]
    raw: Any


from typing import NamedTuple, Tuple, Any


class Rank_Data(NamedTuple):
    rank_seq: Tuple[float]
    raw: Any


def test_rank_data_class() -> None:
    """
    Two similar variations:
    - Rank_Data((rank,), data) -- singleton ranking
    - Rank_Data((rank, rank), data) -- multiple ranking
    """

    data = {"key1": 1, "key2": 2}
    r = Rank_Data((2, 7), data)
    assert r.rank_seq[0] == 2
    assert r.raw == {"key1": 1, "key2": 2}


REPL_test_rank_data = """
>>> raw_data = {'key1': 1, 'key2': 2}
>>> r = RankData((2, 7), raw_data)
>>> r.rank_seq[0]
2
>>> r.raw
{'key1': 1, 'key2': 2}
"""

from collections.abc import Iterator, Iterable
from typing import Any, TypeVar

LL_Type = TypeVar("LL_Type")


def legs(lat_lon_iter: Iterator[LL_Type]) -> Iterator[tuple[LL_Type, LL_Type]]:
    begin = next(lat_lon_iter)
    for end in lat_lon_iter:
        yield begin, end
        begin = end


from collections.abc import Iterator, Iterable, Sequence
from typing import Any, TypeVar

LL_Type = TypeVar("LL_Type")


def legs_g(
    lat_lon_src: Iterator[LL_Type] | Sequence[LL_Type],
) -> Iterator[tuple[LL_Type, LL_Type]]:
    if isinstance(lat_lon_src, Sequence):
        return legs_g(iter(lat_lon_src))
    elif isinstance(lat_lon_src, Iterator):
        begin = next(lat_lon_src)
        for end in lat_lon_iter:
            yield begin, end
            begin = end
    else:
        raise TypeError("not an Iterator or Sequence")


from collections.abc import Sequence, Iterator, Iterable
from typing import Any, TypeVar

LL_Type = TypeVar("LL_Type")


def legs_m(
    lat_lon_src: Iterator[LL_Type] | Sequence[LL_Type],
) -> Iterator[tuple[LL_Type, LL_Type]]:
    # lat_lon_iter: Iterator[LL_Type]  # Needed? Or cast(Iterator[LL_Type], lat_lon_src)?

    match lat_lon_src:
        case Sequence():
            lat_lon_iter = iter(lat_lon_src)
        case Iterator() as lat_lon_iter:
            pass
        case _:
            raise TypeError("not an Iterator or Sequence")

    begin = next(lat_lon_iter)
    for end in lat_lon_iter:
        yield begin, end
        begin = end


from collections.abc import Sequence, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedSample:
    rank_seq: tuple[float]
    raw: Any

    @classmethod
    def create(cls, source: RankData | Any) -> "RankedSample":
        match source:
            case cls():
                return source
            case _:
                return cls((), source)


Source = RankedSample | Any

Ranked = Sequence[Source] | Iterator[Source]

from collections.abc import Callable, Sequence, Iterator, Iterable
from typing import TypeVar, cast, Protocol


class Comparable(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...

    def __gt__(self, __other: Any) -> bool:
        ...

    def __le__(self, __other: Any) -> bool:
        ...

    def __ge__(self, __other: Any) -> bool:
        ...


SortKeyT = TypeVar("SortKeyT", bound=Comparable)
Source = RankData | Any


def rank_data(
    seq_or_iter: Sequence[Source] | Iterator[Source],
    key: Callable[[RankedSample], SortKeyT] = lambda obj: cast(SortKeyT, obj),
) -> Iterable[RankedSample]:

    data_iter: Iterator[Source]

    match seq_or_iter:
        case Iterator() as data_iter:
            pass
        case Sequence():
            data_iter = iter(seq_or_iter)

    wrapped = (RankedSample.create(item) for item in data_iter)
    for r, rd in rerank(wrapped, key):
        new_ranks = rd.rank_seq + (r,)
        yield RankedSample(new_ranks, rd.raw)


def test_rank_data() -> None:
    scalars = [0.8, 1.2, 1.2, 2.3, 18]
    ranked = list(rank_data(scalars))
    expected = [
        RankedSample(rank_seq=(1.0,), raw=0.8),
        RankedSample(rank_seq=(2.5,), raw=1.2),
        RankedSample(rank_seq=(2.5,), raw=1.2),
        RankedSample(rank_seq=(4.0,), raw=2.3),
        RankedSample(rank_seq=(5.0,), raw=18),
    ]
    assert ranked == expected

    pairs = ((2, 0.8), (3, 1.2), (5, 1.2), (7, 2.3), (11, 18))
    rank_x = tuple(rank_data(pairs, key=lambda x: x[0]))
    expected = (
        RankedSample(rank_seq=(1.0,), raw=(2, 0.8)),
        RankedSample(rank_seq=(2.0,), raw=(3, 1.2)),
        RankedSample(rank_seq=(3.0,), raw=(5, 1.2)),
        RankedSample(rank_seq=(4.0,), raw=(7, 2.3)),
        RankedSample(rank_seq=(5.0,), raw=(11, 18)),
    )
    assert rank_x == expected

    rank_xy = tuple(rank_data(rank_x, key=lambda x: x[1]))
    expected = (
        RankedSample(rank_seq=(1.0, 1.0), raw=(2, 0.8)),
        RankedSample(rank_seq=(2.0, 2.5), raw=(3, 1.2)),
        RankedSample(rank_seq=(3.0, 2.5), raw=(5, 1.2)),
        RankedSample(rank_seq=(4.0, 4.0), raw=(7, 2.3)),
        RankedSample(rank_seq=(5.0, 5.0), raw=(11, 18)),
    )
    assert rank_xy == expected


def rerank(
    rank_data_iter: Iterable[RankedSample], key: Callable[[RankedSample], SortKeyT]
) -> Iterator[tuple[float, RankedSample]]:
    sorted_iter = iter(sorted(rank_data_iter, key=lambda obj: key(obj.raw)))
    # Apply ranker to head, *tail = sorted(rank_data_iter)
    try:
        head = next(sorted_iter)
    except StopIteration:
        return
    yield from ranker(sorted_iter, 0, [head], key)


def yield_sequence(
    rank: float, same_rank_iter: Iterator[RankedSample]
) -> Iterator[tuple[float, Rank_Data]]:
    try:
        head = next(same_rank_iter)
    except StopIteration:
        return
    yield rank, head
    yield from yield_sequence(rank, same_rank_iter)


def ranker(
    sorted_iter: Iterator[RankedSample],
    base: float,
    same_rank_seq: list[RankedSample],
    key: Callable[[RankedSample], SortKeyT],
) -> Iterator[tuple[float, RankedSample]]:
    try:
        value = next(sorted_iter)
    except StopIteration:
        dups = len(same_rank_seq)
        yield from yield_sequence((base + 1 + base + dups) / 2, iter(same_rank_seq))
        return
    if key(value.raw) == key(same_rank_seq[0].raw):
        yield from ranker(sorted_iter, base, same_rank_seq + [value], key)
    else:
        dups = len(same_rank_seq)
        yield from yield_sequence((base + 1 + base + dups) / 2, iter(same_rank_seq))
        yield from ranker(sorted_iter, base + dups, [value], key)


from typing import List


def ranker(
    sorted_iter: Iterator[Rank_Data],
    base: float,
    same_rank_seq: List[Rank_Data],
    key: Callable[[Rank_Data], SortKeyT],
) -> Iterator[Tuple[float, Rank_Data]]:
    try:
        value = next(sorted_iter)
    except StopIteration:
        # Final batch
        dups = len(same_rank_seq)
        yield from yield_sequence((base + 1 + base + dups) / 2, iter(same_rank_seq))
        return
    if key(value.raw) == key(same_rank_seq[0].raw):
        # Matching, accumulate a batch
        yield from ranker(sorted_iter, base, same_rank_seq + [value], key)
    else:
        # Non-matching, emit the previous batch and start a new batch
        dups = len(same_rank_seq)
        yield from yield_sequence((base + 1 + base + dups) / 2, iter(same_rank_seq))
        yield from ranker(sorted_iter, base + dups, [value], key)


def test_ranker() -> None:
    scalars = [0.8, 1.2, 1.2, 2.3, 18]
    ranked = list(rank_data(scalars))
    expected = [
        RankedSample(rank_seq=(1.0,), raw=0.8),
        RankedSample(rank_seq=(2.5,), raw=1.2),
        RankedSample(rank_seq=(2.5,), raw=1.2),
        RankedSample(rank_seq=(4.0,), raw=2.3),
        RankedSample(rank_seq=(5.0,), raw=18),
    ]
    assert ranked == expected


from typing import NamedTuple


class Pair(NamedTuple):
    x: float
    y: float


from collections.abc import Sequence
from Chapter04.ch04_ex4 import corr


def pearson_corr(pairs: Sequence[Pair]) -> float:
    X = tuple(p.x for p in pairs)
    Y = tuple(p.y for p in pairs)
    return corr(X, Y)


REPL_example = """
>>> data = [Pair(x=10.0, y=8.04),
... Pair(x=8.0, y=6.95),
... Pair(x=13.0, y=7.58), Pair(x=9.0, y=8.81),
... Pair(x=11.0, y=8.33), Pair(x=14.0, y=9.96),
... Pair(x=6.0, y=7.24), Pair(x=4.0, y=4.26),
... Pair(x=12.0, y=10.84), Pair(x=7.0, y=4.82),
... Pair(x=5.0, y=5.68)]
>>> round(pearson_corr(data), 3)
0.816

>>> from pprint import pprint
>>> scalars = [0.8, 1.2, 1.2, 2.3, 18]
>>> ranked = list(rank_data(scalars))
>>> pprint(ranked)
[RankedSample(rank_seq=(1.0,), raw=0.8),
 RankedSample(rank_seq=(2.5,), raw=1.2),
 RankedSample(rank_seq=(2.5,), raw=1.2),
 RankedSample(rank_seq=(4.0,), raw=2.3),
 RankedSample(rank_seq=(5.0,), raw=18)]

>>> from pprint import pprint

>>> scalars = [0.8, 1.2, 1.2, 2.3, 18]
>>> ranked = list(rank_data(scalars))
>>> pprint(ranked)
[RankedSample(rank_seq=(1.0,), raw=0.8),
 RankedSample(rank_seq=(2.5,), raw=1.2),
 RankedSample(rank_seq=(2.5,), raw=1.2),
 RankedSample(rank_seq=(4.0,), raw=2.3),
 RankedSample(rank_seq=(5.0,), raw=18)]

>>> from pprint import pprint
>>> pairs = ((2, 0.8), (3, 1.2), (5, 1.2), (7, 2.3), (11, 18))
>>> rank_x = list(rank_data(pairs, key=lambda x:x[0]))
>>> pprint(rank_x)
[RankedSample(rank_seq=(1.0,), raw=(2, 0.8)),
 RankedSample(rank_seq=(2.0,), raw=(3, 1.2)),
 RankedSample(rank_seq=(3.0,), raw=(5, 1.2)),
 RankedSample(rank_seq=(4.0,), raw=(7, 2.3)),
 RankedSample(rank_seq=(5.0,), raw=(11, 18))]

>>> rank_xy = list(rank_data(rank_x, key=lambda x:x[1]))
>>> pprint(rank_xy)
[RankedSample(rank_seq=(1.0, 1.0), raw=(2, 0.8)),
 RankedSample(rank_seq=(2.0, 2.5), raw=(3, 1.2)),
 RankedSample(rank_seq=(3.0, 2.5), raw=(5, 1.2)),
 RankedSample(rank_seq=(4.0, 4.0), raw=(7, 2.3)),
 RankedSample(rank_seq=(5.0, 5.0), raw=(11, 18))]

"""

__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}
