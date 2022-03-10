"""Functional Python Programming 3e

Chapter 3, Example Set 4
"""

import csv
from typing import TextIO
from collections.abc import Iterator, Iterable


def row_iter(source: TextIO) -> Iterator[list[str]]:
    return csv.reader(source, delimiter="\t")


#
# import csv
# from typing import TextIO
# from collections.abc import Iterator, Iterable
#
# def row_iter(source: TextIO) -> Iterator[list[str]]:
#     rdr = csv.reader(source, delimiter="\t")
#     return rdr


def test_row_iter() -> None:
    import io

    data = io.StringIO("1\t2\t3\n4\t5\t6\n")
    assert list(row_iter(data)) == [["1", "2", "3"], ["4", "5", "6"]]


REPL_row_iter_1 = """
%{ex_5_c}
%{ex_5_d}
...
"""


def float_none(data: str | None) -> float | None:
    """Float conversion: return None instead of ValueError exception.

    >>> float_none('abc')
    >>> float_none('1.23')
    1.23
    """
    try:
        data_f = float(cast(str, data))
        return data_f
    except ValueError:
        return None


from collections.abc import Callable, Iterator


def head_map_filter(
    row_iter: Iterator[list[str | None]],
) -> Iterator[list[float | None]]:
    """Removing headers by applying a filter to get rows with 8 values.

    >>> rows = [ ["Anscombe's quartet"], ['I', 'II', 'III', 'IV'], ['x','y','x','y','x','y','x','y'], ['1','2','3','4','5','6','7','8']]
    >>> list(head_map_filter( rows ))
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    """
    R_Float = list[float | None]

    def float_row(row: list[str | None]) -> R_Float:
        return list(map(float_none, row))

    def all_numeric(row: R_Float) -> bool:
        return all(row) and len(row) == 8

    return filter(all_numeric, map(float_row, row_iter))


from collections.abc import Iterator


def head_split_fixed(row_iter: Iterator[list[str]]) -> Iterator[list[str]]:
    title = next(row_iter)
    assert len(title) == 1 and title[0] == "Anscombe's quartet"
    heading = next(row_iter)
    assert len(heading) == 4 and heading == ["I", "II", "III", "IV"]

    columns = next(row_iter)
    assert len(columns) == 8 and columns == ["x", "y", "x", "y", "x", "y", "x", "y"]
    return row_iter


#
# from collections.abc import Iterator
# def head_split_fixed(row_iter: Iterator[list[str]]) -> Iterator[list[str]]:
#     title = next(row_iter)
#     assert (len(title) == 1
#             and title[0] == "Anscombe's quartet")
#     heading = next(row_iter)
#     assert (len(heading) == 4
#             and heading == ['I', 'II', 'III', 'IV'])
#     columns = next(row_iter)
#     assert (len(columns) == 8
#             and columns == ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y'])
#     return row_iter


def test_head_split_fixed() -> None:
    rows = [
        ["Anscombe's quartet"],
        ["I", "II", "III", "IV"],
        ["x", "y", "x", "y", "x", "y", "x", "y"],
        ["1", "2", "3", "4", "5", "6", "7", "8"],
    ]
    data = list(head_split_fixed(iter(rows)))
    assert data == [["1", "2", "3", "4", "5", "6", "7", "8"]]


def head_split_recurse(row_iter: Iterator[list[str]]) -> Iterator[list[str]]:
    data = next(row_iter)
    if len(data) == 8 and data == ["x", "y", "x", "y", "x", "y", "x", "y"]:
        return row_iter
    return head_split_recurse(row_iter)


def test_head_split_rescurse() -> None:
    rows = [
        ["Anscombe's quartet"],
        ["I", "II", "III", "IV"],
        ["x", "y", "x", "y", "x", "y", "x", "y"],
        ["1", "2", "3", "4", "5", "6", "7", "8"],
    ]
    data = list(head_split_recurse(iter(rows)))
    assert data == [["1", "2", "3", "4", "5", "6", "7", "8"]]


from pathlib import Path
from collections.abc import Iterator


def get_rows(path: Path) -> Iterator[list[str]]:
    with path.open() as source:
        yield from head_split_fixed(row_iter(source))


from pathlib import Path


def test_get_rows() -> None:
    expected = [
        ["10.0", "8.04", "10.0", "9.14", "10.0", "7.46", "8.0", "6.58"],
        ["8.0", "6.95", "8.0", "8.14", "8.0", "6.77", "8.0", "5.76"],
        ["13.0", "7.58", "13.0", "8.74", "13.0", "12.74", "8.0", "7.71"],
        ["9.0", "8.81", "9.0", "8.77", "9.0", "7.11", "8.0", "8.84"],
        ["11.0", "8.33", "11.0", "9.26", "11.0", "7.81", "8.0", "8.47"],
        ["14.0", "9.96", "14.0", "8.10", "14.0", "8.84", "8.0", "7.04"],
        ["6.0", "7.24", "6.0", "6.13", "6.0", "6.08", "8.0", "5.25"],
        ["4.0", "4.26", "4.0", "3.10", "4.0", "5.39", "19.0", "12.50"],
        ["12.0", "10.84", "12.0", "9.13", "12.0", "8.15", "8.0", "5.56"],
        ["7.0", "4.82", "7.0", "7.26", "7.0", "6.42", "8.0", "7.91"],
        ["5.0", "5.68", "5.0", "4.74", "5.0", "5.73", "8.0", "6.89"],
    ]
    actual = list(get_rows(Path("Anscombe.txt")))
    assert actual == expected


REPL_range_side_bar = """
>>> range(10)
range(0, 10)

>>> [range(10)]
[range(0, 10)]

>>> [x for x in range(10)]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> list(range(10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

REPL_get_rows = """
>>> data = list(get_rows(Path("Anscombe.txt")))
>>> data[0]
['10.0', '8.04', '10.0', '9.14', '10.0', '7.46', '8.0', '6.58']
>>> data[1]
['8.0', '6.95', '8.0', '8.14', '8.0', '6.77', '8.0', '5.76']
>>> data[-1]
['5.0', '5.68', '5.0', '4.74', '5.0', '5.73', '8.0', '6.89']
"""

from typing import cast, TypeVar

from typing import cast, TypeVar
from collections.abc import Iterator, Iterable

SrcT = TypeVar("SrcT")


def series(n: int, row_iter: Iterable[list[SrcT]]) -> Iterator[tuple[SrcT, SrcT]]:
    for row in row_iter:
        yield cast(tuple[SrcT, SrcT], tuple(row[n * 2 : n * 2 + 2]))


def test_series() -> None:
    rows = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
    assert list(series(0, rows)) == [(1, 2), (9, 10)]
    assert list(series(1, rows)) == [(3, 4), (11, 12)]


from collections.abc import Callable, Iterable, Iterator

Pair = tuple[str, str]
row_float: Callable[[Iterable[str]], Iterator[float]] = lambda row: map(float, row)

REPL_parse_1 = """
>>> with open("Anscombe.txt") as source:
...     print(list(head_map_filter(row_iter(source))))
[[10.0, 8.04, 10.0, 9.14, 10.0, 7.46, 8.0, 6.58], [8.0, 6.95, 8.0, 8.14, 8.0, 6.77, 8.0, 5.76], [13.0, 7.58, 13.0, 8.74, 13.0, 12.74, 8.0, 7.71], [9.0, 8.81, 9.0, 8.77, 9.0, 7.11, 8.0, 8.84], [11.0, 8.33, 11.0, 9.26, 11.0, 7.81, 8.0, 8.47], [14.0, 9.96, 14.0, 8.1, 14.0, 8.84, 8.0, 7.04], [6.0, 7.24, 6.0, 6.13, 6.0, 6.08, 8.0, 5.25], [4.0, 4.26, 4.0, 3.1, 4.0, 5.39, 19.0, 12.5], [12.0, 10.84, 12.0, 9.13, 12.0, 8.15, 8.0, 5.56], [7.0, 4.82, 7.0, 7.26, 7.0, 6.42, 8.0, 7.91], [5.0, 5.68, 5.0, 4.74, 5.0, 5.73, 8.0, 6.89]]

>>> with open("Anscombe.txt") as source:
...     print(list(head_split_fixed(row_iter(source))))
[['10.0', '8.04', '10.0', '9.14', '10.0', '7.46', '8.0', '6.58'], ['8.0', '6.95', '8.0', '8.14', '8.0', '6.77', '8.0', '5.76'], ['13.0', '7.58', '13.0', '8.74', '13.0', '12.74', '8.0', '7.71'], ['9.0', '8.81', '9.0', '8.77', '9.0', '7.11', '8.0', '8.84'], ['11.0', '8.33', '11.0', '9.26', '11.0', '7.81', '8.0', '8.47'], ['14.0', '9.96', '14.0', '8.10', '14.0', '8.84', '8.0', '7.04'], ['6.0', '7.24', '6.0', '6.13', '6.0', '6.08', '8.0', '5.25'], ['4.0', '4.26', '4.0', '3.10', '4.0', '5.39', '19.0', '12.50'], ['12.0', '10.84', '12.0', '9.13', '12.0', '8.15', '8.0', '5.56'], ['7.0', '4.82', '7.0', '7.26', '7.0', '6.42', '8.0', '7.91'], ['5.0', '5.68', '5.0', '4.74', '5.0', '5.73', '8.0', '6.89']]

>>> with open("Anscombe.txt") as source:
...     print(list(head_split_recurse(row_iter(source))))
[['10.0', '8.04', '10.0', '9.14', '10.0', '7.46', '8.0', '6.58'], ['8.0', '6.95', '8.0', '8.14', '8.0', '6.77', '8.0', '5.76'], ['13.0', '7.58', '13.0', '8.74', '13.0', '12.74', '8.0', '7.71'], ['9.0', '8.81', '9.0', '8.77', '9.0', '7.11', '8.0', '8.84'], ['11.0', '8.33', '11.0', '9.26', '11.0', '7.81', '8.0', '8.47'], ['14.0', '9.96', '14.0', '8.10', '14.0', '8.84', '8.0', '7.04'], ['6.0', '7.24', '6.0', '6.13', '6.0', '6.08', '8.0', '5.25'], ['4.0', '4.26', '4.0', '3.10', '4.0', '5.39', '19.0', '12.50'], ['12.0', '10.84', '12.0', '9.13', '12.0', '8.15', '8.0', '5.56'], ['7.0', '4.82', '7.0', '7.26', '7.0', '6.42', '8.0', '7.91'], ['5.0', '5.68', '5.0', '4.74', '5.0', '5.73', '8.0', '6.89']]

"""

REPL_parse_2 = """
>>> with open("Anscombe.txt") as source:
...     print( list(series(0, head_split_recurse(row_iter(source)))) )
[('10.0', '8.04'), ('8.0', '6.95'), ('13.0', '7.58'), ('9.0', '8.81'), ('11.0', '8.33'), ('14.0', '9.96'), ('6.0', '7.24'), ('4.0', '4.26'), ('12.0', '10.84'), ('7.0', '4.82'), ('5.0', '5.68')]

>>> with open("Anscombe.txt") as source:
...     print( list(series(0, head_map_filter(row_iter(source)))) )
[(10.0, 8.04), (8.0, 6.95), (13.0, 7.58), (9.0, 8.81), (11.0, 8.33), (14.0, 9.96), (6.0, 7.24), (4.0, 4.26), (12.0, 10.84), (7.0, 4.82), (5.0, 5.68)]

>>> with open("Anscombe.txt") as source:
...     data = head_split_fixed(row_iter(source))
...     print( list(series(0,data)) )
[('10.0', '8.04'), ('8.0', '6.95'), ('13.0', '7.58'), ('9.0', '8.81'), ('11.0', '8.33'), ('14.0', '9.96'), ('6.0', '7.24'), ('4.0', '4.26'), ('12.0', '10.84'), ('7.0', '4.82'), ('5.0', '5.68')]

>>> with open("Anscombe.txt") as source:
...     data = head_split_fixed(row_iter(source))
...     series_I= tuple(series(0,data))
...     print(series_I)
(('10.0', '8.04'), ('8.0', '6.95'), ('13.0', '7.58'), ('9.0', '8.81'), ('11.0', '8.33'), ('14.0', '9.96'), ('6.0', '7.24'), ('4.0', '4.26'), ('12.0', '10.84'), ('7.0', '4.82'), ('5.0', '5.68'))

"""

REPL_mean = """
>>> from pathlib import Path
>>> source_path = Path("Anscombe.txt")
>>> with source_path.open() as source:
...     data = tuple(head_split_fixed(row_iter(source)))
>>> series_I = tuple(series(0,data))
>>> series_II = tuple(series(1,data))
>>> series_III = tuple(series(2,data))
>>> series_IV = tuple(series(3,data))

>>> series_I
(('10.0', '8.04'), ('8.0', '6.95'), ... ('5.0', '5.68'))

>>> for subset in series_I, series_II, series_III, series_IV:
...     mean = sum(float(pair[1]) for pair in subset)/len(subset)
...     print( round(mean,3) )
7.501
7.501
7.5
7.501
"""

__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}
