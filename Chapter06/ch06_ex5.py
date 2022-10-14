"""Functional Python Programming 3e

Chapter 6, Example Set 5
"""
from collections.abc import Iterable
from typing import Any, TypeVar, Protocol, TypeAlias


class Comparable(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...

    def __gt__(self, __other: Any) -> bool:
        ...


SupportsRichComparisonT = TypeVar("SupportsRichComparisonT", bound=Comparable)

Leg: TypeAlias = tuple[Any, Any, float]


def group_sort(trip: Iterable[Leg]) -> dict[int, int]:
    """Group legs into bins of 5 nm.

    >>> trip = [('s1', 'e1', 1), ('s4', 'e4', 4.9), ('s5', 'e5', 5), ('s6', 'e6', 6)]
    >>> group_sort(trip)
    {0: 2, 5: 2}
    """

    def group(
        data: Iterable[SupportsRichComparisonT],
    ) -> Iterable[tuple[SupportsRichComparisonT, int]]:
        sorted_data = iter(sorted(data))
        previous, count = next(sorted_data), 1
        for d in sorted_data:
            if d == previous:
                count += 1
            else:
                yield previous, count
                previous, count = d, 1
        yield previous, count

    quantized = (int(5 * (dist // 5)) for start, stop, dist in trip)
    try:
        return dict(group(quantized))
    except StopIteration:
        return dict()


from collections import Counter


def group_Counter(trip: Iterable[Leg]) -> list[tuple[int, int]]:
    """Group legs into bins of 5 nm.

    >>> trip = [('s1', 'e1', 1), ('s4', 'e4', 4.9), ('s5', 'e5', 5), ('s6', 'e6', 6)]
    >>> group_Counter(trip)
    [(0, 2), (5, 2)]
    """
    quantized = (int(5 * (dist // 5)) for start, stop, dist in trip)
    return Counter(quantized).most_common()


REPL_trip1 = """
>>> import urllib.request
>>> from Chapter04.ch04_ex1 import (
...    floats_from_pair, float_lat_lon, row_iter_kml, haversine, legs
... )
>>> from Chapter03.ch03_ex3 import limits

>>> source_url = "file:./Winter%202012-2013.kml"
>>> with urllib.request.urlopen(source_url) as source:
...     trip = tuple((start, end, round(haversine(start, end),4))
...         for start,end in legs(floats_from_pair(float_lat_lon(row_iter_kml(source)))))

>>> start, end, dist = trip[0]
>>> start, end, dist
((37.54901619777347, -76.33029518659048), (37.840832, -76.273834), 17.7246)
>>> start, end, dist = trip[-1]
>>> start, end, dist
((38.330166, -76.458504), (38.976334, -76.473503), 38.8019)

>>> lat_iter = (lat1 for lat1, lon1 in (start for start,stop,dist in trip) )
>>> north, south = limits(lat_iter)
>>> dist_iter= (dist for start,stop,dist in trip)
>>> total= sum(dist_iter)
>>> average = total/len(trip)

>>> print(f"{south=}")
south=23.9555
>>> print(f"{north=}")
north=38.992832
>>> print(f"{total=}")
total=2481.3662
>>> print(f"average={round(average,3)}")
average=33.991

>>> expected = {0.0: 4, 65.0: 1, 35.0: 5, 5.0: 5, 70.0: 2, 40.0: 3, 10.0: 5, 45.0: 3, 15.0: 9, 80.0: 1, 50.0: 3, 115.0: 1, 20.0: 5, 85.0: 1, 55.0: 1, 25.0: 5, 60.0: 3, 125.0: 1, 30.0: 15}

>>> group_sort(trip) == expected
True
>>> print(f"Mode={group_sort(trip)}")
Mode={0: 4, 5: 5, 10: 5, 15: 9, 20: 5, 25: 5, 30: 15, 35: 5, 40: 3, 45: 3, 50: 3, 55: 1, 60: 3, 65: 1, 70: 2, 80: 1, 85: 1, 115: 1, 125: 1}

>>> expected = [(30.0, 15), (15.0, 9), (35.0, 5), (5.0, 5), (10.0, 5), (20.0, 5), (25.0, 5), (0.0, 4), (40.0, 3), (45.0, 3), (50.0, 3), (60.0, 3), (70.0, 2), (65.0, 1), (80.0, 1), (115.0, 1), (85.0, 1), (55.0, 1), (125.0, 1)]
>>> set(group_Counter(trip)) == set(expected)
True
>>> print(f"Mode={group_Counter(trip)}")
Mode=[(30, 15), (15, 9), (5, 5), (35, 5), (20, 5), (10, 5), (25, 5), (0, 4), (50, 3), (60, 3), (45, 3), (40, 3), (70, 2), (80, 1), (85, 1), (65, 1), (115, 1), (125, 1), (55, 1)]

"""

REPL_trip2 = """
If we modify this demo so that path is an iterable, not a materialized tuple,
we'll see that the ``limits()`` function doesn't really do what we hoped.

>>> import urllib.request
>>> from Chapter04.ch04_ex1 import (
...    floats_from_pair, float_lat_lon, row_iter_kml, haversine, legs
... )
>>> from Chapter03.ch03_ex3 import limits

>>> source_url = "file:./Winter%202012-2013.kml"
>>> with urllib.request.urlopen(source_url) as source:
...    path = floats_from_pair(float_lat_lon(row_iter_kml(source)))

This consumes the interable...

>>> north, south = limits(path)

Nothing left to process...

>>> trip = tuple((start, end, round(haversine(start, end),4))
...     for start,end in legs(iter(path)))
Traceback (most recent call last):
...
RuntimeError: generator raised StopIteration

"""

__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}
