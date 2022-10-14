"""Functional Python Programming 3e

Chapter 16, Example Set 1
"""

REPL_prod = """
>>> prod([1, 2, 3])
6
>>> prod([])
1
"""

from math import prod


class Binomial:
    def __init__(self) -> None:
        self.fact_cache: dict[int, int] = {}
        self.bin_cache: dict[tuple[int, int], int] = {}

    def fact(self, n: int) -> int:
        if n not in self.fact_cache:
            self.fact_cache[n] = prod(range(1, n + 1))
        return self.fact_cache[n]

    def __call__(self, n: int, m: int) -> int:
        if (n, m) not in self.bin_cache:
            self.bin_cache[n, m] = self.fact(n) // (self.fact(m) * self.fact(n - m))
        return self.bin_cache[n, m]


REPL_binom = """
>>> binom = Binomial()
>>> binom(52, 5)
2598960

>>> binom = Binomial()
>>> binom(52, 5)
2598960
"""


def fact(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n * fact(n - 1)


def facti(n: int) -> int:
    if n == 0:
        return 1
    f = 1
    for i in range(2, n):
        f = f * i
    return f


def factr(n: int) -> int:
    """Recursive Factorial
    >>> factr(0)
    1
    >>> factr(1)
    1
    >>> factr(7)
    5040
    """
    if n == 0:
        return 1
    else:
        return n * factr(n - 1)


binom_example = """
>>> binom = Binomial()
>>> binom(52, 5)
2598960
"""

__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}
