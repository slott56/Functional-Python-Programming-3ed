"""Functional Python Programming 3e

Chapter 3, Example Set 1
"""
from typing import Callable


def m(n: int) -> int:
    p: int = 2 ** n - 1
    return p


def test_m() -> None:
    assert m(61) == 2 ** 61 - 1
    assert m(89) == 618970019642690137449562111


global_adjustment: float


def some_function(a: float, b: float, t: float) -> float:
    return a + b * t + global_adjustment


def test_some_function() -> None:
    global global_adjustment
    global_adjustment = 13
    assert some_function(2, 3, 5) == 2 + 3 * 5 + global_adjustment


from typing import TextIO

ifile: TextIO
ofile: TextIO


def open_files(iname: str, oname: str) -> None:
    global ifile, ofile
    ifile = open(iname, "r")
    ofile = open(oname, "w")


from collections.abc import Callable


class Mersenne1:
    def __init__(self, algorithm: Callable[[int], int]) -> None:
        self.pow2 = algorithm

    def __call__(self, arg: int) -> int:
        return self.pow2(arg) - 1


def shifty(b: int) -> int:
    return 1 << b


def multy(b: int) -> int:
    if b == 0:
        return 1
    return 2 * multy(b - 1)


def faster(b: int) -> int:
    if b == 0:
        return 1
    if b % 2 == 1:
        return 2 * faster(b - 1)
    t = faster(b // 2)
    return t * t


def test_mults() -> None:
    assert shifty(17) - 1 == 131071
    assert multy(17) - 1 == 131071
    assert faster(17) - 1 == 131071


# Implementations of Mersenne with strategy objects plugged in properly.

m1s = Mersenne1(shifty)

m1m = Mersenne1(multy)

m1f = Mersenne1(faster)


def test_mersenne_1() -> None:
    assert m1s(17) == 131071
    assert m1m(17) == 131071
    assert m1f(17) == 131071


# Alternative Mersenne using class-level configuration.
# The syntax seems more awkward.

from typing import cast


class Mersenne2:
    pow2: Callable[[int], int]

    def __call__(self, arg: int) -> int:
        # Awkward approach to avoiding a self. reference
        # This is *NOT* a method, but methods are the assumption
        pow2 = getattr(self, "pow2")
        return cast(int, pow2(arg) - 1)


class ShiftyMersenne(Mersenne2):
    pow2 = shifty


class MultyMersenee(Mersenne2):
    pow2 = multy


class FasterMersenne(Mersenne2):
    pow2 = faster


m2s = ShiftyMersenne()
m2m = MultyMersenee()
m2f = FasterMersenne()


def test_mersenne() -> None:
    assert m1s(17) == 131071
    assert m1m(17) == 131071
    assert m1f(17) == 131071
    assert m2s(17) == 131071
    assert m2m(17) == 131071
    assert m2f(17) == 131071
    assert m1s(89) == 618970019642690137449562111
    assert m1m(89) == 618970019642690137449562111
    assert m1f(89) == 618970019642690137449562111


__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}


def performance() -> None:
    import timeit

    print(
        m1s.pow2.__name__,
        timeit.timeit("""m1s(17)""", """from Chapter03.ch03_ex1 import m1s"""),
    )
    print(
        m1m.pow2.__name__,
        timeit.timeit("""m1m(17)""", """from Chapter03.ch03_ex1 import m1m"""),
    )
    print(
        m1f.pow2.__name__,
        timeit.timeit("""m1f(17)""", """from Chapter03.ch03_ex1 import m1f"""),
    )


if __name__ == "__main__":
    performance()
