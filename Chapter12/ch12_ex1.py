"""Functional Python Programming 3e

Chapter 12, Example Set 1
"""
import pytest


from collections.abc import Callable
from functools import wraps
from typing import Optional, Any, TypeVar, cast

FuncType = Callable[..., Any]
FT = TypeVar("FT", bound=FuncType)


def nullable(function: FT) -> FT:
    @wraps(function)
    def null_wrapper(arg: Optional[Any]) -> Optional[Any]:
        return None if arg is None else function(arg)

    return cast(FT, null_wrapper)


# from functools import wraps
# from typing import Callable, Optional, Any, TypeVar, cast
#
# FuncType = Callable[..., Any]
# F = TypeVar('F', bound=FuncType)
#
# def nullable(function: F) -> F:
#     @wraps(function)
#     def null_wrapper(arg: Optional[Any]) -> Optional[Any]:
#         return None if arg is None else function(arg)
#     return cast(F, null_wrapper)

import math


@nullable
def st_miles(nm: Optional[float]) -> Optional[float]:
    return 1.15078 * cast(float, nm)


# reveal_type(st_miles)

REPL_st_miles = """
>>> some_data = [8.7, 86.9, None, 43.4, 60]
>>> scaled = map(st_miles, some_data)
>>> list(scaled)
[10.011785999999999, 100.002782, None, 49.94385199999999, 69.04679999999999]
"""


@nullable
def nround4(x: Optional[float]) -> Optional[float]:
    return round(cast(float, x), 4)


# @nullable
# def nround4(x: Optional[float]) -> Optional[float]:
#     return round(x, 4)

REPL_st_miles_nround4 = """
>>> some_data = [8.7, 86.9, None, 43.4, 60]
>>> scaled = map(st_miles, some_data)
>>> [nround4(v) for v in scaled]
[10.0118, 100.0028, None, 49.9439, 69.0468]

"""

st_miles_2: Callable[[float | None], float | None] = nullable(lambda nm: nm * 1.15078)
nround4_2: Callable[[float | None], float | None] = nullable(lambda x: round(x, 4))


def test_Null_st_miles() -> None:
    some_data = [8.7, 86.9, None, 43.4, 60]
    scaled = map(st_miles_2, some_data)
    rounded = [nround4_2(v) for v in scaled]
    assert rounded == [10.0118, 100.0028, None, 49.9439, 69.0468]


def null2(function: FT) -> FT:
    @wraps(function)
    def null_wrapper(*arg: Any, **kw: Any) -> Any | None:
        try:
            return function(*arg, **kw)
        except TypeError as e:
            if "NoneType" in e.args[0]:
                return None
            raise

    return cast(FT, null_wrapper)


def test_null2() -> None:
    """Note that mypy spots several suspicious constructs."""
    ndivmod = null2(divmod)
    assert ndivmod(None, 2) is None  # type: ignore[misc]
    assert ndivmod(2, None) is None  # type: ignore[misc]
    with pytest.raises(TypeError):
        ndivmod("22", "7")  # type: ignore[call-overload]


import logging


def logged(function: FT) -> FT:
    @wraps(function)
    def log_wrapper(*args: Any, **kw: Any) -> Any:
        log = logging.getLogger(function.__qualname__)
        try:
            result = function(*args, **kw)
            log.info("(%r %r) => %r", args, kw, result)
        except Exception:
            log.exception("(%r %r)", args, kw)
            raise

    return cast(FT, log_wrapper)


def test_logged_divmod_1(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    ldivmod = logged(divmod)
    with pytest.raises(TypeError):
        ldivmod(3, None)  # type: ignore[misc]
    assert caplog.text.startswith("ERROR    divmod:ch12_ex1.py:")


def test_logged_divmod_2(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    ldivmod = logged(divmod)
    ldivmod(22, 7)
    assert caplog.text.startswith("INFO     divmod:ch12_ex1.py:")


import decimal
from collections.abc import Callable
from typing import Any, TypeVar, cast

ConversionFunction = Callable[..., Any]
DFT = TypeVar("DFT", bound=ConversionFunction)


def bad_data(function: DFT) -> DFT:
    @wraps(function)
    def wrap_bad_data(text: str, *args: Any, **kw: Any) -> Any:
        try:
            return function(text, *args, **kw)
        except (ValueError, decimal.InvalidOperation):
            cleaned = text.replace(",", "")
            return function(cleaned, *args, **kw)

    return cast(DFT, wrap_bad_data)


#
# import decimal
# def bad_data(function: F) -> F:
#     @wraps(function)
#     def wrap_bad_data(text: str, *args: Any, **kw: Any) -> Any:
#         try:
#             return function(text, *args, **kw)
#         except (ValueError, decimal.InvalidOperation):
#             cleaned = text.replace(",", "")
#             return function(cleaned, *args, **kw)
#     return cast(F, wrap_bad_data)

from decimal import Decimal

bd_int = bad_data(int)
bd_float = bad_data(float)
bd_decimal = bad_data(Decimal)
# from decimal import Decimal
# bd_int = bad_data(int)
# bd_float = bad_data(float)
# bd_decimal = bad_data(Decimal)

REPL_bad_data = """
>>> bd_int("13")
13
>>> bd_int("1,371")
1371
>>> bd_int("1,371", base=16)
4977

>>> from decimal import Decimal
>>> bd_int( "13" )
13
>>> bd_int( "1,371" )
1371
>>> bd_int( "1,371", base=16 )
4977
>>> bd_float("17")
17.0
>>> bd_float("1,701")
1701.0
>>> bd_decimal(19)
Decimal('19')
>>> bd_decimal("1,956")
Decimal('1956')
"""


def clean_list(text: str, char_list: tuple[str, ...]) -> str:
    if char_list:
        return clean_list(text.replace(char_list[0], ""), char_list[1:])
    return text


#
# def clean_list(text: str, char_list: tuple[str, ...]) -> str:
#     if char_list:
#         return clean_list(text.replace(char_list[0], ""), char_list[1:])
#     return text

import decimal


def bad_char_remove(*char_list: str) -> Callable[[FT], FT]:
    def cr_decorator(function: FT) -> FT:
        @wraps(function)
        def wrap_char_remove(text: str, *args: Any, **kw: Any) -> Any:
            try:
                return function(text, *args, **kw)
            except (ValueError, decimal.InvalidOperation):
                cleaned = clean_list(text, char_list)
                return function(cleaned, *args, **kw)

        return cast(FT, wrap_char_remove)

    return cr_decorator


# import decimal
# def bad_char_remove(*char_list: str) -> Callable[[F], F]:
#     def cr_decorator(function: F) -> F:
#         @wraps(function)
#         def wrap_char_remove(text, *args, **kw):
#             try:
#                 return function(text, *args, **kw)
#             except (ValueError, decimal.InvalidOperation):
#                 cleaned = clean_list(text, char_list)
#                 return function(cleaned, *args, **kw)
#         return cast(F, wrap_char_remove)
#     return cr_decorator

from decimal import Decimal


@bad_char_remove("$", ",")
def currency(text: str, **kw: Any) -> Decimal:
    return Decimal(text, **kw)


# from decimal import Decimal
# @bad_char_remove("$", ",")
# def currency(text: str, **kw) -> Decimal:
#     return Decimal(text, **kw)

REPL_bad_char_remove = """
>>> currency("13")
Decimal('13')
>>> currency("$3.14")
Decimal('3.14')
>>> currency("$1,701.00")
Decimal('1701.00')

>>> currency( "13" )
Decimal('13')
>>> currency( "$3.14" )
Decimal('3.14')
>>> currency( "$1,701.00" )
Decimal('1701.00')
"""


# WHY WE DON'T DO THIS!!
# The type signatures are a mess.

CF = TypeVar("CF", bound=FuncType)


def then_convert(convert_function: Callable[[str], Any]) -> Callable[[CF], CF]:
    def abstract_decorator(clean_func: CF) -> CF:
        @wraps(clean_func)
        def cc_wrapper(text: str, *args: Any, **kw: Any) -> Any:
            try:
                return convert_function(text, *args, **kw)
            except (ValueError, decimal.InvalidOperation):
                cleaned = clean_func(text)
                return convert_function(cleaned, *args, **kw)

        return cast(CF, cc_wrapper)

    return abstract_decorator


@then_convert(int)
def drop_punct(text: str) -> str:  # Callable[[str], str] is Not the *real* signature!
    return text.replace(",", "").replace("$", "")


# reveal_type(drop_punct)

REPL_then_convert_1 = """
>>> drop_punct("1,701")
1701
>>> drop_punct("97")
97
>>>
"""

REPL_then_convert_2 = """
>>> def drop_punct(text):
...    return text.replace(",", "").replace("$", "")
>>> drop_punct_int = then_convert(int)(drop_punct)
>>> drop_punct_int("1,701")
1701
>>> drop_punct_int("97")
97
>>>
"""

# Much nicer


def cleanse_before(cleanse_function: Callable[[str], Any]) -> Callable[[FT], FT]:
    def concrete_decorator(converter: FT) -> FT:
        @wraps(converter)
        def cc_wrapper(text: str, *args: Any, **kw: Any) -> Any:
            try:
                return converter(text, *args, **kw)
            except (ValueError, decimal.InvalidOperation):
                cleaned = cleanse_function(text)
                return converter(cleaned, *args, **kw)

        return cast(FT, cc_wrapper)

    return concrete_decorator


#
# def cleanse_before(
#         cleanse_function: Callable
#     ) -> Callable[[F], F]:
#     def abstract_decorator(converter: F) -> F:
#         @wraps(converter)
#         def cc_wrapper(text: str, *args, **kw) -> Any:
#             try:
#                 return converter(text, *args, **kw)
#             except (ValueError, decimal.InvalidOperation):
#                 cleaned = cleanse_function(text)
#                 return converter(cleaned, *args, **kw)
#         return cast(F, cc_wrapper)
#     return abstract_decorator


def drop_punct2(text: str) -> str:
    return text.replace(",", "").replace("$", "")


@cleanse_before(drop_punct2)
def to_int(text: str, base: int = 10) -> int:
    return int(text, base)


# def drop_punct2(text: str) -> str:
#     return text.replace(",", "").replace("$", "")
#
# @cleanse_before(drop_punct)
# def to_int(text: str, base: int = 10) -> int:
#     return int(text, base)

to_int2 = cleanse_before(drop_punct)(int)
# to_int2 = cleanse_before(drop_punct)(int)

# reveal_type(to_int)
# reveal_type(to_int2)
# reveal_type(int)

REPL_cleanse_before = """
>>> to_int("1,701")
1701
>>> to_int("97")
97
>>> to_int2("1,701")
1701
>>> to_int2("97")
97
"""

from collections.abc import Iterable, Iterator

FloatFuncT = Callable[..., Iterator[float]]
FDT = TypeVar("FDT", bound=FloatFuncT)


def normalized(mean: float, stdev: float) -> FDT:
    z_score: Callable[[float], float] = lambda x: (x - mean) / stdev

    def concrete_decorator(function: FDT) -> FDT:
        @wraps(function)
        def wrapped(data_arg: Iterable[float]) -> Iterator[float]:
            z = map(z_score, data_arg)
            return function(z)

        return cast(FDT, wrapped)

    return cast(FDT, concrete_decorator)


REPL_normalized = """
>>> d = [ 2, 4, 4, 4, 5, 5, 7, 9 ]
>>> from Chapter04.ch04_ex4 import mean, stdev
>>> m_d, s_d =  mean(d), stdev(d)
>>> @normalized(m_d, s_d)
... def norm_list(d):
...    return list(d)
>>> norm_list(d)
[-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]

Alternative, just to show it works.
>>> z = lambda x, m, s: (x-m)/s
>>> list(z(x, mean(d), stdev(d)) for x in d)
[-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]

>>> @normalized(m_d, s_d)
... def norm_sum(d):
...      return sum(d)
>>> norm_sum(d)
0.0

"""

__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}
