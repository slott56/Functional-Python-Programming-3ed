"""Functional Python Programming 3e

Chapter 11, Example Set 2
"""

from typing import Match, Pattern


def matcher(text: str, *patterns: Pattern[str]) -> Match[str] | None:
    matching = (p.search(text) for p in patterns)
    try:
        good = next(filter(None, matching))
        return good
    except StopIteration:
        pass
    return None


#
# from typing import Optional, Match
# def matcher(text: str) -> Optional[Match[str]]:
#     patterns = [p1, p2]
#     matching = (p.search(text) for p in patterns)
#     try:
#         good = next(filter(None, matching))
#         return good
#     except StopIteration:
#         pass
#     return None

REPL_matcher = """
>>> import re
>>> p1 = re.compile(r"(\w+) text")
>>> p2 = re.compile(r"perhaps (\w+) text")

>>> matcher("some text", p1, p2)
<re.Match object; span=(0, 9), match='some text'>
"""


def test_matcher() -> None:
    import re

    p1 = re.compile(r"(\w+) text")
    p2 = re.compile(r"perhaps (\w+) content")

    text_1 = "some text"
    m_1 = matcher(text_1, p1, p2)
    assert m_1 and m_1.string == text_1

    m_2 = matcher("nothing", p1, p2)
    assert m_2 is None

    text_3 = "perhaps more content"
    m_3 = matcher(text_3, p1, p2)
    assert m_3 and m_3.string == text_3


__test__ = {name: value for name, value in globals().items() if name.startswith("REPL")}
