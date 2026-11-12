import re
from fractions import Fraction

_SUFFIX = {
    "n": Fraction(1, 10**9), "u": Fraction(1, 10**6), "m": Fraction(1, 1000),
    "": Fraction(1),
    "k": Fraction(10**3), "K": Fraction(10**3), "M": Fraction(10**6),
    "G": Fraction(10**9), "T": Fraction(10**12), "P": Fraction(10**15),
    "E": Fraction(10**18),
    "Ki": Fraction(2**10), "Mi": Fraction(2**20), "Gi": Fraction(2**30),
    "Ti": Fraction(2**40), "Pi": Fraction(2**50), "Ei": Fraction(2**60),
}
_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))(?:([eE][+-]?\d+)|(Ki|Mi|Gi|Ti|Pi|Ei|[numkKMGTPE]))?$")


def parse_quantity(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return Fraction(value).limit_denominator(10**12) if isinstance(value, float) else Fraction(value)
    if not isinstance(value, str):
        return None
    m = _RE.match(value.strip())
    if not m:
        return None
    num, exp, suf = m.groups()
    base = Fraction(num)
    if exp:
        return base * (Fraction(10) ** int(exp[1:]))
    return base * _SUFFIX[suf or ""]
