#!/usr/bin/env python3
"""Chemical-formula parsing shared by the materials evaluators."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from typing import Any


ELEMENT_TOKEN = re.compile(r"[A-Z][a-z]?")
AMOUNT_TOKEN = re.compile(r"(?:[0-9]+/[0-9]+|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og".split()
)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def parse_formula(formula: Any) -> dict[str, Fraction]:
    """Parse MP ``formula_pretty``, including parenthesized groups."""
    text = str(formula or "").replace(" ", "").strip()
    if not text:
        raise ValueError("empty formula")

    def amount_at(position: int) -> tuple[Fraction, int]:
        match = AMOUNT_TOKEN.match(text, position)
        if match is None:
            return Fraction(1), position
        amount = Fraction(match.group())
        if amount <= 0:
            raise ValueError(f"non-positive stoichiometry: {text!r}")
        return amount, match.end()

    def group_at(position: int, nested: bool) -> tuple[dict[str, Fraction], int]:
        counts: dict[str, Fraction] = defaultdict(Fraction)
        while position < len(text):
            if text[position] == ")":
                if not nested or not counts:
                    raise ValueError(f"unsupported formula syntax: {text!r}")
                return dict(counts), position + 1
            if text[position] == "(":
                subgroup, position = group_at(position + 1, nested=True)
                multiplier, position = amount_at(position)
                for element, amount in subgroup.items():
                    counts[element] += amount * multiplier
                continue
            element_match = ELEMENT_TOKEN.match(text, position)
            if element_match is None:
                raise ValueError(f"unsupported formula syntax: {text!r}")
            element = element_match.group()
            if element not in ELEMENT_SYMBOLS:
                raise ValueError(f"unknown element symbol {element!r} in formula: {text!r}")
            amount, position = amount_at(element_match.end())
            counts[element] += amount
        if nested:
            raise ValueError(f"unclosed parenthesis in formula: {text!r}")
        return dict(counts), position

    counts, position = group_at(0, nested=False)
    if position != len(text) or not counts:
        raise ValueError(f"unsupported formula syntax: {text!r}")
    return counts


def canonical_formula(formula: Any) -> tuple[str, str, str]:
    """Return (element-count key, reduced display formula, chemical system)."""
    counts = parse_formula(formula)
    denominator = reduce(lcm, (value.denominator for value in counts.values()), 1)
    integers = {element: int(value * denominator) for element, value in counts.items()}
    divisor = reduce(math.gcd, integers.values())
    integers = {element: value // divisor for element, value in integers.items()}
    elements = sorted(integers)
    key = "|".join(f"{element}:{integers[element]}" for element in elements)
    display = "".join(
        element + (str(integers[element]) if integers[element] != 1 else "")
        for element in elements
    )
    return key, display, "-".join(elements)
