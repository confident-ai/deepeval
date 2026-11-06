"""Safe, three-valued matching of immutable execution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Iterable, Optional, Tuple

from .evidence import EvidenceLimits, SpanRecord, canonical_json


class Match(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class _Missing:
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


def _pointer_parts(pointer: str) -> Tuple[str, ...]:
    if not isinstance(pointer, str) or len(pointer) > 4096:
        raise ValueError(
            "JSON pointer must be a string of at most 4096 characters"
        )
    if not pointer:
        return ()
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with '/' or be empty")
    parts = pointer[1:].split("/")
    if len(parts) > 128:
        raise ValueError("JSON pointer depth limit exceeded")
    decoded = []
    for part in parts:
        index = 0
        while index < len(part):
            if part[index] == "~":
                if index + 1 == len(part) or part[index + 1] not in "01":
                    raise ValueError("invalid JSON pointer escape")
                index += 2
            else:
                index += 1
        decoded.append(part.replace("~1", "/").replace("~0", "~"))
    return tuple(decoded)


def resolve_pointer(value: Any, pointer: str) -> Any:
    """Resolve RFC 6901 object/array segments without evaluating expressions.

    An invalid pointer is a configuration error. A valid pointer with no matching
    value returns MISSING, which is distinct from a present JSON null.
    """
    return _resolve_parts(value, _pointer_parts(pointer))


def _resolve_parts(value: Any, parts: Tuple[str, ...]) -> Any:
    for part in parts:
        if type(value) is dict:
            value = value.get(part, MISSING)
        elif type(value) is list:
            if (
                not part
                or not part.isascii()
                or not part.isdecimal()
                or (len(part) > 1 and part[0] == "0")
                or len(part) > len(str(len(value)))
            ):
                return MISSING
            index = int(part)
            value = value[index] if index < len(value) else MISSING
        else:
            return MISSING
        if value is MISSING:
            return MISSING
    return value


def _no_charge(amount=1):
    pass


def json_equal(left: Any, right: Any, *, charge=_no_charge) -> bool:
    """Compare validated JSON values without Python's bool/int coercion."""
    stack = [(left, right)]
    while stack:
        first, second = stack.pop()
        charge()
        if type(first) is not type(second):
            return False
        if type(first) is dict:
            charge(len(first) + sum(len(key) // 64 for key in first))
            if first.keys() != second.keys():
                return False
            stack.extend((value, second[key]) for key, value in first.items())
        elif type(first) is list:
            charge(len(first))
            if len(first) != len(second):
                return False
            stack.extend(zip(first, second))
        else:
            if type(first) is str:
                charge(max(1, len(first) // 64))
            if first != second:
                return False
    return True


@dataclass(frozen=True, init=False)
class Predicate:
    pointer: str
    operator: str
    expected_json: str
    _parts: Tuple[str, ...]

    def __init__(self, pointer: str, operator: str, expected: Any):
        parts = _pointer_parts(pointer)
        if operator not in ("eq", "ne", "exists"):
            raise ValueError("predicate operator must be eq, ne or exists")
        if operator == "exists" and type(expected) is not bool:
            raise ValueError("exists predicate expects a boolean")
        encoded = canonical_json(expected, EvidenceLimits())
        object.__setattr__(self, "pointer", pointer)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "expected_json", encoded)
        object.__setattr__(self, "_parts", parts)

    @property
    def expected(self) -> Any:
        return json.loads(self.expected_json)

    def matches(
        self, payload: Any, *, expected=MISSING, charge=_no_charge
    ) -> Match:
        charge(max(1, len(self._parts)))
        observed = _resolve_parts(payload, self._parts)
        if expected is MISSING:
            expected = self.expected
        if self.operator == "exists":
            exists = observed is not MISSING
            return Match.YES if exists == expected else Match.NO
        if observed is MISSING:
            return Match.UNKNOWN
        same = json_equal(observed, expected, charge=charge)
        if self.operator == "ne":
            same = not same
        return Match.YES if same else Match.NO


@dataclass(frozen=True)
class Selector:
    name: Optional[str] = None
    kind: Optional[str] = None
    status: Optional[str] = None
    predicates: Tuple[Predicate, ...] = ()

    def __post_init__(self):
        for field in ("name", "kind", "status"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"selector {field} must be a nonempty string")
        predicates = []
        for predicate in self.predicates:
            if len(predicates) >= 64 or type(predicate) is not Predicate:
                raise ValueError("selector requires at most 64 predicates")
            predicates.append(predicate)
        object.__setattr__(self, "predicates", tuple(predicates))

    def matches(
        self,
        span: SpanRecord,
        *,
        payload=MISSING,
        expected_values=None,
        charge=_no_charge,
    ) -> Match:
        unknown = False
        for field in ("name", "kind", "status"):
            expected = getattr(self, field)
            if expected is None:
                continue
            actual = getattr(span, field)
            charge(max(1, len(expected) // 64))
            if actual is None or (field == "status" and actual == "unknown"):
                unknown = True
            elif actual != expected:
                return Match.NO
        if payload is MISSING:
            payload = span.payload
        for index, predicate in enumerate(self.predicates):
            expected = (
                expected_values[index]
                if expected_values is not None
                else MISSING
            )
            result = predicate.matches(
                payload, expected=expected, charge=charge
            )
            if result is Match.NO:
                return Match.NO
            unknown |= result is Match.UNKNOWN
        return Match.UNKNOWN if unknown else Match.YES


def match_bindings(
    left: Any,
    right: Any,
    bindings: Iterable[Tuple[str, str]],
    *,
    charge=_no_charge,
) -> Match:
    """Join operation subjects using explicit JSON pointer pairs.

    One unequal observed field rules out the join, even when another field is
    missing. Missing fields never act as wildcards.
    """
    unknown = False
    for index, (left_pointer, right_pointer) in enumerate(bindings):
        if index >= 64:
            raise ValueError("binding limit exceeded")
        charge(max(1, (len(left_pointer) + len(right_pointer)) // 64))
        first = resolve_pointer(left, left_pointer)
        second = resolve_pointer(right, right_pointer)
        if first is MISSING or second is MISSING:
            unknown = True
        elif not json_equal(first, second, charge=charge):
            return Match.NO
    return Match.UNKNOWN if unknown else Match.YES
