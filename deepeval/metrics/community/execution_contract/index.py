"""Subject-binding candidate indexes with conservative missing-value joins."""

from __future__ import annotations

from collections import defaultdict
from itertools import chain

from .selectors import MISSING, _pointer_parts, _resolve_parts


def _value_key(value, charge):
    """A hashable key with the exact semantics of typed JSON equality.

    The input was validated by Snapshot. Charges include nested values and
    string lengths so indexing large subject objects is not unmetered work.
    Float signed zero compares equal; booleans, integers and floats are distinct.
    """
    charge()
    kind = type(value)
    if value is None:
        return ("null",)
    if kind is bool:
        return ("bool", value)
    if kind is int:
        charge(max(1, value.bit_length() // 64))
        return ("int", value)
    if kind is float:
        return ("float", value)
    if kind is str:
        charge(max(1, len(value) // 64))
        return ("str", value)
    if kind is list:
        return ("list", tuple(_value_key(item, charge) for item in value))
    if kind is dict:
        return (
            "dict",
            tuple(
                (_value_key(key, charge), _value_key(value[key], charge))
                for key in sorted(value)
            ),
        )
    raise ValueError("binding index requires validated JSON values")


class BindingIndex:
    """Reduce a binding join without excluding uncertain candidates.

    Every binding column has observed-value postings and a missing-value list.
    A known target field searches its exact posting plus missing candidates.
    A missing target field cannot narrow candidates. Use the narrowest known
    column, then filter contradictions in remaining columns. The evaluator
    still decides YES/UNKNOWN after this prefilter, including selector and
    temporal uncertainty; the index never turns missing values into matches.
    """

    def __init__(self, candidates, payloads, bindings, charge):
        self._candidates = tuple(candidates)
        self._payloads = payloads
        self._charge = charge
        self._left_parts = tuple(_pointer_parts(left) for left, _ in bindings)
        right_parts = tuple(_pointer_parts(right) for _, right in bindings)
        self._postings = [defaultdict(list) for _ in bindings]
        self._missing = [[] for _ in bindings]
        self._keys = []
        for index, candidate in enumerate(self._candidates):
            charge()
            keys = []
            for column, parts in enumerate(right_parts):
                charge(max(1, len(parts)))
                value = _resolve_parts(payloads[candidate.id], parts)
                key = MISSING if value is MISSING else _value_key(value, charge)
                keys.append(key)
                if key is MISSING:
                    self._missing[column].append(index)
                else:
                    self._postings[column][key].append(index)
            self._keys.append(tuple(keys))

    def candidates(self, target):
        keys, choices = [], []
        for column, parts in enumerate(self._left_parts):
            self._charge(max(1, len(parts)))
            value = _resolve_parts(self._payloads[target.id], parts)
            key = (
                MISSING if value is MISSING else _value_key(value, self._charge)
            )
            keys.append(key)
            if key is not MISSING:
                size = len(self._postings[column].get(key, ())) + len(
                    self._missing[column]
                )
                choices.append((size, column))
        if choices:
            _, column = min(choices)
            indexes = chain(
                self._postings[column].get(keys[column], ()),
                self._missing[column],
            )
        else:
            indexes = range(len(self._candidates))
        # Preserve source ordering independently of which column was narrowest.
        matched = []
        for index in indexes:
            self._charge(max(1, len(keys)))
            candidate_keys = self._keys[index]
            if all(
                left is MISSING or right is MISSING or left == right
                for left, right in zip(keys, candidate_keys)
            ):
                matched.append(index)
        for index in sorted(matched):
            yield self._candidates[index]
