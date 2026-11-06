"""Declarative execution contracts, validated before evaluating a trace."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple, Union

from .evidence import _identifier
from .selectors import Selector, _pointer_parts


def _selector(value):
    if type(value) is not Selector:
        raise ValueError("rule selectors must be Selector instances")


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _duration(value, label):
    if value is None:
        return
    try:
        valid = (
            type(value) in (int, float) and math.isfinite(value) and value >= 0
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite nonnegative number")


def _bindings(value):
    pairs = []
    for pair in value:
        if len(pairs) >= 64:
            raise ValueError("binding limit exceeded")
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each binding must contain two JSON pointers")
        for pointer in pair:
            _pointer_parts(pointer)
        pairs.append(tuple(pair))
    return tuple(pairs)


@dataclass(frozen=True)
class EvaluationLimits:
    max_rules: int = 256
    max_work: int = 1000000
    max_decisions: int = 100
    max_witnesses: int = 16

    def __post_init__(self):
        for name, value in vars(self).items():
            _count(value, name)
            if value == 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Occurrence:
    """Bound how often a selector matches; maximum=0 prohibits matches."""

    id: str
    selector: Selector
    minimum: int = 1
    maximum: Optional[int] = None
    required: bool = True

    def __post_init__(self):
        _identifier(self.id, "rule id")
        _selector(self.selector)
        _count(self.minimum, "minimum")
        if self.maximum is not None:
            _count(self.maximum, "maximum")
            if self.maximum < self.minimum:
                raise ValueError("maximum must not be less than minimum")
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")


@dataclass(frozen=True)
class Prerequisite:
    """Require a completed matching operation before each target starts.

    Bindings are (target pointer, prerequisite pointer) pairs. A prerequisite
    can authorize multiple targets. Optional invalidators use the same binding
    schema; overlapping verification/revocation windows remain ambiguous.
    """

    id: str
    target: Selector
    prerequisite: Selector
    bindings: Tuple[Tuple[str, str], ...] = ()
    max_age: Optional[float] = None
    invalidated_by: Optional[Selector] = None
    required: bool = True

    def __post_init__(self):
        _identifier(self.id, "rule id")
        _selector(self.target)
        _selector(self.prerequisite)
        if self.invalidated_by is not None:
            _selector(self.invalidated_by)
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        _duration(self.max_age, "max_age")
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")


@dataclass(frozen=True)
class Recovery:
    """Require a fresh completed recovery for each selected failed operation.

    Bindings should include an operation identity when accounts can have multiple
    operations. Recovery evidence is reusable for the same declared identity.
    """

    id: str
    failure: Selector
    recovery: Selector
    bindings: Tuple[Tuple[str, str], ...] = ()
    within: Optional[float] = None
    before: Optional[Selector] = None
    required: bool = True

    def __post_init__(self):
        _identifier(self.id, "rule id")
        _selector(self.failure)
        _selector(self.recovery)
        if self.before is not None:
            _selector(self.before)
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        if not self.bindings:
            raise ValueError("recovery requires explicit operation bindings")
        _duration(self.within, "within")
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")


@dataclass(frozen=True)
class Budget:
    """Sum observed nonnegative resources in a declared capture scope.

    By default selected ancestors and descendants cannot be summed together.
    exclusive=True is a caller contract that each value excludes child totals.
    """

    id: str
    selector: Selector
    pointer: str
    maximum: float
    exclusive: bool = False
    required: bool = True

    def __post_init__(self):
        _identifier(self.id, "rule id")
        _selector(self.selector)
        _pointer_parts(self.pointer)
        if self.maximum is None:
            raise ValueError("maximum resource budget is required")
        _duration(self.maximum, "maximum")
        if type(self.exclusive) is not bool or type(self.required) is not bool:
            raise ValueError("exclusive and required must be booleans")


Rule = Union[Occurrence, Prerequisite, Recovery, Budget]


@dataclass(frozen=True)
class Contract:
    rules: Tuple[Rule, ...]
    limits: EvaluationLimits = field(default_factory=EvaluationLimits)

    def __post_init__(self):
        if type(self.limits) is not EvaluationLimits:
            raise ValueError("limits must be EvaluationLimits")
        rules = []
        identifiers = set()
        for rule in self.rules:
            if len(rules) >= self.limits.max_rules:
                raise ValueError("contract rule limit exceeded")
            if type(rule) not in (Occurrence, Prerequisite, Recovery, Budget):
                raise ValueError("unsupported contract rule")
            if rule.id in identifiers:
                raise ValueError("duplicate rule ID")
            identifiers.add(rule.id)
            rules.append(rule)
        if not rules:
            raise ValueError("a contract must contain at least one rule")
        object.__setattr__(self, "rules", tuple(rules))
