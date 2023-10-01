"""Providing modifier abstractions
"""
from typing import Any
from abc import abstractmethod


class Modifier:
    """Providing modifiers."""

    @abstractmethod
    def modify(self, input: Any):
        self.input = input


class QueryModifier(Modifier):
    """Add a query modifier based on the result."""
