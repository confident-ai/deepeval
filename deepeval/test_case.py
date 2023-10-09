"""Investigate test case.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class TestCase:
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            id_string = "".join(str(value) for value in self.__dict__.values())
            self.id = hashlib.md5(id_string.encode()).hexdigest()


@dataclass
class LLMTestCase(TestCase):
    def __init__(
        self,
        query: str,
        output: str,
        expected_output: str = "-",
        context: Optional[Union[str, List[str]]] = None,
        id: Optional[str] = None,
    ):
        super().__init__(id)
        self.query = query
        self.output = output
        self.expected_output = expected_output
        # Force context to be a list
        if isinstance(context, str):
            context = [context]
        self.context = context

    def __post_init__(self):
        super().__post_init__()
        self.__name__ = f"LLMTestCase_{self.id}"


@dataclass
class SearchTestCase(TestCase):
    def __init__(
        self,
        output_list: List[Any],
        golden_list: List[Any],
        query: Optional[str] = None,
        id: Optional[str] = None,
    ):
        super().__init__(id)
        self.output_list = output_list
        self.golden_list = golden_list
        self.query = query


class AgentTestCase(TestCase):
    """Test Case For Agents"""

    pass


@dataclass
class ImageTestCase(TestCase):
    """Test Case For Images. This is a beta interface and is subject to change."""

    def __init__(
        self,
        image_path: str,
        query: Optional[str] = None,
        ground_truth_image_path: Optional[str] = None,
        id: Optional[str] = None,
    ):
        self.query = query
        self.image_path = image_path
        self.ground_truth_image_path = ground_truth_image_path
        super().__init__(id)
