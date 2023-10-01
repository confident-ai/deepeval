"""Investigate test case.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, List, Optional, Union
from PIL import Image


@dataclass
class TestCase:
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            id_string = "".join(str(value) for value in self.__dict__.values())
            self.id = hashlib.md5(id_string.encode()).hexdigest()


@dataclass
class LLMTestCase(TestCase):
    query: str = "-"
    expected_output: str = "-"
    context: Optional[Union[str, List[str]]] = None
    output: str = "-"

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
        image: str,
        query: Optional[str] = None,
        ground_truth_image: Optional[str] = None,
        minimum_score: float = 0.3,
        id: Optional[str] = None,
    ):
        self.query = query
        self.image = image
        self.ground_truth_image = Image.open(ground_truth_image)
        self.minimum_score = minimum_score
        super().__init__(id)
