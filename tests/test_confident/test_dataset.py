import pytest
import json
import os
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.test_case import ToolCall
from collections import Counter


def create_tool_calls_from_data(tools_data):
    """Convert JSON tool data to ToolCall objects"""
    if not tools_data:
        return None

    tool_calls = []
    for tool_data in tools_data:
        if isinstance(tool_data, dict) and "name" in tool_data:
            tool_call = ToolCall(
                name=tool_data["name"], input=tool_data.get("input", None)
            )
            tool_calls.append(tool_call)
    return tool_calls


def load_goldens_data(path: str):
    """Load golden data from JSON file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, path)

    with open(json_path, "r") as f:
        return json.load(f)


def deep_equal_unordered(a, b):
    """Compare two objects, handling Pydantic models and unordered lists"""
    from pydantic import BaseModel

    # Handle Pydantic models by converting to dict
    if isinstance(a, BaseModel) and isinstance(b, BaseModel):
        return deep_equal_unordered(a.model_dump(), b.model_dump())
    elif isinstance(a, BaseModel):
        return deep_equal_unordered(a.model_dump(), b)
    elif isinstance(b, BaseModel):
        return deep_equal_unordered(a, b.model_dump())

    # Handle lists (order doesn't matter)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        # For small lists, use simple comparison
        if len(a) <= 10:
            a_sorted = sorted(a, key=lambda x: str(freeze_for_comparison(x)))
            b_sorted = sorted(b, key=lambda x: str(freeze_for_comparison(x)))
            return all(
                deep_equal_unordered(x, y) for x, y in zip(a_sorted, b_sorted)
            )
        else:
            # For larger lists, use Counter approach
            return Counter(
                map(lambda x: freeze_for_comparison(x), a)
            ) == Counter(map(lambda x: freeze_for_comparison(x), b))

    # Handle dictionaries
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            deep_equal_unordered(a[k], b[k]) for k in a
        )

    # Base case: direct comparison
    return a == b


def freeze_for_comparison(obj):
    """Convert object to hashable form for comparison"""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return freeze_for_comparison(obj.model_dump())
    elif isinstance(obj, dict):
        return tuple(
            sorted((k, freeze_for_comparison(v)) for k, v in obj.items())
        )
    elif isinstance(obj, list):
        return tuple(freeze_for_comparison(x) for x in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For other types, convert to string
        return str(obj)


class TestSingleTurnDataset:

    PUSH_ALIAS = "test_single_turn_realistic_push"
    QUEUE_ALIAS = "test_single_turn_realistic_queue"

    def create_golden_from_data(self, data):
        """Create a Golden object from JSON data"""
        return Golden(
            input=data.get("input", None),
            actual_output=data.get("actual_output", None),
            expected_output=data.get("expected_output", None),
            context=data.get("context", None),
            retrieval_context=data.get("retrieval_context", None),
            additional_metadata=data.get("additional_metadata", None),
            comments=data.get("comments", None),
            tools_called=create_tool_calls_from_data(
                data.get("tools_called", None)
            ),
            expected_tools=create_tool_calls_from_data(
                data.get("expected_tools", None)
            ),
            custom_column_key_values=data.get("custom_column_key_values", None),
        )

    def test_dataset_push_pull(self):
        goldens_data = load_goldens_data("goldens.json")

        initial_goldens = []
        for data in goldens_data:
            golden = self.create_golden_from_data(data)
            initial_goldens.append(golden)

        dataset = EvaluationDataset(goldens=initial_goldens)
        dataset.delete(alias=self.PUSH_ALIAS)
        dataset.push(alias=self.PUSH_ALIAS)

        dataset.goldens = []
        dataset.pull(alias=self.PUSH_ALIAS)

        assert len(dataset.goldens) == len(initial_goldens)
        assert deep_equal_unordered(dataset.goldens, initial_goldens)


class TestMultiTurnDataset:

    PUSH_ALIAS = "test_multi_turn_realistic_push"
    QUEUE_ALIAS = "test_multi_turn_realistic_queue"

    def create_golden_from_data(self, data):
        """Create a Golden object from JSON data"""
        return ConversationalGolden(
            scenario=data.get("scenario", None),
            expected_outcome=data.get("expected_outcome", None),
            user_description=data.get("user_description", None),
            context=data.get("context", None),
            additional_metadata=data.get("additional_metadata", None),
            comments=data.get("comments", None),
            turns=data.get("turns", None),
            custom_column_key_values=data.get("custom_column_key_values", None),
        )

    def test_dataset_push_pull(self):
        goldens_data = load_goldens_data("goldens_multi_turn.json")

        initial_goldens = []
        for data in goldens_data:
            golden = self.create_golden_from_data(data)
            initial_goldens.append(golden)

        dataset = EvaluationDataset(goldens=initial_goldens)
        dataset.delete(alias=self.PUSH_ALIAS)
        dataset.push(alias=self.PUSH_ALIAS)

        dataset.goldens = []
        dataset.pull(alias=self.PUSH_ALIAS)

        assert len(dataset.goldens) == len(initial_goldens)
        assert deep_equal_unordered(dataset.goldens, initial_goldens)

    # def test_dataset_queue(self):
    #     goldens_data = load_goldens_data("goldens_multi_turn.json")

    #     initial_goldens = []
    #     for data in goldens_data:
    #         golden = self.create_golden_from_data(data)
    #         initial_goldens.append(golden)

    #     dataset = EvaluationDataset()
    #     dataset.queue(alias=self.QUEUE_ALIAS, goldens=initial_goldens)
    #     dataset.goldens = []

    #     with pytest.raises(Exception):
    #         dataset.pull(alias=self.QUEUE_ALIAS)

    #     dataset.pull(alias=self.QUEUE_ALIAS, finalized=False)
    #     assert len(dataset.goldens) == len(initial_goldens)
    #     assert deep_equal_unordered(dataset.goldens, initial_goldens)
