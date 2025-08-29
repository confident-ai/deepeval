import pytest
import json
import tempfile
import os
from time import sleep
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.test_case import ToolCall, Turn
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
        dataset.push(alias=self.PUSH_ALIAS, overwrite=True)

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
        dataset.push(alias=self.PUSH_ALIAS, overwrite=True)

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


class TestSaveAndLoad:
    def test_dataset_save_load_goldens(self):
        """Saves Goldens in a file, reloads them from file and deletes the files"""
        goldens = [
            Golden(
                input="What is the tallest mountain in the world?",
                expected_output="Mount Everest",
                context=["Geography", "Mountains"],
                expected_tools=[
                    ToolCall(
                        name="GeoAPI",
                        description="Fetches geographical data",
                        reasoning="To retrieve mountain heights",
                        output="Mount Everest data",
                        input_parameters={"region": "Asia"}
                    )
                ],
                additional_metadata={"difficulty": "medium", "verified": True},
                comments="Basic geography question",
                custom_column_key_values={"category": "geography", "priority": "high"},
                actual_output="Mount Everest",
                retrieval_context=["Mountain heights list"],
                tools_called=[
                    ToolCall(
                        name="GeoAPI",
                        description="Fetches geographical data",
                        reasoning="To retrieve mountain heights",
                        output="Mount Everest data",
                        input_parameters={"region": "Asia"}
                    )
                ],
            ),
            Golden(
                input="Solve 5 * 7",
                expected_output="35",
                context=["Math", "Multiplication"],
                expected_tools=[
                    ToolCall(
                        name="Calculator",
                        description="Performs arithmetic calculations",
                        reasoning="To calculate product of numbers",
                        output="35",
                        input_parameters={"operation": "multiply", "operands": [5,7]}
                    )
                ],
                additional_metadata={"difficulty": "easy"},
                comments="Simple multiplication",
                custom_column_key_values={"category": "math", "priority": "medium"},
                actual_output="35",
                retrieval_context=["Basic arithmetic"],
                tools_called=[
                    ToolCall(
                        name="Calculator",
                        description="Performs arithmetic calculations",
                        reasoning="To calculate product of numbers",
                        output="35",
                        input_parameters={"operation": "multiply", "operands": [5,7]}
                    )
                ],
            ),
            Golden(
                input="Define 'photosynthesis'",
                expected_output="Process by which green plants convert light energy into chemical energy",
                context=["Biology", "Photosynthesis"],
                expected_tools=[
                    ToolCall(
                        name="EncyclopediaAPI",
                        description="Provides definitions and explanations",
                        reasoning="To retrieve the definition of photosynthesis",
                        output="Process by which green plants convert light energy into chemical energy",
                        input_parameters={"term": "photosynthesis"}
                    )
                ],
                additional_metadata={"difficulty": "hard"},
                comments="Science definition",
                custom_column_key_values={"category": "science", "priority": "medium"},
                actual_output="Process by which green plants convert light energy into chemical energy",
                retrieval_context=["Biology textbooks", "Science API"],
                tools_called=[
                    ToolCall(
                        name="EncyclopediaAPI",
                        description="Provides definitions and explanations",
                        reasoning="To retrieve the definition of photosynthesis",
                        output="Process by which green plants convert light energy into chemical energy",
                        input_parameters={"term": "photosynthesis"}
                    )
                ],
            ),
        ]

        json_file = "test_goldens.json"
        csv_file = "test_goldens.csv"
        directory = "."

        json_path = os.path.join(directory, json_file)
        csv_path = os.path.join(directory, csv_file)

        try:
            dataset = EvaluationDataset(goldens)
            dataset.save_as(file_type="json", file_name="test_goldens", directory=directory)
            dataset.save_as(file_type="csv", file_name="test_goldens", directory=directory)

            sleep(5)

            dataset1 = EvaluationDataset()
            dataset2 = EvaluationDataset()
            dataset1.add_goldens_from_json_file(file_path=json_path)
            dataset2.add_goldens_from_csv_file(file_path=csv_path)

            assert len(dataset1.goldens) == len(dataset2.goldens) == len(goldens)
            for i in range(len(goldens)):
                assert goldens[i].input == dataset1.goldens[i].input == dataset2.goldens[i].input
                assert goldens[i].expected_output == dataset1.goldens[i].expected_output == dataset2.goldens[i].expected_output
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_dataset_save_load_conversational_goldens(self):
        """Saves ConversationalGoldens in a file, reloads them from file and deletes the files"""
        convo_goldens = [
            ConversationalGolden(
                scenario="User asks for weather forecast for Paris",
                expected_outcome="User receives accurate weather forecast",
                user_description="User is planning a trip to Paris and wants to check the weather",
                context=["Weather", "Paris"],
                additional_metadata={"difficulty": "easy", "intent": "information_retrieval"},
                comments="Tests weather API integration",
                custom_column_key_values={"category": "weather", "priority": "high"},
                turns=[
                    Turn(
                        role="user",
                        content="What's the weather like in Paris this weekend?",
                        user_id="user_123",
                        retrieval_context=["Weather API docs"],
                        tools_called=[
                            ToolCall(
                                name="WeatherAPI",
                                description="Fetches weather data",
                                reasoning="User asked for weather forecast",
                                output="Rainy, 18°C",
                                input_parameters={"location": "Paris", "date_range": "weekend"}
                            )
                        ]
                    ),
                    Turn(
                        role="assistant",
                        content="It's expected to be rainy with temperatures around 18°C.",
                        retrieval_context=["Weather API response"],
                        tools_called=[]
                    ),
                ]
            ),
            ConversationalGolden(
                scenario="User translates a phrase from English to Japanese",
                expected_outcome="Assistant correctly translates the phrase",
                user_description="User needs a quick translation",
                context=["Translation", "English to Japanese"],
                additional_metadata={"intent": "translation"},
                comments="Tests translation capabilities",
                custom_column_key_values={"category": "language", "priority": "medium"},
                turns=[
                    Turn(
                        role="user",
                        content="How do you say 'thank you' in Japanese?",
                        retrieval_context=["Translation database"],
                        tools_called=[]
                    ),
                    Turn(
                        role="assistant",
                        content="You can say 'ありがとう' (arigatou).",
                        retrieval_context=[],
                        tools_called=[
                            ToolCall(
                                name="Translator",
                                description="Translates text between languages",
                                reasoning="Translate 'thank you' to Japanese",
                                output="ありがとう",
                                input_parameters={"text": "thank you", "target_lang": "ja"}
                            )
                        ]
                    ),
                ]
            ),
            ConversationalGolden(
                scenario="User books a restaurant reservation",
                expected_outcome="Reservation is confirmed",
                user_description="User wants to book a dinner reservation for 2",
                context=["Restaurants", "Booking"],
                additional_metadata={"channel": "voice_assistant", "intent": "booking"},
                comments="Tests integration with reservation system",
                custom_column_key_values={"category": "hospitality", "priority": "critical"},
                turns=[
                    Turn(
                        role="user",
                        content="Can you book a table for 2 at 7 PM tonight at Luigi's?",
                        user_id="user_999",
                        retrieval_context=["Reservation service API"],
                        tools_called=[]
                    ),
                    Turn(
                        role="assistant",
                        content="Done! Your table for 2 at Luigi's is confirmed for 7 PM.",
                        retrieval_context=["Booking confirmation"],
                        tools_called=[
                            ToolCall(
                                name="ReservationAPI",
                                description="Books restaurant reservations",
                                reasoning="User requested a dinner reservation",
                                output="Confirmed reservation at Luigi's at 7 PM for 2 people",
                                input_parameters={
                                    "restaurant": "Luigi's", "time": "7 PM", "party_size": 2
                                }
                            )
                        ]
                    ),
                ]
            ),
        ]

        json_file = "convo_goldens.json"
        csv_file = "convo_goldens.csv"
        directory = "."

        json_path = os.path.join(directory, json_file)
        csv_path = os.path.join(directory, csv_file)

        try:
            dataset = EvaluationDataset(convo_goldens)
            dataset.save_as(file_type="json", file_name="convo_goldens", directory=directory)
            dataset.save_as(file_type="csv", file_name="convo_goldens", directory=directory)

            sleep(2)

            dataset1 = EvaluationDataset()
            dataset2 = EvaluationDataset()
            dataset1.add_goldens_from_json_file(file_path=json_path)
            dataset2.add_goldens_from_csv_file(file_path=csv_path)

            assert len(dataset1.goldens) == len(dataset2.goldens) == len(convo_goldens)
            for i in range(len(convo_goldens)):
                assert convo_goldens[i].scenario == dataset1.goldens[i].scenario == dataset2.goldens[i].scenario
                assert convo_goldens[i].expected_outcome == dataset1.goldens[i].expected_outcome == dataset2.goldens[i].expected_outcome
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)