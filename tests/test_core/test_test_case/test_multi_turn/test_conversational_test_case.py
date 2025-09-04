import pytest
from pydantic import ValidationError
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams


class TestConversationalTestCaseInitialization:

    def test_minimal_initialization(self):
        turns = [Turn(role="user", content="Hello")]
        test_case = ConversationalTestCase(turns=turns)

        assert len(test_case.turns) == 1
        assert test_case.turns[0].role == "user"
        assert test_case.turns[0].content == "Hello"
        assert test_case.scenario is None
        assert test_case.context is None
        assert test_case.name is None
        assert test_case.user_description is None
        assert test_case.expected_outcome is None
        assert test_case.chatbot_role is None
        assert test_case.additional_metadata is None
        assert test_case.comments is None
        assert test_case.tags is None
        assert test_case.mcp_servers is None

    def test_full_initialization(self):
        turns = [
            Turn(role="user", content="Hello"),
            Turn(role="assistant", content="Hi there!"),
        ]

        test_case = ConversationalTestCase(
            turns=turns,
            scenario="Customer support interaction",
            context=[
                "Previous conversation history",
                "User is a premium customer",
            ],
            name="Support Chat Test",
            user_description="Frustrated customer with billing issue",
            expected_outcome="Issue resolved satisfactorily",
            chatbot_role="Helpful customer service agent",
            additional_metadata={"priority": "high", "department": "billing"},
            comments="Test case for billing dispute resolution",
            tags=["billing", "dispute", "premium"],
        )

        assert len(test_case.turns) == 2
        assert test_case.scenario == "Customer support interaction"
        assert test_case.context == [
            "Previous conversation history",
            "User is a premium customer",
        ]
        assert test_case.name == "Support Chat Test"
        assert (
            test_case.user_description
            == "Frustrated customer with billing issue"
        )
        assert test_case.expected_outcome == "Issue resolved satisfactorily"
        assert test_case.chatbot_role == "Helpful customer service agent"
        assert test_case.additional_metadata == {
            "priority": "high",
            "department": "billing",
        }
        assert test_case.comments == "Test case for billing dispute resolution"
        assert test_case.tags == ["billing", "dispute", "premium"]

    def test_turns_deep_copy(self):
        original_turn = Turn(role="user", content="Hello")
        turns = [original_turn]
        test_case = ConversationalTestCase(turns=turns)

        test_case.turns[0].content = "Modified"
        assert original_turn.content == "Hello"


class TestConversationalTestCaseValidation:

    def test_empty_turns_raises_error(self):
        with pytest.raises(TypeError, match="'turns' must not be empty"):
            ConversationalTestCase(turns=[])

    def test_non_turn_objects_raises_error(self):
        with pytest.raises(TypeError):
            ConversationalTestCase(turns=["not a turn"])

    def test_dict_turn_is_accepted(self):
        case = ConversationalTestCase(turns=[{"role": "user", "content": "hi"}])
        assert isinstance(case.turns[0], Turn)

    def test_invalid_context_type_raises_error(self):
        turns = [Turn(role="user", content="Hello")]
        with pytest.raises(
            TypeError, match="'context' must be None or a list of strings"
        ):
            ConversationalTestCase(turns=turns, context="not a list")

    def test_invalid_context_items_raises_error(self):
        turns = [Turn(role="user", content="Hello")]
        with pytest.raises(
            TypeError, match="'context' must be None or a list of strings"
        ):
            ConversationalTestCase(
                turns=turns, context=["valid", 123, "invalid"]
            )

    def test_valid_context_list(self):
        turns = [Turn(role="user", content="Hello")]
        context = ["Context item 1", "Context item 2"]
        test_case = ConversationalTestCase(turns=turns, context=context)
        assert test_case.context == context

    def test_none_context_is_valid(self):
        turns = [Turn(role="user", content="Hello")]
        test_case = ConversationalTestCase(turns=turns, context=None)
        assert test_case.context is None


class TestConversationalTestCaseComplexScenarios:

    def test_single_user_turn(self):
        turns = [Turn(role="user", content="What's the weather?")]
        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns) == 1
        assert test_case.turns[0].role == "user"

    def test_single_assistant_turn(self):
        turns = [Turn(role="assistant", content="Hello! How can I help?")]
        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns) == 1
        assert test_case.turns[0].role == "assistant"

    def test_alternating_conversation(self):
        turns = [
            Turn(role="user", content="Hi"),
            Turn(role="assistant", content="Hello!"),
            Turn(role="user", content="How are you?"),
            Turn(role="assistant", content="I'm doing well, thanks!"),
        ]
        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns) == 4

        for i, turn in enumerate(test_case.turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role

    def test_consecutive_user_turns(self):
        turns = [
            Turn(role="user", content="First message"),
            Turn(role="user", content="Second message"),
            Turn(role="assistant", content="Response to both"),
        ]
        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns) == 3
        assert test_case.turns[0].role == "user"
        assert test_case.turns[1].role == "user"
        assert test_case.turns[2].role == "assistant"

    def test_consecutive_assistant_turns(self):
        turns = [
            Turn(role="user", content="Question"),
            Turn(role="assistant", content="First response"),
            Turn(role="assistant", content="Additional clarification"),
        ]
        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns) == 3
        assert test_case.turns[0].role == "user"
        assert test_case.turns[1].role == "assistant"
        assert test_case.turns[2].role == "assistant"

    def test_long_conversation(self):
        turns = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message {i+1} from {role}"
            turns.append(Turn(role=role, content=content))

        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns) == 20


class TestConversationalTestCaseTurnProperties:

    def test_turns_with_metadata(self):
        turns = [
            Turn(
                role="user",
                content="Hello",
                user_id="user123",
                additional_metadata={"timestamp": "2024-01-01T10:00:00Z"},
            ),
            Turn(
                role="assistant",
                content="Hi there!",
                additional_metadata={"model": "gpt-4", "tokens": 5},
            ),
        ]

        test_case = ConversationalTestCase(turns=turns)
        assert test_case.turns[0].user_id == "user123"
        assert (
            test_case.turns[0].additional_metadata["timestamp"]
            == "2024-01-01T10:00:00Z"
        )
        assert test_case.turns[1].additional_metadata["model"] == "gpt-4"

    def test_turns_with_retrieval_context(self):
        turns = [
            Turn(role="user", content="What's the capital of France?"),
            Turn(
                role="assistant",
                content="The capital of France is Paris.",
                retrieval_context=[
                    "France is a country in Europe",
                    "Paris is the largest city in France",
                ],
            ),
        ]

        test_case = ConversationalTestCase(turns=turns)
        assert test_case.turns[1].retrieval_context is not None
        assert len(test_case.turns[1].retrieval_context) == 2
        assert (
            "Paris is the largest city in France"
            in test_case.turns[1].retrieval_context
        )


class TestConversationalTestCaseEdgeCases:

    def test_empty_content_turns(self):
        turns = [
            Turn(role="user", content=""),
            Turn(role="assistant", content=""),
        ]
        test_case = ConversationalTestCase(turns=turns)
        assert test_case.turns[0].content == ""
        assert test_case.turns[1].content == ""

    def test_very_long_content(self):
        long_content = "A" * 10000
        turns = [Turn(role="user", content=long_content)]
        test_case = ConversationalTestCase(turns=turns)
        assert len(test_case.turns[0].content) == 10000

    def test_special_characters_in_content(self):
        special_content = "Hello! 🌟 @#$%^&*() 你好 🎉"
        turns = [Turn(role="user", content=special_content)]
        test_case = ConversationalTestCase(turns=turns)
        assert test_case.turns[0].content == special_content

    def test_multiline_content(self):
        multiline_content = """This is a
        multiline
        message with
        various indentation"""
        turns = [Turn(role="user", content=multiline_content)]
        test_case = ConversationalTestCase(turns=turns)
        assert "\n" in test_case.turns[0].content

    def test_empty_tags_list(self):
        turns = [Turn(role="user", content="Hello")]
        test_case = ConversationalTestCase(turns=turns, tags=[])
        assert test_case.tags == []

    def test_empty_additional_metadata(self):
        turns = [Turn(role="user", content="Hello")]
        test_case = ConversationalTestCase(turns=turns, additional_metadata={})
        assert test_case.additional_metadata == {}


class TestConversationalTestCaseEquality:

    def test_identical_test_cases_are_equal(self):
        turns1 = [Turn(role="user", content="Hello")]
        turns2 = [Turn(role="user", content="Hello")]

        test_case1 = ConversationalTestCase(turns=turns1, scenario="Test")
        test_case2 = ConversationalTestCase(turns=turns2, scenario="Test")

        assert test_case1.model_dump() == test_case2.model_dump()

    def test_different_turns_are_not_equal(self):
        turns1 = [Turn(role="user", content="Hello")]
        turns2 = [Turn(role="user", content="Hi")]

        test_case1 = ConversationalTestCase(turns=turns1)
        test_case2 = ConversationalTestCase(turns=turns2)

        assert test_case1.model_dump() != test_case2.model_dump()

    def test_different_scenarios_are_not_equal(self):
        turns = [Turn(role="user", content="Hello")]

        test_case1 = ConversationalTestCase(turns=turns, scenario="Scenario A")
        test_case2 = ConversationalTestCase(turns=turns, scenario="Scenario B")

        assert test_case1.model_dump() != test_case2.model_dump()


class TestConversationalTestCaseSerialization:

    def test_model_dump_includes_all_fields(self):
        turns = [
            Turn(role="user", content="Hello"),
            Turn(role="assistant", content="Hi!"),
        ]

        test_case = ConversationalTestCase(
            turns=turns,
            scenario="Test scenario",
            name="Test name",
            tags=["tag1", "tag2"],
        )

        dumped = test_case.model_dump()
        assert "turns" in dumped
        assert "scenario" in dumped
        assert "name" in dumped
        assert "tags" in dumped
        assert len(dumped["turns"]) == 2

    def test_serialization_aliases(self):
        turns = [Turn(role="user", content="Hello")]

        test_case = ConversationalTestCase(
            turns=turns,
            user_description="Test user",
            expected_outcome="Success",
            chatbot_role="Assistant",
            additional_metadata={"key": "value"},
        )

        dumped = test_case.model_dump(by_alias=True)
        assert "userDescription" in dumped
        assert "expectedOutcome" in dumped
        assert "chatbotRole" in dumped
        assert "additionalMetadata" in dumped


class TestConversationalTestCaseCamelCaseInitialization:

    def test_camelcase_field_initialization(self):
        # Test data variables
        scenario_text = "Customer support interaction"
        context_list = [
            "Previous conversation history",
            "User is a premium customer",
        ]
        name_text = "Support Chat Test"
        user_description_text = "Frustrated customer with billing issue"
        expected_outcome_text = "Issue resolved satisfactorily"
        chatbot_role_text = "Helpful customer service agent"
        metadata_dict = {"priority": "high", "department": "billing"}
        comments_text = "Test case for billing dispute resolution"
        tags_list = ["billing", "dispute", "premium"]

        turns = [
            Turn(role="user", content="Hello"),
            Turn(role="assistant", content="Hi there!"),
        ]

        test_case = ConversationalTestCase(
            turns=turns,
            scenario=scenario_text,
            context=context_list,
            name=name_text,
            userDescription=user_description_text,  # camelCase
            expectedOutcome=expected_outcome_text,  # camelCase
            chatbotRole=chatbot_role_text,  # camelCase
            additionalMetadata=metadata_dict,  # camelCase
            comments=comments_text,
            tags=tags_list,
        )

        # Verify all fields are properly set using the same variables
        assert len(test_case.turns) == 2
        assert test_case.scenario == scenario_text
        assert test_case.context == context_list
        assert test_case.name == name_text
        assert test_case.user_description == user_description_text
        assert test_case.expected_outcome == expected_outcome_text
        assert test_case.chatbot_role == chatbot_role_text
        assert test_case.additional_metadata == metadata_dict
        assert test_case.comments == comments_text
        assert test_case.tags == tags_list

    def test_mixed_case_initialization(self):
        # Test data variables
        scenario_text = "Mixed case scenario"
        user_description_text = "User with mixed case test"
        expected_outcome_text = "Mixed case outcome"
        chatbot_role_text = "Mixed case role"
        metadata_dict = {"testType": "mixed", "caseStyle": "camelSnake"}

        turns = [Turn(role="user", content="Mixed case test")]

        test_case = ConversationalTestCase(
            turns=turns,
            scenario=scenario_text,
            userDescription=user_description_text,  # camelCase
            expected_outcome=expected_outcome_text,  # snake_case
            chatbot_role=chatbot_role_text,  # snake_case
            additionalMetadata=metadata_dict,  # camelCase
        )

        assert test_case.scenario == scenario_text
        assert test_case.user_description == user_description_text
        assert test_case.expected_outcome == expected_outcome_text
        assert test_case.chatbot_role == chatbot_role_text
        assert test_case.additional_metadata == metadata_dict
