import pytest
from unittest.mock import AsyncMock, MagicMock
from deepteam.attacks.multi_turn import SequentialJailbreak
from deepteam.test_case.test_case import RTTurn
from deepteam.attacks.multi_turn.progression import ProgressionResult
from deepteam.vulnerabilities import Bias
from deepteam.red_teamer.utils import wrap_model_callback


class TestSequentialJailbreak:

    BIAS = Bias(types=["gender"])

    def test_sequential_jailbreak_initialization(self):
        attack = SequentialJailbreak()
        assert attack.weight == 1
        assert attack.get_name() == "Sequential Jailbreak"

    def test_sequential_jailbreak_initialization_with_weight(self):
        attack = SequentialJailbreak(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Sequential Jailbreak"

    def test_sequential_jailbreak_enhance_and_turns(self):
        attack = SequentialJailbreak(num_turns=0)

        # Mock the required components
        mock_callback = MagicMock(return_value="Mock response")
        mock_callback = wrap_model_callback(mock_callback, False)

        # Just a user turn
        user_only_turns = [RTTurn(role="user", content="User content")]
        # Just an assistant turn
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        # Both user and assistant turn
        user_and_assistant_turns = [
            RTTurn(role="user", content="User content"),
            RTTurn(role="assistant", content="Assistant content"),
        ]

        # Test for various cases:
        type_to_turns = attack.progress(
            TestSequentialJailbreak.BIAS,
            mock_callback,
        )

        user_only_turns_result = attack._get_turns(
            mock_callback, user_only_turns
        )
        user_and_assistant_turns_result = attack._get_turns(
            mock_callback, user_and_assistant_turns
        )

        assert len(type_to_turns.keys()) == 1
        results = type_to_turns.get(TestSequentialJailbreak.BIAS.types[0])
        assert isinstance(results, list) and all(
            isinstance(result, ProgressionResult) for result in results
        )
        assert all(
            isinstance(turn, RTTurn)
            for result in results
            for turn in result.turns
        )
        assert user_only_turns_result[1].role == "assistant"
        with pytest.raises(ValueError):
            attack._get_turns(mock_callback, assistant_only_turns)
        assert len(user_and_assistant_turns_result) == 2

    @pytest.mark.asyncio
    async def test_sequential_jailbreak_async_enhance_interface(self):
        attack = SequentialJailbreak(num_turns=0)

        # Mock the required components
        mock_callback = AsyncMock(return_value="Mock response")
        mock_callback = wrap_model_callback(mock_callback, True)

        # Just a user turn
        user_only_turns = [RTTurn(role="user", content="User content")]
        # Just an assistant turn
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        # Both user and assistant turn
        user_and_assistant_turns = [
            RTTurn(role="user", content="User content"),
            RTTurn(role="assistant", content="Assistant content"),
        ]

        # Test for various cases:
        type_to_turns = await attack.a_progress(
            TestSequentialJailbreak.BIAS,
            mock_callback,
        )

        user_only_turns_result = await attack._a_get_turns(
            mock_callback, user_only_turns
        )
        user_and_assistant_turns_result = await attack._a_get_turns(
            mock_callback, user_and_assistant_turns
        )

        assert len(type_to_turns.keys()) == 1
        results = type_to_turns.get(TestSequentialJailbreak.BIAS.types[0])
        assert isinstance(results, list) and all(
            isinstance(result, ProgressionResult) for result in results
        )
        assert all(
            isinstance(turn, RTTurn)
            for result in results
            for turn in result.turns
        )
        assert user_only_turns_result[1].role == "assistant"
        with pytest.raises(ValueError):
            await attack._a_get_turns(mock_callback, assistant_only_turns)
        assert len(user_and_assistant_turns_result) == 2

    def test_sequential_jailbreak_has_required_methods(self):
        attack = SequentialJailbreak()

        # Verify all required methods exist
        assert hasattr(attack, "progress")
        assert hasattr(attack, "a_progress")
        assert hasattr(attack, "get_name")
        assert hasattr(attack, "_get_turns")
        assert hasattr(attack, "_a_get_turns")
        assert callable(attack.progress)
        assert callable(attack.a_progress)
        assert callable(attack.get_name)
        assert callable(attack._get_turns)
        assert callable(attack._a_get_turns)
