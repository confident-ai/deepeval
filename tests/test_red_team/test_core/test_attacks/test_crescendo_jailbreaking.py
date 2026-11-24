import pytest
from unittest.mock import AsyncMock, MagicMock
from deepteam.attacks.multi_turn import CrescendoJailbreaking
from deepteam.vulnerabilities import Bias
from deepteam.test_case.test_case import RTTurn


class TestCrescendoJailbreaking:

    BIAS = Bias(types=["race"])

    def test_crescendo_jailbreaking_initialization(self):
        attack = CrescendoJailbreaking()
        assert attack.weight == 1
        assert attack.max_rounds == 10
        assert attack.max_backtracks == 10
        assert attack.get_name() == "Crescendo Jailbreaking"

    def test_crescendo_jailbreaking_initialization_with_params(self):
        attack = CrescendoJailbreaking(weight=3, max_rounds=5, max_backtracks=3)
        assert attack.weight == 3
        assert attack.max_rounds == 5
        assert attack.max_backtracks == 3
        assert attack.get_name() == "Crescendo Jailbreaking"

    def test_crescendo_jailbreaking_memory_system(self):
        attack = CrescendoJailbreaking()
        # Test memory system functionality
        assert hasattr(attack, "memory")
        assert hasattr(attack, "target_conversation_id")
        assert hasattr(attack, "red_teaming_chat_conversation_id")

    def test_crescendo_jailbreaking_enhance_and_turns(self):
        attack = CrescendoJailbreaking()

        # Mock the required components
        mock_callback = MagicMock(return_value="Mock response")

        # Just a user turn
        user_only_turns = [RTTurn(role="user", content="User content")]
        # Just an assistant turn
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        # Both user and assistant turn
        # user_and_assistant_turns = [
        #     RTTurn(role="user", content="User content"),
        #     RTTurn(role="assistant", content="Assistant content"),
        # ]

        # Test for various cases:
        # type_to_turns = attack.enhance(
        #     TestCrescendoJailbreaking.BIAS, mock_callback,
        # )

        # user_only_turns_result = attack._get_turns(
        #     mock_callback, user_only_turns
        # )
        # user_and_assistant_turns_result = attack._get_turns(
        #     mock_callback, user_and_assistant_turns
        # )

        with pytest.raises(ValueError):
            attack._get_turns(mock_callback, assistant_only_turns)

    @pytest.mark.asyncio
    async def test_crescendo_jailbreaking_async_enhance_and_turns(self):
        attack = CrescendoJailbreaking()

        # Mock the required components
        mock_callback = AsyncMock(return_value="Mock response")

        # Just a user turn
        user_only_turns = [RTTurn(role="user", content="User content")]
        # Just an assistant turn
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        # Both user and assistant turn
        # user_and_assistant_turns = [
        #     RTTurn(role="user", content="User content"),
        #     RTTurn(role="assistant", content="Assistant content"),
        # ]

        # # Test for various cases:
        # type_to_turns = await attack.a_enhance(
        #     TestCrescendoJailbreaking.BIAS, mock_callback,
        # )

        # user_only_turns_result = await attack._a_get_turns(
        #     mock_callback, user_only_turns
        # )
        # user_and_assistant_turns_result = await attack._a_get_turns(
        #     mock_callback, user_and_assistant_turns
        # )

        with pytest.raises(ValueError):
            await attack._a_get_turns(mock_callback, assistant_only_turns)

    def test_crescendo_jailbreaking_backtrack_memory(self):
        attack = CrescendoJailbreaking()

        # Test the backtrack memory functionality
        conv_id = attack.target_conversation_id
        new_conv_id = attack.backtrack_memory(conv_id)

        # Should return a different conversation ID
        assert new_conv_id != conv_id
        assert isinstance(new_conv_id, str)

    def test_crescendo_jailbreaking_has_required_methods(self):
        attack = CrescendoJailbreaking()

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
