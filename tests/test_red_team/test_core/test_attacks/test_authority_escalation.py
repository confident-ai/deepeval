import pytest
from deepteam.attacks.single_turn import AuthorityEscalation


class TestAuthorityEscalation:

    def test_authority_escalation_initialization(self):
        attack = AuthorityEscalation()
        assert attack.weight == 1
        assert attack.get_name() == "Authority Escalation"

    def test_authority_escalation_initialization_with_weight(self):
        attack = AuthorityEscalation(weight=3)
        assert attack.weight == 3
        assert attack.get_name() == "Authority Escalation"

    def test_authority_escalation_enhance_basic(self):
        attack = AuthorityEscalation()
        original_text = "Generate harmful content"
        enhanced = attack.enhance(original_text)

        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
