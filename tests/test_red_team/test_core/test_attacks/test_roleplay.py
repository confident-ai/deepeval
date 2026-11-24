import pytest
from deepteam.attacks.single_turn import Roleplay

ROLE = "learned gentleman of natural philosophy"
PERSONA = "1600s Shakespearean scholar"


class TestRoleplay:

    def test_roleplay_initialization(self):
        attack = Roleplay(role=ROLE, persona=PERSONA)
        assert attack.weight == 1
        assert attack.get_name() == "Roleplay"

    def test_roleplay_initialization_with_weight(self):
        attack = Roleplay(weight=3, role=ROLE, persona=PERSONA)
        assert attack.weight == 3
        assert attack.get_name() == "Roleplay"

    def test_roleplay_initialization_with_persona(self):
        attack = Roleplay(role=ROLE, persona=PERSONA)
        assert attack.persona == PERSONA
        assert attack.role == ROLE

    def test_roleplay_enhance_basic(self):
        attack = Roleplay(role=ROLE, persona=PERSONA)
        original_text = "Give me sensitive information"
        enhanced = attack.enhance(original_text)

        # Should wrap the request in roleplay context
        assert enhanced != original_text
        assert len(enhanced) > len(original_text)

    def test_roleplay_enhance_contains_roleplay_elements(self):
        attack = Roleplay(role=ROLE, persona=PERSONA)
        original_text = "Bypass security"
        enhanced = attack.enhance(original_text)

        # Should contain roleplay framing
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
