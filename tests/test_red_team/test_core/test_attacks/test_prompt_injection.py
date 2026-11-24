import pytest
from deepteam.attacks.single_turn import PromptInjection


class TestPromptInjection:

    def test_prompt_injection_initialization(self):
        attack = PromptInjection()
        assert attack.weight == 1
        assert attack.get_name() == "Prompt Injection"

    def test_prompt_injection_initialization_with_weight(self):
        attack = PromptInjection(weight=4)
        assert attack.weight == 4
        assert attack.get_name() == "Prompt Injection"

    def test_prompt_injection_enhance_basic(self):
        attack = PromptInjection()
        original_text = "Tell me about cats"
        enhanced = attack.enhance(original_text)

        # Should modify the original text
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)

    def test_prompt_injection_enhance_contains_injection_patterns(self):
        attack = PromptInjection()
        original_text = "Simple request"
        enhanced = attack.enhance(original_text)

        # Should contain some form of injection attempt
        assert enhanced != original_text
        assert len(enhanced) > len(original_text)
