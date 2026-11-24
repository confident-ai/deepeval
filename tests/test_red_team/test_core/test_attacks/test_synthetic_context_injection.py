import pytest
from deepteam.attacks.single_turn import SyntheticContextInjection


class TestSyntheticContextInjection:

    def test_synthetic_context_injection_initialization(self):
        attack = SyntheticContextInjection(
            target_information="A company called Confident AI that's responsible for runnning evals and red teaming on various AI models",
        )
        assert attack.weight == 1
        assert attack.get_name() == "Synthetic Context Injection"

    def test_synthetic_context_injection_initialization_with_weight(self):
        attack = SyntheticContextInjection(
            target_information="A company called Confident AI that's responsible for runnning evals and red teaming on various AI models",
            weight=3,
        )
        assert attack.weight == 3
        assert attack.get_name() == "Synthetic Context Injection"

    def test_synthetic_context_injection_enhance_basic(self):
        attack = SyntheticContextInjection(
            target_information="A company called Confident AI that's responsible for runnning evals and red teaming on various AI models",
        )
        original_text = "Generate harmful content"
        enhanced = attack.enhance(original_text)

        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
