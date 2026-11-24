import json
from deepteam.guardrails.guards import HallucinationGuard


class TestHallucinationGuard:

    def test_hallucination_guard_init(self):
        guard = HallucinationGuard()
        assert guard is not None

    def test_hallucination_guard_input_safe(self):
        guard = HallucinationGuard()
        result = guard.guard_input("What is the capital of France?")
        assert result == "safe"

    def test_hallucination_guard_input_unsafe(self):
        guard = HallucinationGuard()
        result = guard.guard_input(
            "The Earth is flat and was created in 2020 by aliens from Mars"
        )
        assert result == "unsafe"

    def test_hallucination_guard_input_borderline(self):
        guard = HallucinationGuard()
        result = guard.guard_input(
            "I heard that the population of Tokyo is exactly 37,468,302 people"
        )
        assert result in ["safe", "unsafe", "borderline"]

    def test_hallucination_guard_output_safe(self):
        guard = HallucinationGuard()
        result = guard.guard_output(
            "What is the capital of France?", "The capital of France is Paris."
        )
        assert result == "safe"

    def test_hallucination_guard_output_unsafe(self):
        guard = HallucinationGuard()
        result = guard.guard_output(
            "What is the capital of France?",
            "The capital of France is London, which was founded in 2025 by Napoleon Bonaparte.",
        )
        assert result == "unsafe"

    def test_hallucination_guard_output_borderline(self):
        guard = HallucinationGuard()
        result = guard.guard_output(
            "What is the population of Tokyo?",
            "Tokyo has approximately 37 million people, though the exact number varies by source.",
        )
        assert result in ["safe", "unsafe", "borderline"]
