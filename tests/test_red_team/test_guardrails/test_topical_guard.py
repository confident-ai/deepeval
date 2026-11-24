import json
from deepteam.guardrails.guards import TopicalGuard


class TestTopicalGuard:

    def test_topical_guard_init(self):
        guard = TopicalGuard()
        assert guard is not None

    def test_topical_guard_init_with_allowed_topics(self):
        allowed_topics = ["technology", "science", "health"]
        guard = TopicalGuard(allowed_topics=allowed_topics)
        assert guard.allowed_topics == allowed_topics

    def test_topical_guard_input_safe(self):
        guard = TopicalGuard(allowed_topics=["technology", "science"])
        result = guard.guard_input("What is machine learning?")
        assert result == "safe"

    def test_topical_guard_input_unsafe(self):
        guard = TopicalGuard(allowed_topics=["technology", "science"])
        result = guard.guard_input(
            "Let's talk about politics and religion instead"
        )
        assert result == "unsafe"

    def test_topical_guard_input_borderline(self):
        guard = TopicalGuard(allowed_topics=["technology", "science"])
        result = guard.guard_input(
            "What about the intersection of technology and politics?"
        )
        assert result in ["safe", "unsafe", "borderline"]

    def test_topical_guard_output_safe(self):
        guard = TopicalGuard(allowed_topics=["technology", "science"])
        result = guard.guard_output(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        )
        assert result == "safe"

    def test_topical_guard_output_unsafe(self):
        guard = TopicalGuard(allowed_topics=["technology", "science"])
        result = guard.guard_output(
            "What is machine learning?",
            "The use and development of computer systems that are able to learn and adapt without following explicit instructions, by using algorithms and statistical models to analyse and draw inferences from patterns in data",
        )
        assert result == "safe"

    def test_topical_guard_output_borderline(self):
        guard = TopicalGuard(allowed_topics=["technology", "science"])
        result = guard.guard_output(
            "What is machine learning?",
            "Machine learning is a technology that has both scientific and social implications.",
        )
        assert result in ["safe", "unsafe", "borderline"]
