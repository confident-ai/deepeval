import pytest
from deepeval.benchmarks.ifeval.ifeval import IFEvalInstructionVerifier as V

def test_unknown_instruction_fails_explicitly():
    # Proves the silent fallthrough bug is fixed
    passed, reason = V.verify_instruction_compliance("response", "unknown:id", {})
    assert not passed
    assert "Unknown or unsupported" in reason

def test_keywords_existence():
    # Proves the actual logic works
    passed, _ = V.verify_instruction_compliance("I love quantum computing", "keywords:existence", {"keywords": ["quantum"]})
    assert passed

    failed, _ = V.verify_instruction_compliance("I love classical computing", "keywords:existence", {"keywords": ["quantum"]})
    assert not failed