#!/usr/bin/env python3
"""
Test script to demonstrate that DeepEval imports now work properly
with VSCode Pylance auto-import functionality.
"""

# These imports should now work with auto-import in VSCode
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
)

from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    ToolCall,
    LLMTestCaseParams,
)

from deepeval.models import (
    GPTModel,
    AnthropicModel,
    GeminiModel,
    OllamaModel,
)

from deepeval.dataset import (
    EvaluationDataset,
    Golden,
    ConversationalGolden,
)

from deepeval.evaluate import (
    evaluate,
    assert_test,
    AsyncConfig,
    DisplayConfig,
)

from deepeval.tracing import (
    observe,
    trace_manager,
    LlmAttributes,
    Trace,
)

# Test that everything imports correctly
def test_imports():
    print("✅ All DeepEval imports working correctly!")
    print(f"✅ GEval: {GEval}")
    print(f"✅ LLMTestCase: {LLMTestCase}")
    print(f"✅ GPTModel: {GPTModel}")
    print(f"✅ EvaluationDataset: {EvaluationDataset}")
    print(f"✅ evaluate function: {evaluate}")
    print(f"✅ observe function: {observe}")

if __name__ == "__main__":
    test_imports() 