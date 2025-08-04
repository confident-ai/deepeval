#!/usr/bin/env python3
"""
Simple Import Test - Test that all imports work without requiring API keys
"""

# Test 1: Basic imports (should work without warnings)
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
)

# Test 2: Test case imports
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    ToolCall,
    LLMTestCaseParams,
)

# Test 3: Model imports
from deepeval.models import (
    GPTModel,
    AnthropicModel,
    GeminiModel,
    OllamaModel,
)

# Test 4: Dataset imports
from deepeval.dataset import (
    EvaluationDataset,
    Golden,
    ConversationalGolden,
)

# Test 5: Evaluation imports
from deepeval.evaluate import (
    evaluate,
    assert_test,
    AsyncConfig,
    DisplayConfig,
)

# Test 6: Tracing imports
from deepeval.tracing import (
    observe,
    trace_manager,
    LlmAttributes,
    Trace,
)

# Test 7: Submodule imports
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval.metrics.faithfulness import FaithfulnessTemplate

def test_imports():
    """Test that all imports work correctly"""
    print("✅ All imports successful!")
    
    # Test test case creation (doesn't require API keys)
    test_case = LLMTestCase(
        input="What is 2+2?",
        actual_output="4",
        expected_output="4"
    )
    print(f"✅ LLMTestCase created: {test_case}")
    
    # Test dataset creation (doesn't require API keys)
    dataset = EvaluationDataset(alias="Test Dataset", test_cases=[])
    print(f"✅ EvaluationDataset created: {dataset}")
    
    # Test metric classes (just check they exist, don't instantiate)
    print(f"✅ GEval class: {GEval}")
    print(f"✅ AnswerRelevancyMetric class: {AnswerRelevancyMetric}")
    print(f"✅ FaithfulnessMetric class: {FaithfulnessMetric}")
    
    # Test model classes (just check they exist)
    print(f"✅ GPTModel class: {GPTModel}")
    print(f"✅ AnthropicModel class: {AnthropicModel}")
    
    # Test function imports
    print(f"✅ evaluate function: {evaluate}")
    print(f"✅ assert_test function: {assert_test}")
    print(f"✅ observe function: {observe}")

if __name__ == "__main__":
    test_imports() 