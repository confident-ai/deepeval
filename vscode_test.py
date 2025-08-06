#!/usr/bin/env python3
"""
VSCode Test File - Test the import fixes in VSCode
Open this file in VSCode and check for:
1. No Pylance warnings
2. Auto-import suggestions work
3. IntelliSense works properly
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

# Test 7: Submodule imports (should also work)
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval.metrics.faithfulness import FaithfulnessTemplate

def test_imports():
    """Test that all imports work correctly"""
    print("✅ All imports successful!")
    
    # Test metric instantiation
    geval = GEval(
        name="Test GEval",
        criteria="Test criteria",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    print(f"✅ GEval created: {geval}")
    
    # Test test case creation
    test_case = LLMTestCase(
        input="What is 2+2?",
        actual_output="4",
        expected_output="4"
    )
    print(f"✅ LLMTestCase created: {test_case}")
    
    # Test model creation
    model = GPTModel(model_name="gpt-4")
    print(f"✅ GPTModel created: {model}")
    
    # Test dataset creation
    dataset = EvaluationDataset(alias="Test Dataset", test_cases=[])
    print(f"✅ EvaluationDataset created: {dataset}")

if __name__ == "__main__":
    test_imports() 