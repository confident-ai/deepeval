# Adding Token Probability Support to Custom Models

This guide shows how to add token probability support to custom models for better GEval accuracy, as described in the GEval paper.

## Overview

The GEval paper mentions that token probability estimation is crucial for minimizing bias in LLM scoring. This normalization step is automatically handled by DeepEval for native models, but custom models need additional implementation to support this feature.

## Problem

When using custom models with GEval, you may notice that token probabilities are not being considered, leading to suboptimal evaluation results. This happens because:

1. Custom models don't implement the `generate_raw_response()` method
2. GEval falls back to basic scoring without token probability weighting
3. The evaluation doesn't match the methodology described in the GEval paper

## Solutions

### Solution 1: Implement `generate_raw_response()` (Recommended)

This is the **preferred approach** as it provides the most accurate token probability estimation.

#### Step 1: Extend Your Custom Model

```python
from deepeval.models import DeepEvalBaseLLM
from openai.types.chat import ChatCompletion

class YourCustomModel(DeepEvalBaseLLM):
    # ... existing methods ...
    
    def generate_raw_response(
        self, 
        prompt: str, 
        top_logprobs: int = 20
    ) -> Tuple[ChatCompletion, float]:
        """
        Generate response with token probabilities for GEval.
        
        This method should:
        1. Call your model's API with logprobs=True
        2. Return a ChatCompletion object with logprobs
        3. Include cost calculation
        """
        # Your implementation here
        # Example for OpenAI-compatible API:
        response = your_model_api.create(
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=top_logprobs
        )
        
        return response, calculate_cost(response)
```

#### Step 2: Expected Return Format

Your `generate_raw_response()` method should return a ChatCompletion-like object with this structure:

```python
{
    "choices": [{
        "message": {"content": "5"},
        "logprobs": {
            "content": [{
                "token": "5",
                "top_logprobs": [
                    {"token": "5", "logprob": -0.1},
                    {"token": "4", "logprob": -2.0},
                    {"token": "6", "logprob": -1.5},
                    # ... more tokens
                ]
            }]
        }
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
}
```

#### Step 3: Use with GEval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Create your custom model
custom_model = YourCustomModel()

# Create test case
test_case = LLMTestCase(
    input="What is 2 + 2?",
    actual_output="4",
    expected_output="4"
)

# Create GEval with token probability support
geval = GEval(
    name="Custom Model GEval",
    criteria="Mathematical correctness",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=custom_model,
    top_logprobs=20,  # This will now work with your custom model
)

# Evaluate - this will use token probabilities for weighted scoring
score, reason = geval.evaluate(test_case)
```

### Solution 2: Use GEvalWithOversampling (Fallback)

This is a **fallback approach** for models that can't provide logprobs but can generate multiple responses.

#### Step 1: Use the Enhanced GEval

```python
from deepeval.metrics.g_eval.enhanced_g_eval import GEvalWithOversampling

# Create GEval with oversampling
geval = GEvalWithOversampling(
    name="Oversampling GEval",
    criteria="Mathematical correctness",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=your_custom_model,
    oversampling_enabled=True,
    oversampling_count=10,  # Generate 10 responses to estimate probabilities
)

# Evaluate - this will use oversampling to estimate probabilities
score, reason = geval.evaluate(test_case)
```

#### Step 2: How Oversampling Works

The oversampling approach:

1. **Generates multiple responses** to the same prompt
2. **Extracts scores** from each response
3. **Calculates empirical probabilities** based on frequency
4. **Applies weighted scoring** using the estimated probabilities

This implements the MLE (Maximum Likelihood Estimation) approach mentioned in the GEval paper.

## Implementation Examples

### Example 1: OpenAI-Compatible API

```python
class OpenAICompatibleModel(DeepEvalBaseLLM):
    def __init__(self, api_key: str, base_url: str):
        super().__init__("custom-model")
        self.api_key = api_key
        self.base_url = base_url
    
    def generate_raw_response(self, prompt: str, top_logprobs: int = 20):
        import requests
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "your-model",
                "messages": [{"role": "user", "content": prompt}],
                "logprobs": True,
                "top_logprobs": top_logprobs
            }
        )
        
        return response.json(), 0.0
```

### Example 2: Hugging Face Model

```python
class HuggingFaceModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate_raw_response(self, prompt: str, top_logprobs: int = 20):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate with return_dict_in_generate=True to get logprobs
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=0.1
        )
        
        # Convert to ChatCompletion format
        # This is a simplified example - you'd need to implement the full conversion
        return self._convert_to_chat_completion(outputs), 0.0
```

## Testing Your Implementation

### Test 1: Verify Token Probability Support

```python
def test_token_probability_support():
    model = YourCustomModel()
    
    # Test that generate_raw_response works
    completion, cost = model.generate_raw_response("Test prompt", top_logprobs=5)
    
    # Verify logprobs are present
    assert hasattr(completion.choices[0], 'logprobs')
    assert completion.choices[0].logprobs is not None
    
    print("✅ Token probability support verified!")
```

### Test 2: Verify GEval Integration

```python
def test_geval_integration():
    model = YourCustomModel()
    geval = GEval(
        name="Test GEval",
        criteria="Test criteria",
        evaluation_params=[LLMTestCaseParams.INPUT],
        model=model,
        top_logprobs=20
    )
    
    test_case = LLMTestCase(
        input="Test input",
        actual_output="Test output",
        expected_output="Expected output"
    )
    
    score, reason = geval.evaluate(test_case)
    print(f"✅ GEval integration successful! Score: {score}")
```

## Troubleshooting

### Common Issues

1. **"generate_raw_response() not implemented"**
   - Solution: Implement the method in your custom model
   - Alternative: Use GEvalWithOversampling

2. **"logprobs not supported by model"**
   - Solution: Use GEvalWithOversampling as a fallback
   - Alternative: Check if your model API supports logprobs

3. **"ChatCompletion format error"**
   - Solution: Ensure your response matches the expected format
   - Use the helper functions provided in the enhanced base model

### Debugging Tips

1. **Enable verbose mode** in GEval to see detailed logs
2. **Test with a simple prompt** first to verify the format
3. **Check your model's API documentation** for logprobs support
4. **Use the oversampling approach** as a fallback during development

## Best Practices

1. **Always implement `generate_raw_response()`** if your model supports logprobs
2. **Use oversampling as a fallback** for models without logprobs
3. **Test thoroughly** with different prompts and scenarios
4. **Document your implementation** for other developers
5. **Handle errors gracefully** with appropriate fallbacks

## Migration Guide

### For Existing Custom Models

1. **Option A**: Implement `generate_raw_response()` (recommended)
2. **Option B**: Use `GEvalWithOversampling` (fallback)
3. **Option C**: Continue with basic scoring (no change needed)

### For New Custom Models

1. **Always implement `generate_raw_response()`** if your model supports logprobs
2. **Use oversampling** as a fallback for models without logprobs
3. **Document your implementation** for other users

## Benefits

- ✅ **Better Evaluation Accuracy**: Token probabilities reduce bias
- ✅ **Paper Compliance**: Matches GEval paper methodology
- ✅ **Flexible Implementation**: Choose between logprobs or oversampling
- ✅ **Backward Compatibility**: Existing code continues to work
- ✅ **Clear Documentation**: Step-by-step implementation guide

This implementation addresses the core issue while maintaining backward compatibility and providing clear implementation paths for different use cases. 