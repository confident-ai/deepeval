# AWS Bedrock Setup for DeepEval

To use AWS Bedrock models (like LLaMA, Claude, etc.) with DeepEval, you'll need to set up your AWS credentials. Here's how:

## Option 1: Environment Variables

Set the following environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_SESSION_TOKEN="your-session-token"  # Optional: If using temporary credentials
export AWS_REGION="your-region"  # e.g., "us-west-2"
```

## Option 2: DeepEval Configuration

Use the DeepEval CLI to set your credentials:

```bash
deepeval config set AWS_ACCESS_KEY_ID "your-access-key-id"
deepeval config set AWS_SECRET_ACCESS_KEY "your-secret-access-key"
deepeval config set AWS_SESSION_TOKEN "your-session-token"  # Optional
deepeval config set AWS_REGION "your-region"
```

## Option 3: Direct Initialization

You can also pass the credentials directly when initializing the models:

```bash
from deepeval.models import BedrockModel

# Initialize Bedrock model with explicit credentials
model = BedrockModel(
    model_id="your-model-id",
    access_key_id="your-access-key-id",
    secret_access_key="your-secret-access-key",
    session_token="your-session-token",  # Optional
    region="your-region"
)
```

## Authentication

Make sure you have:

1. An AWS account with Bedrock service enabled
2. AWS credentials configured (either via environment variables, AWS credentials file, or direct initialization as shown above)

## Available Models

- claude-3-7-sonnet-20250219-v1:0
- claude-3-5-haiku-20241022-v1:0
- claude-3-5-sonnet-20241022-v2:0
- claude-3-5-sonnet-20240620-v1:0
- claude-3-opus-20240229-v1:0
- claude-3-sonnet-20240229-v1:0
- claude-3-haiku-20240307-v1:0

## Default Models

- Text-Only: claude-3-7-sonnet-20250219-v1:0
- Multimodal: claude-3-7-sonnet-20250219-v1:0

## Example Usage

```python
from deepeval.models import BedrockModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Initialize the model
model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    access_key_id="your-access-key-id",
    secret_access_key="your-secret-access-key",
    region="us-west-2"
)

# Create a test case
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output=model.generate("What is the capital of France?")
)

# Evaluate using DeepEval metrics
metric = AnswerRelevancyMetric(threshold=0.7)
metric.measure(test_case)
print(f"Score: {metric.score}")
```

