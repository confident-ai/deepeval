# Google Cloud Setup for DeepEval

To use Google Cloud's Vertex AI models (like Gemini) with DeepEval, you'll need to set up your Google Cloud credentials. Here's how:

## Option 1: Environment Variables

Set the following environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="your-region"  # e.g., "us-central1"
```

## Option 2: DeepEval Configuration

Use the DeepEval CLI to set your credentials:

```bash
deepeval config set GOOGLE_CLOUD_PROJECT "your-project-id"
deepeval config set GOOGLE_CLOUD_LOCATION "your-region"
```

## Option 3: Direct Initialization

You can also pass the credentials directly when initializing the models:

```python
from deepeval.models import GeminiModel, MultimodalGeminiModel

# For text-only evaluation
model = GeminiModel(
    model_name="gemini-1.5-pro",
    project_id="your-project-id",
    location="us-central1"
)

# For multimodal evaluation
model = MultimodalGeminiModel(
    model_name="gemini-1.5-pro",
    project_id="your-project-id",
    location="us-central1"
)
```

## Authentication

Make sure you have:

1. A Google Cloud project with the Vertex AI API enabled
2. Application Default Credentials set up:
   ```bash
   gcloud auth application-default login
   ```

## Available Models

### Text-Only Models
- gemini-1.5-flash
- gemini-1.5-flash-001
- gemini-1.5-flash-002
- gemini-1.5-pro
- gemini-1.5-pro-001
- gemini-1.5-pro-002
- gemini-1.0-pro
- gemini-1.0-pro-001
- gemini-1.0-pro-002

### Multimodal Models
- gemini-1.5-flash
- gemini-1.5-flash-001
- gemini-1.5-flash-002
- gemini-1.5-pro
- gemini-1.5-pro-001
- gemini-1.5-pro-002
- gemini-1.0-pro-vision
- gemini-1.0-pro-vision-001

## Default Models
- Text-Only: gemini-1.5-pro
- Multimodal: gemini-1.5-pro

## Example Usage

```python
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Initialize the model
model = GeminiModel(
    model_name="gemini-1.5-pro",
    project_id="your-project-id",
    location="us-central1"
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
