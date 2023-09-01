# Create Synthetic Data

## Problem synthetic data creation solves

- When there isn't much data or any data to start with for evaluating langchain pipelines
- When getting an eyeball check of current performance is done very quickly

Generating synthetic queries allows you to quickly evaluate the queries related to your prompts.

We help developers get up and running with example queries from just raw text based on OpenAI's model. In this model, we generate query-answer pairs based on teh text.

```python
# Loads the synthetic query model to generate them based on data you get.
# These automatically create synthetic queries and adds them to our online database
from deepeval.dataset import create_evaluation_query_output_pairs
dataset = create_evaluation_query_output_pairs("Python is a great language for mathematical expression and machine learning.")
```

## Running test cases.

Once you have defined a number of test cases, you can easily run it in bulk if required.

```python
# test_bulk_runner.py

def generate_llm_output(query: str) -> str:
    return "sample output"

# Run an evaluation as you would any normal evaluation.
dataset.run_evaluation(completion_fn=generate_llm_output)
```
