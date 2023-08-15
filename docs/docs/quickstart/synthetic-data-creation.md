# Synthetic Data Creation

## Problem synthetic data creation solves

- When there isn't much data or any data to start with for evaluating langchain pipelines
- When getting an eyeball check of current performance is done very quickly

Generating synthetic queries allows you to quickly evaluate the queries related to your prompts.
We help developers get up and running with example queries from just raw text.

```python
# Loads the synthetic query model to generate them based on data you get.
# These automatically create synthetic queries and adds them to our online database
from deepeval.query_generator import BEIRQueryGenerator
# NOTE: loading this may take a while as the model used is quite big
gen = BEIRQueryGenerator()
text = "Synthetic queries are useful for scenraios where there is no data."
queries = gen.generate_queries(
    texts=[text],
    num_queries=1
)
print(queries)
```

## How to use synthetic queries with DeepEval Framework

Once you have created synthetic queries, we recommend saving them in a CSV (future editions will automatically upload these queries into the cloud once they are generated so that they aren't lost).

```python
from deepeval.test_case import TestCase
test_cases = []
for q in queries:
    test_cases.append(
        TestCase(
            input=q,
            expected_output=text
        )
    )
```

This allows you to get started immediately with test cases.
