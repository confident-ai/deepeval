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

Here is an example of how you can define test cases.

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

## Running test cases in bulk

Once you have defined a number of test cases, you can easily run it in bulk if required.

```python
# test_bulk_runner.py

def generate_llm_output(input: str) -> str:
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": input}]
    )
    return chat_completion.choices[0].message.content

def test_bulk_runner():
    class BulkTester(BulkTestRunner):
        @property
        def bulk_test_cases(self):
            return test_cases

    tester = BulkTester()
    tester.run(callable_fn=generate_llm_output)

```

Once you have written these tests, you can then simply call `pytest` via CLI again to trigger these tests.

```bash
python -m pytest test_bulk_runner.py

# Output
Running tests ... ✅✅✅
```


