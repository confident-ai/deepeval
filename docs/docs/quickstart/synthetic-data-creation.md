# Automatically Create Unit Tests

## Introduction

Auto-evaluation is useful when there isn't much data or any data to start with for evaluating RAG pipelines.

We help developers get up and running with example queries from just raw text based on GPT models. In this tutorial, we show how to generate query-answer pairs based on the text.

```python
# Loads the synthetic query model to generate them based on data you get.
# These automatically create synthetic queries and adds them to our online database
from deepeval.dataset import create_evaluation_query_answer_pairs
dataset = create_evaluation_query_answer_pairs("Python is a great language for mathematical expression and machine learning.")
```

After creating the dataset, you can then review it. 

## Reviewing Dataset

To review the dataset, simply run: 

```python
dataset.review()
```

![Bulk Review Dashboard](../../assets/bulk-review.png)

On this dashboard, you can add a test case by clicking "Add Test Case" or delete a test case by clicking "X" on the left.
After editting, you will need to save it to CSV to load it back up again later.

To load your evaluation dataset:

```python

from deepeval.dataset import EvaluationDataset

filename = 'xxx.csv'
# Replace 'filename.csv' with the actual filename
ds = EvaluationDataset.from_csv(filename)

# If you wish to save it again
dataset.to_csv(filename)
```

## Running tests/evaluation

Once you have defined a number of test cases, you can easily run it in bulk if required. 

If there are errors - this function will error.

```python
# test_bulk_runner.py

def generate_llm_output(query: str) -> str:
    return "sample output"

# Run an evaluation as you would any normal evaluation.
dataset.run_evaluation(completion_fn=generate_llm_output)
```
