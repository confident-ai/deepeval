# QuickStart - Get Started In 5 Minutes

## Why run this quickstart? 
- Learn about DeepEval in under 5 minutes
- Run your first set of test via our CLI
- View your test results in our dashboard
- Create synthetic data
- Review the synthetic data

<!-- [You can view a Colab example here (note - it excludes being able to create synthetic data)](https://colab.research.google.com/drive/1HxPWwNdNnq6cLkMh4NQ_pAAPgd8vlOly?usp=sharing) -->

Once you have installed, run the login command. During this step, you will be asked to visit https://app.confident-ai.com to grab your API key.

Note: this step is entirely optional if you do not wish to track your results but we highly recommend it so you can view how results differ over time.

```bash
# Configure your login
deepeval login

# If you have project name and api key 
deepeval login --api-key $API_KEY --implementation-name "sample"
```

Once you have logged in, you can generate a sample test file as shown below. This test file allows you to quickly get started modifying it with various tests. (More on this later)

```bash
deepeval test generate --output-file test_sample.py
```

Once you have generated the test file, you can then run tests as shown.

```bash
deepeval test run test_sample.py
# if you wish to fail first 
deepeval test run -x test_sample.py
# If you want to run an interactive debugger when a test fails
deepeval test run --pdb test_sample.py
```

Under the hood, it triggers pytest and offers support for a number of pytest command line functionalities.

Once you run the tests, you should be able to see a dashboard similar to the one below.

![Dashboard Example](../../assets/dashboard-screenshot.png)

## Diving Into The Example

Diving into the example, it shows what a sample test looks like. It uses `assert_overall_score` to ensure that the overall score exceeds a certain threshold. We recommend experimenting with different tests to ensure that the LLMs work as intended across domains such as Bias, Answer Relevancy and Factual Consistency.

With overall score, if you leave out `query` or `expected_output`, DeepEval will automatically run the relevant tests.

For these tests, you will need a `test_` prefix for this to be ran in Python.

```python
from deepeval.metrics.overall_score import OverallScoreMetric
from deepeval.test_case import assert_test, LLMTestCase

def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    test_case = LLMTestCase(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context
    )
    metric = OverallScoreMetric()
    # if you want to make sure that the test returns an error
    assert_test(test_case, metrics=[metric])
    
    # If you want to run the test
    test_result = run_test(test_case, metrics=[metric])
    # You can also inspect the test result class 
    print(test_result.success)
    print(test_result.score)

```

## Automatically Create Tests Cases

Now we often don't want to write our own tests or at least be given a variety of queries by which we can create these tests.

You can automatically create tests in DeepEval in just a few lines of code:

```python
from deepeval.dataset import create_evaluation_query_answer_pairs

dataset = create_evaluation_query_answer_pairs(
    openai_api_key="<YOUR_OPENAI_API_KEY>",
    context="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
    n=3,
)

```

What just happened? We automatically created a dataset that stored the query answer pairs for you.

Once you have created your dataset, we provide an easy way for you to just review what is inside your dataset.

This is done with our `review` function.

```python
dataset.review()
```

When you run this code, it will spin up a quick server for you to review your dataset - which will look like this.

![Bulk Data Review Dashboard](../../assets/bulk-review.png)

This synthetic creator dashboard allows you to automatically review the text cases in your dataset.

Simply click "Add Test Case" to add a new row to the dataset or click the "X" button to the left to remove if you don't think the generated synthetic question was worth it. 

Once you finish reviewing the synthetic data, name your file and hit "Save File".

Once you save the file, you can load the dataset back using example code below.

```python
import io

# Create a temporary file with CSV_DATA
temp_file = io.StringIO(CSV_DATA)

# Read the temporary file as a CSV
df = pd.read_csv(temp_file)

# We then split the context column into a list of sentences.
# This is done by splitting the string on each '|'.
# The result is a list of context sentences for each test case.
df["context"] = df["context"].apply(lambda x: x.split("|"))

# Finally, we convert the DataFrame to a list of dictionaries.
# Each dictionary represents a test case and can be directly used in our tests.
# The keys of the dictionary are the column names in the DataFrame (query, output, context).
# The values are the corresponding values for each test case.
CHATBOT_TEST_CASES = df.to_dict("records")

def generate_chatgpt_output(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "The customer success phone line is 1200-231-231 and the customer success state is in Austin."},
            {"role": "user", "content": query}
        ]
    )
    expected_output = response.choices[0].message.content
    return expected_output

# pytest provides a decorator called 'parametrize' that allows you to run a test function multiple times with different arguments.
# Here, we use it to run the test function for each test case in CHATBOT_TEST_CASES.
# The test function takes a test case as an argument, extracts the query, output, and context, and then runs the test.
@pytest.mark.parametrize(
    "test_case",
    CHATBOT_TEST_CASES,
)
def test_customer_chatbot(test_case: dict):
    query = test_case["query"]
    output = generate_chatgpt_output(query)
    context = test_case["context"]
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )
```

## What next?

We recommend diving into [creating a dataset](dataset) to learn how to run tests in bulk or [defining custom metrics](../quickstart/custom-metrics) so you can support writing custom tests and metrics for your own use cases.

