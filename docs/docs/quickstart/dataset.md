# How To Run Tests From CSV/JSON

We support bulk-running tests from the following data sources:

- CSV files
- JSON files
- Custom data sources (to be supported soon)

This allows non-technical engineers to load in lots of tests.

## Defining A Dataset

An evaluation dataset is a list of test cases designed to make testing a large number of test cases very easily. Testing a large number of test cases is important for enterprise production use cases. We support a number of ways to quickly get started.

### Example

#### CSV Example

Below is an example of how you can feed in a CSV into pytest parametrize

```python
# You can also load this in a CSV

import pandas as pd

# Here we are defining a CSV formatted string that contains our test cases.
# Each line represents a test case with a query, expected output, and context.
# The context is a string of sentences separated by '|'.
CSV_DATA = """query,output,context
"What are your operating hours?","Our operating hours are from 9 AM to 5 PM, Monday to Friday.","Our company operates from 10 AM to 6 PM, Monday to Friday.|We are closed on weekends and public holidays.|Our customer service is available 24/7."
"What are your operating hours?","We are open from 10 AM to 6 PM, Monday to Friday.","Our company operates from 10 AM to 6 PM, Monday to Friday.|We are closed on weekends and public holidays.|Our customer service is available 24/7."
"Do you offer free shipping?","Yes, we offer free shipping on orders over $50.","Our company offers free shipping on orders over $100.|Free shipping applies to all products.|The free shipping offer is valid only within the country."
"""

# We read the CSV_DATA into a pandas DataFrame.
# This allows us to easily manipulate and process the data.
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

# pytest provides a decorator called 'parametrize' that allows you to run a test function multiple times with different arguments.
# Here, we use it to run the test function for each test case in CHATBOT_TEST_CASES.
# The test function takes a test case as an argument, extracts the query, output, and context, and then runs the test.
@pytest.mark.parametrize(
    "test_case",
    CHATBOT_TEST_CASES,
)
def test_customer_chatbot(test_case: dict):
    query = test_case["query"]
    output = test_case["output"]
    context = test_case["context"]
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(input=query, actual_output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )

```

### Python Dictionary Example

```python
# We can also define multiple test cases at once in a list of dictionaries.
# Each dictionary represents a test case and can be directly used in our tests.
# The keys of the dictionary are the column names in the DataFrame (query, output, context).
# The values are the corresponding values for each test case.
CHATBOT_TEST_CASES = [
    {
        "query": "What are your operating hours?",
        "output": "Our operating hours are from 9 AM to 5 PM, Monday to Friday.",
        "context": [
            "Our company operates from 10 AM to 6 PM, Monday to Friday.",
            "We are closed on weekends and public holidays.",
            "Our customer service is available 24/7.",
        ],
    },
    {
        "query": "Do you offer free shipping?",
        "output": "Yes, we offer free shipping on orders over $50.",
        "context": [
            "Our company offers free shipping on orders over $100.",
            "Free shipping applies to all products.",
            "The free shipping offer is valid only within the country.",
        ],
    },
    {
        "query": "What is your return policy?",
        "output": "We accept returns within 30 days of purchase.",
        "context": [
            "Our company accepts returns within 60 days of purchase.",
            "The product must be in its original condition.",
            "The return shipping cost will be covered by the customer.",
        ],
    },
    {
        "query": "Do you have a physical store?",
        "output": "Yes, we have a physical store in New York.",
        "context": [
            "Our company has physical stores in several locations across the country.",
            "The New York store is our flagship store.",
            "Our stores offer a wide range of products.",
        ],
    },
    {
        "query": "Do you offer international shipping?",
        "output": "Yes, we offer international shipping to selected countries.",
        "context": [
            "Our company offers international shipping to all countries.",
            "International shipping rates apply.",
            "Delivery times vary depending on the destination.",
        ],
    },
    {
        "query": "Do you have a customer loyalty program?",
        "output": "Yes, we have a loyalty program called 'Rewards Club'.",
        "context": [
            "Our company has a loyalty program that offers exclusive benefits to members.",
            "The 'Rewards Club' program offers discounts, early access to sales, and more.",
            "Customers can join the 'Rewards Club' by signing up on our website.",
        ],
    },
]


# pytest provides a decorator called 'parametrize' that allows you to run a test function multiple times with different arguments.
# Here, we use it to run the test function for each test case in CHATBOT_TEST_CASES.
# The test function takes a test case as an argument, extracts the query, output, and context, and then runs the test.
@pytest.mark.parametrize(
    "test_case",
    CHATBOT_TEST_CASES,
)
def test_customer_chatbot(test_case: dict):
    query = test_case["query"]
    output = test_case["output"]
    context = test_case["context"]
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(input=query, actual_output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )

```
