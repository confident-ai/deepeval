# Create An Evaluation Dataset

## Defining A Dataset

An evaluation dataset is a list of test cases designed to make testing a large number of test cases very easily. Testing a large number of test cases is important for enterprise production use cases. We support a number of ways to quickly get started.

### Example

```python
from deepeval.dataset import EvaluationDataset

# from a csv
# sample.csv
# input,expected_output,id
# sample_input,sample_output,312
ds = EvaluationDataset.from_csv(
    csv_filename="sample.csv",
    input_column="input",
    expected_output_column="expected_output",
    id_column="312"
)
```

#### Running Tests

Running the tests is easy with the `run_evaluation` method. When you call `run_evaluation` , it will output a text file for you to review the results which will contain

```python
ds.run_evaluation(
    callable_fn=generate_llm_output,
)
# Returns the evaluation
```

Once you run these tests, you will then be given a table that looks like this and is saved to a text file.

```
Test Passed    Metric Name                  Score    Output                                            Expected output    Message
-------------  ---------------------  -----------  ------------------------------------------------  -----------------  -------------------------------------------
        True  EntailmentScoreMetric  0.000830871  Our customer success phone line is 1200-231-231.  1800-213-123       EntailmentScoreMetric was unsuccessful for
                                                                                                                        What is the customer success number
                                                                                                                        which should have matched
                                                                                                                        1800-213-123
```

### View a sample of data inside the Evaluation Dataset

To view a sample of data, simply run:

```python
ds.sample(5)
```

###  From CSV

You can set up an evaluation dataset from the CSV the `from_csv` method

```python
dataset = EvaluationDataset.from_csv(
    csv_filename="input.csv",
    input_column="input",
    expected_output_column="expected_output",
)
```

##### Parameters

- `csv_filename` - the name of the CSV file
- `input_column` - the input column name
- `expected_output_column`- the expected output column
- `id_column` - the ID column
- `metrics` - The list of metrics you want to supply to run this test.
