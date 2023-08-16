# Create An Evaluation Dataset

## Defining A Dataset

An evaluation dataset is a list of test cases designed to make testing a number of evaluations very easy.

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

#### Running The Evaluation

```python
ds.run_evaluation(
    callable_fn=generate_llm_output,
)
# Returns the evaluation
```

Once you run these tests, you will then be given a table that looks like this and is saved to a text file.

```
Test Passed  Metric Name                  Score    Output                                            Expected output    Message
-------------  ---------------------  -----------  ------------------------------------------------  -----------------  -------------------------------------------
            1  EntailmentScoreMetric  0.000830871  Our customer success phone line is 1200-231-231.  1800-213-123       EntailmentScoreMetric was unsuccessful for
                                                                                                                        What is the customer success number
                                                                                                                        which should have matched
                                                                                                                        1800-213-123
```
