# AnswerLength 

Answer Length is simply a character count of model output and intended to balance the case if a model meeting additional eval criteria by simply adding a longer answer, if this becomes the case. 
- Answer length removes any preceding and trailing spaces, if exists
- Originally intended for use in natural language models to measure brevity, however it may also be relevent for additional modalities if the user feels it relevant.

## Usage

```python
from deepeval.test_utils import assert_llm_output
from deepeval.metrics.answer_length import LengthMetric

def generate_llm_output(text_input: str):
    return text_input


def test_llm_output(self):
    input = " some output "
    expected_output = len(input.strip())
    output = generate_llm_output(input)
    metric = LengthMetric()
    assert_llm_output(output, expected_output, metric=metric)
```