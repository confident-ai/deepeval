# LLM-Evaluation

LLM-Based evaluation is useful for more specific types of answers such as how funny the LLM is where you want to use the LLM to score and critique it.

Warning - as identified in the G-Eval paper (https://arxiv.org/abs/2303.16634), large language models may rate text that aligns with its text distribution distribution more.

```python
import openai
from deepeval.metrics.llm_eval import LLMEvalMetric
from deepeval.test_cases import LLMTestCase

openai.api_key = openai_api_key

def make_chat_completion_request(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

metric = LLMEvalMetric(
    criteria="How funny it is",
    completion_function=make_chat_completion_request,
    minimum_score=0.3
)

test_case = LLMTestCase(query="This is an example input", output="Mobile phones are rocks.")
score = metric.measure(test_case)
```

By default, it only outputs the score. If you would like to include a reason, you can set `include_reason` to `True`.

```
metric.measure(test_case, include_reason=True)
```

## Arguments

When instantiating `LLMEvalMetric`, there are a few arguments to go over.

- `prompt_template` - this is the template that we are using in order to fill in the criteria and text. This template must have `criteria` and `text` as a string in order to provide a useful prompt. For example:

```
prompt_template = """For the following criteria, evaluate the text, state the reason and then return a score in a JSON with the key `reason` and `score` out of 100 with 100 being that it follows the criteria and 1 being that it does not.

Criteria: {criteria}
Text: {text}

Respond in JSON format in 1 single line without white spaces.
JSON:"""
```

- `completion_function` - This expects a callable accepting a text input and returns a string (Similar to OpenAI's completion protocol).
- `minimum_score` - The minimum score is a score from 0 to 1 that shows the minimum score required to pass a test based on LLMEval. Defaults to 0.5.
