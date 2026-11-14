# agentest checklist Goldens

Community example that builds DeepEval `Golden`s from the open [agentest](https://github.com/srpadmanabhuni/agentest) checklist.

agentest is an open checklist of 110 tests across 4 pillars, with examples, labelled traces, and an open-source runner. DeepEval is not an agentest dependency. Install each project on its own, then choose your own metrics and models when you evaluate.

`deepeval_goldens.py` loads the Hugging Face dataset [`srinivaspadmanabhuni/agentic-ai-test-checklist-110`](https://huggingface.co/datasets/srinivaspadmanabhuni/agentic-ai-test-checklist-110), config `examples` (one row per checklist example). The full checklist is config `all`. Labelled traces are config `traces`. Building the Goldens does not need an API key.

## Run

```bash
pip install deepeval datasets
python examples/community/agentest/deepeval_goldens.py
```

The script prints how many Goldens it built and the first example's `test_id`.

Contact: srinivas AT testAIng.com
