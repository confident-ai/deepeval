"""Build a DeepEval EvaluationDataset of Goldens from the agentest checklist (external example).
No API key needed to build the dataset; pick your own metric/model to evaluate it.
"""

from datasets import load_dataset
from deepeval.dataset import EvaluationDataset, Golden

rows = load_dataset(
    "srinivaspadmanabhuni/agentic-ai-test-checklist-110",
    "examples",
    split="train",
)
goldens = [
    Golden(
        input=r["input"],
        expected_output=r["expected_behavior"],
        context=[r["scenario"]],
        additional_metadata={
            "test_id": r["test_id"],
            "example_id": r["example_id"],
            "severity": r["severity"],
            "fail_signal": r["fail_signal"],
        },
    )
    for r in rows
]
dataset = EvaluationDataset(goldens=goldens)
print(
    f"{len(dataset.goldens)} goldens; "
    f"first: {dataset.goldens[0].additional_metadata['test_id']}"
)
