import pytest

from deepeval.benchmarks import PubMedQA
from deepeval.benchmarks.pub_med_qa.template import PubMedQATemplate
from deepeval.dataset import Golden


class StructuredModel:
    def __init__(self, return_cost=False):
        self.return_cost = return_cost

    def generate(self, prompt, schema):
        assert "JSON object" in prompt
        if "treatment" in prompt:
            answer = "yes"
        elif "no benefit" in prompt:
            answer = "no"
        else:
            answer = "maybe"
        response = schema(answer=answer)
        return (response, 0.0) if self.return_cost else response


class UnstructuredModel:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return " YES \n"


def test_format_question_preserves_abstract_sections():
    data = {
        "question": "Does the treatment work?",
        "context": {
            "contexts": ["Study objective.", "The treatment helped."],
            "labels": ["BACKGROUND", "RESULTS"],
        },
        "final_decision": "yes",
    }

    prompt = PubMedQATemplate.format_question(data)

    assert "Question: Does the treatment work?" in prompt
    assert "BACKGROUND: Study objective." in prompt
    assert "RESULTS: The treatment helped." in prompt
    assert prompt.endswith("Answer (yes, no, or maybe): ")
    assert PubMedQATemplate.format_answer(data) == "yes"


def test_format_question_handles_missing_section_labels():
    data = {
        "question": "Is the evidence conclusive?",
        "context": {"contexts": ["More research is needed."], "labels": []},
        "final_decision": "MAYBE",
    }

    prompt = PubMedQATemplate.format_question(data)

    assert "More research is needed." in prompt
    assert PubMedQATemplate.format_answer(data) == "maybe"


def test_predict_supports_structured_and_unstructured_models():
    benchmark = PubMedQA(n_problems=1, dataset={"train": []})
    golden = Golden(input="Does the treatment work?", expected_output="yes")

    structured_result = benchmark.predict(StructuredModel(), golden)
    fallback_model = UnstructuredModel()
    fallback_result = benchmark.predict(fallback_model, golden)

    assert structured_result == {"prediction": "yes", "score": 1}
    assert fallback_result == {"prediction": "yes", "score": 1}
    assert fallback_model.prompt.endswith(
        "Output only 'yes', 'no', or 'maybe'."
    )
    assert "JSON object" not in fallback_model.prompt


def test_predict_handles_native_model_cost_tuple():
    benchmark = PubMedQA(n_problems=1, dataset={"train": []})
    golden = Golden(input="Does the treatment work?", expected_output="yes")

    result = benchmark.predict(StructuredModel(return_cost=True), golden)

    assert result == {"prediction": "yes", "score": 1}


def test_evaluate_uses_available_goldens_for_accuracy():
    dataset = {
        "train": [
            {
                "question": "Does the treatment work?",
                "context": {
                    "contexts": ["The treatment helped."],
                    "labels": ["RESULTS"],
                },
                "final_decision": "yes",
            },
            {
                "question": "Is the evidence conclusive?",
                "context": {
                    "contexts": ["More research is needed."],
                    "labels": ["CONCLUSIONS"],
                },
                "final_decision": "maybe",
            },
            {
                "question": "Was there no benefit?",
                "context": {
                    "contexts": ["The intervention did not improve outcomes."],
                    "labels": ["RESULTS"],
                },
                "final_decision": "no",
            },
        ]
    }
    benchmark = PubMedQA(n_problems=10, dataset=dataset)

    result = benchmark.evaluate(StructuredModel())

    assert result.overall_accuracy == pytest.approx(1.0)
    assert benchmark.overall_score == pytest.approx(1.0)
    assert benchmark.predictions["Expected Output"].tolist() == [
        "yes",
        "maybe",
        "no",
    ]
    assert benchmark.predictions["Correct"].tolist() == [1, 1, 1]


def test_evaluate_rejects_empty_dataset():
    benchmark = PubMedQA(n_problems=10, dataset={"train": []})

    with pytest.raises(ValueError, match="at least one problem"):
        benchmark.evaluate(StructuredModel())
