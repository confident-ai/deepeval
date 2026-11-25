from deepeval.benchmarks.logi_qa.logi_qa import LogiQA
from deepeval.benchmarks.logi_qa.task import LogiQATask


def test_load_necessary_reasoning_accepts_both_dataset_spellings():
    upstream = "Necessry Condtional Reasoning"
    necessary = LogiQATask.NECESSARY_CONDITIONAL_REASONING
    sufficient = LogiQATask.SUFFICIENT_CONDITIONAL_REASONING

    # Exercise the loader with local records, without fetching the benchmark.
    benchmark = LogiQA.__new__(LogiQA)
    benchmark.dataset = [
        {
            "text": "Read the passage.",
            "question": "Which option follows?",
            "options": ["first", "second", "third", "fourth"],
            "answer": answer,
            "type": labels,
        }
        for answer, labels in enumerate(
            [
                {upstream: True, necessary.value: None},
                {necessary.value: True},
                {upstream: True, necessary.value: True},
                {upstream: False, sufficient.value: True},
            ]
        )
    ]

    assert [
        golden.expected_output
        for golden in benchmark.load_benchmark_dataset(necessary)
    ] == ["A", "B", "C"]
    assert [
        golden.expected_output
        for golden in benchmark.load_benchmark_dataset(sufficient)
    ] == ["D"]
