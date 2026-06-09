from types import MethodType

from deepeval.metrics.contextual_precision.contextual_precision import (
    ContextualPrecisionMetric,
)
from deepeval.metrics.turn_contextual_precision.turn_contextual_precision import (
    TurnContextualPrecisionMetric,
)
from deepeval.test_case import RetrievedContextData, Turn


def test_contextual_precision_groups_retrieved_context_data_by_source():
    metric = object.__new__(ContextualPrecisionMetric)

    grouped_contexts = metric._group_retrieval_contexts(
        [
            "standalone introduction",
            RetrievedContextData(
                source="chunk-hash/revenue",
                context="dense retrieval copy",
            ),
            "standalone appendix",
            RetrievedContextData(
                source="chunk-hash/revenue",
                context="sparse retrieval copy",
            ),
            RetrievedContextData(
                source="chunk-hash/risk",
                context="risk factor section",
            ),
        ]
    )

    assert grouped_contexts == [
        "standalone introduction",
        "Source: chunk-hash/revenue\n"
        "dense retrieval copy\n---\nsparse retrieval copy",
        "standalone appendix",
        "Source: chunk-hash/risk\nrisk factor section",
    ]


def test_turn_contextual_precision_groups_retrieved_context_data_by_source():
    metric = object.__new__(TurnContextualPrecisionMetric)
    metric.include_reason = False
    metric.strict_mode = False
    metric.threshold = 0.5
    generated_contexts = []

    def fake_generate_verdicts(
        self, input, expected_outcome, retrieval_context, multimodal
    ):
        generated_contexts.append(retrieval_context)
        return []

    metric._generate_verdicts = MethodType(fake_generate_verdicts, metric)

    metric._get_contextual_precision_scores(
        [
            Turn(role="user", content="What was revenue?"),
            Turn(
                role="assistant",
                content="Revenue was up.",
                retrieval_context=[
                    RetrievedContextData(
                        source="chunk-hash/revenue",
                        context="dense retrieval copy",
                    ),
                    "standalone appendix",
                    RetrievedContextData(
                        source="chunk-hash/revenue",
                        context="sparse retrieval copy",
                    ),
                ],
            ),
        ],
        expected_outcome="The assistant should answer from revenue data.",
        multimodal=False,
    )

    assert generated_contexts == [
        [
            "Source: chunk-hash/revenue\n"
            "dense retrieval copy\n---\nsparse retrieval copy",
            "standalone appendix",
        ]
    ]
