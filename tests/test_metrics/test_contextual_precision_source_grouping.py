from types import MethodType

from deepeval.metrics.contextual_precision.contextual_precision import (
    ContextualPrecisionMetric,
)
from deepeval.metrics.utils import group_retrieval_contexts_by_source
from deepeval.metrics.turn_contextual_precision.turn_contextual_precision import (
    TurnContextualPrecisionMetric,
)
from deepeval.test_case import RetrievedContextData, Turn
import pytest


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


def test_source_grouping_warns_and_skips_empty_sources():
    with pytest.warns(UserWarning, match="empty source"):
        grouped_contexts = group_retrieval_contexts_by_source(
            [
                RetrievedContextData(
                    source="",
                    context="context without source",
                ),
                RetrievedContextData(
                    source="chunk-hash/revenue",
                    context="context with source",
                ),
            ]
        )

    assert grouped_contexts == [
        "context without source",
        "Source: chunk-hash/revenue\ncontext with source",
    ]


def test_source_grouping_warns_when_source_granularity_is_too_broad():
    with pytest.warns(UserWarning, match="may be too broad"):
        grouped_contexts = group_retrieval_contexts_by_source(
            [
                RetrievedContextData(
                    source="document",
                    context="revenue section",
                ),
                RetrievedContextData(
                    source="document",
                    context="risk factor section",
                ),
                RetrievedContextData(
                    source="document",
                    context="appendix section",
                ),
                RetrievedContextData(
                    source="chunk-hash/guidance",
                    context="guidance section",
                ),
            ]
        )

    assert grouped_contexts == [
        "Source: document\n"
        "revenue section\n---\nrisk factor section\n---\nappendix section",
        "Source: chunk-hash/guidance\nguidance section",
    ]
