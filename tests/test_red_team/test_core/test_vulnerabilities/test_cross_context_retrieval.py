import pytest

from deepteam.vulnerabilities import CrossContextRetrieval
from deepteam.vulnerabilities.cross_context_retrieval import (
    CrossContextRetrievalType,
)
from deepteam.test_case import RTTestCase


class TestCrossContextRetrieval:

    def test_cross_context_retrieval_all_types(self):
        types = ["tenant", "user", "role"]
        cross_context_retrieval = CrossContextRetrieval(types=types)
        assert sorted(
            type.value for type in cross_context_retrieval.types
        ) == sorted(types)

    def test_cross_context_retrieval_all_types_default(self):
        cross_context_retrieval = CrossContextRetrieval()
        assert sorted(
            type.value for type in cross_context_retrieval.types
        ) == sorted(type.value for type in CrossContextRetrievalType)

    def test_cross_context_retrieval_tenant(self):
        types = ["tenant"]
        cross_context_retrieval = CrossContextRetrieval(types=types)
        assert sorted(
            type.value for type in cross_context_retrieval.types
        ) == sorted(types)

    def test_cross_context_retrieval_user(self):
        types = ["user"]
        cross_context_retrieval = CrossContextRetrieval(types=types)
        assert sorted(
            type.value for type in cross_context_retrieval.types
        ) == sorted(types)

    def test_cross_context_retrieval_role(self):
        types = ["role"]
        cross_context_retrieval = CrossContextRetrieval(types=types)
        assert sorted(
            type.value for type in cross_context_retrieval.types
        ) == sorted(types)

    def test_cross_context_retrieval_all_types_invalid(self):
        types = ["tenant", "user", "role", "invalid"]
        with pytest.raises(ValueError):
            CrossContextRetrieval(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        cross_context_retrieval = CrossContextRetrieval(types=["tenant"])
        test_cases = cross_context_retrieval.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Cross-Context Retrieval" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == CrossContextRetrievalType.TENANT
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        cross_context_retrieval = CrossContextRetrieval(
            types=["user"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = cross_context_retrieval.assess(
            model_callback=dummy_model_callback,
        )

        assert cross_context_retrieval.is_vulnerable() is not None
        assert (
            cross_context_retrieval.simulated_attacks is not None
            and isinstance(cross_context_retrieval.simulated_attacks, dict)
        )
        assert cross_context_retrieval.res is not None and isinstance(
            cross_context_retrieval.res, dict
        )
        assert CrossContextRetrievalType.USER in results
        assert len(results[CrossContextRetrievalType.USER]) == 1
        test_case = results[CrossContextRetrievalType.USER][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_cross_context_retrieval_metric(self):
        from deepteam.metrics import CrossContextRetrievalMetric

        cross_context_retrieval = CrossContextRetrieval(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = cross_context_retrieval._get_metric(
            CrossContextRetrievalType.ROLE
        )
        assert isinstance(metric, CrossContextRetrievalMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        cross_context_retrieval = CrossContextRetrieval(
            types=["user"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await cross_context_retrieval.a_assess(
            model_callback=dummy_model_callback,
        )

        assert cross_context_retrieval.is_vulnerable() is not None
        assert (
            cross_context_retrieval.simulated_attacks is not None
            and isinstance(cross_context_retrieval.simulated_attacks, dict)
        )
        assert cross_context_retrieval.res is not None and isinstance(
            cross_context_retrieval.res, dict
        )
        assert CrossContextRetrievalType.USER in results
        assert len(results[CrossContextRetrievalType.USER]) == 1
        test_case = results[CrossContextRetrievalType.USER][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
