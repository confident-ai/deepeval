import pytest

from deepeval.red_team.vulnerabilities import PathTraversal
from deepeval.red_team.vulnerabilities.path_traversal import PathTraversalType
from deepeval.red_team.vulnerabilities.path_traversal import (
    PathTraversalTemplate,
)
from deepeval.red_team.test_case import RTTestCase


class TestPathTraversal:

    def test_path_traversal_all_types(self):
        types = [
            "relative_path_traversal",
            "absolute_path_traversal",
            "encoded_path_traversal",
        ]
        path_traversal = PathTraversal(types=types)
        assert sorted(type.value for type in path_traversal.types) == sorted(
            types
        )

    def test_path_traversal_all_types_default(self):
        path_traversal = PathTraversal()
        assert sorted(type.value for type in path_traversal.types) == sorted(
            type.value for type in PathTraversalType
        )

    def test_path_traversal_relative(self):
        types = ["relative_path_traversal"]
        path_traversal = PathTraversal(types=types)
        assert sorted(type.value for type in path_traversal.types) == sorted(
            types
        )

    def test_path_traversal_absolute(self):
        types = ["absolute_path_traversal"]
        path_traversal = PathTraversal(types=types)
        assert sorted(type.value for type in path_traversal.types) == sorted(
            types
        )

    def test_path_traversal_encoded(self):
        types = ["encoded_path_traversal"]
        path_traversal = PathTraversal(types=types)
        assert sorted(type.value for type in path_traversal.types) == sorted(
            types
        )

    def test_path_traversal_all_types_invalid(self):
        types = [
            "relative_path_traversal",
            "absolute_path_traversal",
            "encoded_path_traversal",
            "invalid",
        ]
        with pytest.raises(ValueError):
            PathTraversal(types=types)

    def test_template_generates_attacks_for_each_type(self):
        # Pure (no model): every sub-type must dispatch to a distinct,
        # non-empty baseline-attack prompt that asks for JSON output.
        prompts = {}
        for vuln_type in PathTraversalType:
            prompt = PathTraversalTemplate.generate_baseline_attacks(
                vuln_type, max_goldens=3, purpose="document retrieval assistant"
            )
            assert isinstance(prompt, str) and prompt.strip()
            assert "JSON" in prompt
            assert "3" in prompt
            prompts[vuln_type] = prompt
        # Each sub-type produces its own prompt, not a shared one.
        assert len(set(prompts.values())) == len(list(PathTraversalType))

    def test_simulate_attacks_returns_expected_cases(self):
        path_traversal = PathTraversal(types=["relative_path_traversal"])
        test_cases = path_traversal.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Path Traversal" for tc in test_cases)
        assert all(
            tc.vulnerability_type == PathTraversalType.RELATIVE_PATH_TRAVERSAL
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        path_traversal = PathTraversal(
            types=["relative_path_traversal"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = path_traversal.assess(model_callback=dummy_model_callback)

        assert path_traversal.is_vulnerable() is not None
        assert path_traversal.simulated_attacks is not None and isinstance(
            path_traversal.simulated_attacks, dict
        )
        assert path_traversal.res is not None and isinstance(
            path_traversal.res, dict
        )
        assert PathTraversalType.RELATIVE_PATH_TRAVERSAL in results
        assert len(results[PathTraversalType.RELATIVE_PATH_TRAVERSAL]) == 1
        test_case = results[PathTraversalType.RELATIVE_PATH_TRAVERSAL][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_PathTraversal_metric(self):
        from deepeval.red_team._metrics import PathTraversalMetric

        path_traversal = PathTraversal(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = path_traversal._get_metric(
            PathTraversalType.RELATIVE_PATH_TRAVERSAL
        )
        assert isinstance(metric, PathTraversalMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        path_traversal = PathTraversal(
            types=["relative_path_traversal"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await path_traversal.a_assess(
            model_callback=dummy_model_callback
        )

        assert path_traversal.is_vulnerable() is not None
        assert path_traversal.simulated_attacks is not None and isinstance(
            path_traversal.simulated_attacks, dict
        )
        assert path_traversal.res is not None and isinstance(
            path_traversal.res, dict
        )
        assert PathTraversalType.RELATIVE_PATH_TRAVERSAL in results
        assert len(results[PathTraversalType.RELATIVE_PATH_TRAVERSAL]) == 1
        test_case = results[PathTraversalType.RELATIVE_PATH_TRAVERSAL][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
