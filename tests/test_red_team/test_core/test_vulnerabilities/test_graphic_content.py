import pytest

from deepteam.vulnerabilities import GraphicContent
from deepteam.vulnerabilities.graphic_content import GraphicContentType
from deepteam.test_case import RTTestCase


class TestGraphicContent:

    def test_graphic_content_all_types(self):
        types = ["sexual_content", "graphic_content", "pornographic_content"]
        graphic_content = GraphicContent(types=types)
        assert sorted(type.value for type in graphic_content.types) == sorted(
            types
        )

    def test_graphic_content_all_types_default(self):
        graphic_content = GraphicContent()
        assert sorted(type.value for type in graphic_content.types) == sorted(
            type.value for type in GraphicContentType
        )

    def test_graphic_content_sexual_content(self):
        types = ["sexual_content"]
        graphic_content = GraphicContent(types=types)
        assert sorted(type.value for type in graphic_content.types) == sorted(
            types
        )

    def test_graphic_content_graphic_content(self):
        types = ["graphic_content"]
        graphic_content = GraphicContent(types=types)
        assert sorted(type.value for type in graphic_content.types) == sorted(
            types
        )

    def test_graphic_content_pornographic_content(self):
        types = ["pornographic_content"]
        graphic_content = GraphicContent(types=types)
        assert sorted(type.value for type in graphic_content.types) == sorted(
            types
        )

    def test_graphic_content_all_types_invalid(self):
        types = [
            "sexual_content",
            "graphic_content",
            "pornographic_content",
            "invalid",
        ]
        with pytest.raises(ValueError):
            GraphicContent(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        graphic_content = GraphicContent(types=["graphic_content"])
        test_cases = graphic_content.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Graphic Content" for tc in test_cases)
        assert all(
            tc.vulnerability_type == GraphicContentType.GRAPHIC_CONTENT
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        graphic_content = GraphicContent(
            types=["graphic_content"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = graphic_content.assess(
            model_callback=dummy_model_callback,
        )

        assert graphic_content.is_vulnerable() is not None
        assert graphic_content.simulated_attacks is not None and isinstance(
            graphic_content.simulated_attacks, dict
        )
        assert graphic_content.res is not None and isinstance(
            graphic_content.res, dict
        )
        assert GraphicContentType.GRAPHIC_CONTENT in results
        assert len(results[GraphicContentType.GRAPHIC_CONTENT]) == 1
        test_case = results[GraphicContentType.GRAPHIC_CONTENT][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_GraphicContent_metric(self):
        from deepteam.metrics import GraphicMetric

        graphic_content = GraphicContent(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = graphic_content._get_metric(GraphicContentType.GRAPHIC_CONTENT)
        assert isinstance(metric, GraphicMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        graphic_content = GraphicContent(
            types=["graphic_content"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await graphic_content.a_assess(
            model_callback=dummy_model_callback,
        )

        assert graphic_content.is_vulnerable() is not None
        assert graphic_content.simulated_attacks is not None and isinstance(
            graphic_content.simulated_attacks, dict
        )
        assert graphic_content.res is not None and isinstance(
            graphic_content.res, dict
        )
        assert GraphicContentType.GRAPHIC_CONTENT in results
        assert len(results[GraphicContentType.GRAPHIC_CONTENT]) == 1
        test_case = results[GraphicContentType.GRAPHIC_CONTENT][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
