import pytest

from deepteam.vulnerabilities import Competition
from deepteam.vulnerabilities.competition import CompetitionType
from deepteam.test_case import RTTestCase


class TestCompetition:

    def test_competition_all_types(self):
        types = [
            "competitor_mention",
            "market_manipulation",
            "discreditation",
            "confidential_strategies",
        ]
        competition = Competition(types=types)
        assert sorted(type.value for type in competition.types) == sorted(types)

    def test_competition_all_types_default(self):
        competition = Competition()
        assert sorted(type.value for type in competition.types) == sorted(
            type.value for type in CompetitionType
        )

    def test_competition_competitor_mention(self):
        types = ["competitor_mention"]
        competition = Competition(types=types)
        assert sorted(type.value for type in competition.types) == sorted(types)

    def test_competition_market_manipulation(self):
        types = ["market_manipulation"]
        competition = Competition(types=types)
        assert sorted(type.value for type in competition.types) == sorted(types)

    def test_competition_discreditation(self):
        types = ["discreditation"]
        competition = Competition(types=types)
        assert sorted(type.value for type in competition.types) == sorted(types)

    def test_competition_confidential_strategies(self):
        types = ["confidential_strategies"]
        competition = Competition(types=types)
        assert sorted(type.value for type in competition.types) == sorted(types)

    def test_competition_all_types_invalid(self):
        types = [
            "competitor_mention",
            "market_manipulation",
            "discreditation",
            "confidential_strategies",
            "invalid",
        ]
        with pytest.raises(ValueError):
            Competition(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        competition = Competition(types=["competitor_mention"])
        test_cases = competition.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Competition" for tc in test_cases)
        assert all(
            tc.vulnerability_type == CompetitionType.COMPETITOR_MENTION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        competition = Competition(
            types=["competitor_mention"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = competition.assess(
            model_callback=dummy_model_callback,
        )

        assert competition.is_vulnerable() is not None
        assert competition.simulated_attacks is not None and isinstance(
            competition.simulated_attacks, dict
        )
        assert competition.res is not None and isinstance(competition.res, dict)
        assert CompetitionType.COMPETITOR_MENTION in results
        assert len(results[CompetitionType.COMPETITOR_MENTION]) == 1
        test_case = results[CompetitionType.COMPETITOR_MENTION][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_Competition_metric(self):
        from deepteam.metrics import CompetitorsMetric

        competition = Competition(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = competition._get_metric(CompetitionType.COMPETITOR_MENTION)
        assert isinstance(metric, CompetitorsMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        competition = Competition(types=["competitor_mention"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await competition.a_assess(
            model_callback=dummy_model_callback,
        )

        assert competition.is_vulnerable() is not None
        assert competition.simulated_attacks is not None and isinstance(
            competition.simulated_attacks, dict
        )
        assert competition.res is not None and isinstance(competition.res, dict)
        assert CompetitionType.COMPETITOR_MENTION in results
        assert len(results[CompetitionType.COMPETITOR_MENTION]) == 1
        test_case = results[CompetitionType.COMPETITOR_MENTION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
