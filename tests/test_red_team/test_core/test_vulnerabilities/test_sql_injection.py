import pytest

from deepteam.vulnerabilities import SQLInjection
from deepteam.vulnerabilities.sql_injection import SQLInjectionType
from deepteam.test_case import RTTestCase


class TestSQLInjection:

    def test_sql_injection_all_types(self):
        types = [
            "blind_sql_injection",
            "union_based_injection",
            "error_based_injection",
        ]
        sql_injection = SQLInjection(types=types)
        assert sorted(type.value for type in sql_injection.types) == sorted(
            types
        )

    def test_sql_injection_all_types_default(self):
        sql_injection = SQLInjection()
        assert sorted(type.value for type in sql_injection.types) == sorted(
            type.value for type in SQLInjectionType
        )

    def test_sql_injection_blind_sql_injection(self):
        types = ["blind_sql_injection"]
        sql_injection = SQLInjection(types=types)
        assert sorted(type.value for type in sql_injection.types) == sorted(
            types
        )

    def test_sql_injection_union_based_injection(self):
        types = ["union_based_injection"]
        sql_injection = SQLInjection(types=types)
        assert sorted(type.value for type in sql_injection.types) == sorted(
            types
        )

    def test_sql_injection_error_based_injection(self):
        types = ["error_based_injection"]
        sql_injection = SQLInjection(types=types)
        assert sorted(type.value for type in sql_injection.types) == sorted(
            types
        )

    def test_sql_injection_all_types_invalid(self):
        types = [
            "blind_sql_injection",
            "union_based_injection",
            "error_based_injection",
            "invalid",
        ]
        with pytest.raises(ValueError):
            SQLInjection(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        sql_injection = SQLInjection(types=["blind_sql_injection"])
        test_cases = sql_injection.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "SQL Injection" for tc in test_cases)
        assert all(
            tc.vulnerability_type == SQLInjectionType.BLIND_SQL_INJECTION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        sql_injection = SQLInjection(
            types=["blind_sql_injection"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = sql_injection.assess(
            model_callback=dummy_model_callback,
        )

        assert sql_injection.is_vulnerable() is not None
        assert sql_injection.simulated_attacks is not None and isinstance(
            sql_injection.simulated_attacks, dict
        )
        assert sql_injection.res is not None and isinstance(
            sql_injection.res, dict
        )
        assert SQLInjectionType.BLIND_SQL_INJECTION in results
        assert len(results[SQLInjectionType.BLIND_SQL_INJECTION]) == 1
        test_case = results[SQLInjectionType.BLIND_SQL_INJECTION][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_SQLInjection_metric(self):
        from deepteam.metrics import SQLInjectionMetric

        sql_injection = SQLInjection(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = sql_injection._get_metric(SQLInjectionType.BLIND_SQL_INJECTION)
        assert isinstance(metric, SQLInjectionMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        sql_injection = SQLInjection(
            types=["blind_sql_injection"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await sql_injection.a_assess(
            model_callback=dummy_model_callback,
        )

        assert sql_injection.is_vulnerable() is not None
        assert sql_injection.simulated_attacks is not None and isinstance(
            sql_injection.simulated_attacks, dict
        )
        assert sql_injection.res is not None and isinstance(
            sql_injection.res, dict
        )
        assert SQLInjectionType.BLIND_SQL_INJECTION in results
        assert len(results[SQLInjectionType.BLIND_SQL_INJECTION]) == 1
        test_case = results[SQLInjectionType.BLIND_SQL_INJECTION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
