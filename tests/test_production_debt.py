import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../deepeval/metrics/production_debt/production_debt.py",
)
spec = importlib.util.spec_from_file_location("deepeval_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["deepeval_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtEvaluator = production_debt_mod.ProductionDebtEvaluator
ProductionDebtMetric = production_debt_mod.ProductionDebtMetric
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = ProductionDebtMetric(threshold=85.0)

    def test_clean_eval_passes_production_readiness(self) -> None:
        score = self.metric.measure(
            test_case_id="tc_rag_001",
            input_tokens=1000,
            output_tokens=100,
            step_latency=0.75,
            reasoning_loops=0,
            un_gated_mutations=0,
        )
        self.assertTrue(self.metric.is_successful())
        self.assertGreaterEqual(score, 85.0)

    def test_degraded_eval_fails_production_debt(self) -> None:
        score = self.metric.measure(
            test_case_id="tc_agent_loop_999",
            input_tokens=1000,
            output_tokens=3000,  # High token inflation (4.0x)
            step_latency=6.5,  # High latency
            reasoning_loops=5,  # 5 loops
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(self.metric.is_successful())
        self.assertLess(score, 50.0)

    def test_cryptographic_ledger_integrity(self) -> None:
        evaluator = ProductionDebtEvaluator()
        evaluator.evaluate_test_case("tc-1")
        evaluator.evaluate_test_case("tc-2")
        evaluator.evaluate_test_case("tc-3")

        entries = evaluator.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(evaluator.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
