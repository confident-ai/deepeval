"""Run CPU-only native agent evaluation and export deterministic replay files.

Usage: python examples/community/execution_contract/evaluate.py --output-dir ./contract-replay
The exported evidence contains this example's synthetic tool inputs/outputs.
"""

import argparse
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics.community import ExecutionContractMetric
from deepeval.metrics.community.execution_contract import (
    Contract,
    Occurrence,
    Predicate,
    Prerequisite,
    Recovery,
    Selector,
)
from deepeval.metrics.community.execution_contract.artifacts import (
    write_artifact,
)
from deepeval.tracing import observe


@observe(type="tool", name="verify")
def verify(account: str):
    return {"valid": True, "account": account}


@observe(type="tool", name="pay")
def pay(account: str, operation_id: str, fail: bool = False):
    if fail:
        raise RuntimeError("synthetic payment failure")
    return {"paid": True}


@observe(type="tool", name="refund")
def refund(account: str, operation_id: str):
    return {"refunded": True}


@observe(type="tool", name="notify")
def notify(account: str, operation_id: str):
    return "complete"


@observe(type="agent", name="checkout")
def checkout(scenario: str):
    account, operation_id = "alice", "payment-1"
    verify("bob" if scenario == "wrong_subject" else account)
    try:
        pay(
            account, operation_id, fail=scenario in ("recovered", "unrecovered")
        )
    except RuntimeError:
        if scenario == "recovered":
            refund(account, operation_id)
    return notify(account, operation_id)


def workflow_contract():
    return Contract(
        (
            Occurrence(
                "payment_required", Selector(name="pay"), minimum=1, maximum=1
            ),
            Prerequisite(
                "verified_account",
                Selector(name="pay"),
                Selector(
                    name="verify",
                    status="success",
                    predicates=(Predicate("/output/valid", "eq", True),),
                ),
                bindings=(("/input/account", "/input/account"),),
                max_age=60,
            ),
            Recovery(
                "failed_payment_refunded",
                Selector(name="pay", status="error"),
                Selector(name="refund", status="success"),
                bindings=(
                    ("/input/account", "/input/account"),
                    ("/input/operation_id", "/input/operation_id"),
                ),
                within=30,
                before=Selector(name="notify"),
            ),
        )
    )


def run_example(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = workflow_contract()
    write_artifact(output_dir / "contract.json", contract)
    reports = {}
    for scenario in ("valid", "wrong_subject", "recovered", "unrecovered"):
        metric = ExecutionContractMetric(
            contract,
            clock_domain="local-perf-counter",
            capture_evidence=True,
        )
        dataset = EvaluationDataset(goldens=[Golden(input=scenario)])
        for golden in dataset.evals_iterator(
            metrics=[metric],
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=False),
            display_config=DisplayConfig(
                show_indicator=False, print_results=False
            ),
        ):
            checkout(golden.input)
        reports[scenario] = metric.report
        write_artifact(
            output_dir / f"{scenario}.evidence.json", metric.evidence_snapshot
        )
        write_artifact(output_dir / f"{scenario}.report.json", metric.report)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for scenario, report in run_example(args.output_dir).items():
        print(f"{scenario}: {report.status.value} ({report.score:.3f})")
