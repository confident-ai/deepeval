from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class ProductionDebtReport:
    test_case_id: str
    pdi_score: float  # Production Debt Index (target <= 15.0)
    token_inflation_multiplier: float  # Target <= 1.15x
    step_latency_seconds: float  # Target <= 1.5s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for DeepEval CI/CD test runs.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_test_evaluation(
        self,
        test_case_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{test_case_id}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "test_case_id": test_case_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtEvaluator:
    """
    A2Z SOC Production Debt & Technical Due Diligence Evaluator for DeepEval.

    Quantifies LLM test cases against 4 Enterprise Forward Deployed Engineering KPIs:
    1. Production Debt Index (PDI <= 15.0)
    2. Context Token Inflation (TIM <= 1.15x)
    3. P99 Reasoning Step Latency (<= 1.5s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_pdi: float = 15.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_pdi = max_acceptable_pdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_test_case(
        self,
        test_case_id: str,
        input_tokens: int = 1000,
        output_tokens: int = 120,
        step_latency_seconds: float = 0.85,
        reasoning_loops: int = 0,
        un_gated_mutations: int = 0,
    ) -> ProductionDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_test_evaluation(
                test_case_id=test_case_id,
                event_type="evaluation_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. CI/CD evaluation halted."
            )

        critical_smells: List[str] = []

        # KPI 2: Token Inflation Multiplier
        token_ratio = (input_tokens + output_tokens) / max(1, input_tokens)
        if token_ratio > 2.0:
            critical_smells.append(f"HIGH_TOKEN_INFLATION_{token_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if step_latency_seconds > 4.0:
            critical_smells.append(f"HIGH_STEP_LATENCY_{step_latency_seconds:.2f}S")

        # Reasoning Loops
        if reasoning_loops > 2:
            critical_smells.append(f"DETECTED_{reasoning_loops}_REASONING_LOOPS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_MUTATIONS")

        # KPI 1: Production Debt Index (0 = Clean, 100 = Catastrophic)
        pdi = (
            max(0.0, (token_ratio - 1.0) * 15.0)
            + max(0.0, (step_latency_seconds - 1.5) * 10.0)
            + (reasoning_loops * 12.0)
            + (un_gated_mutations * 25.0)
        )
        pdi_score = round(min(100.0, pdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - pdi_score)
        is_production_ready = (
            pdi_score <= self.max_acceptable_pdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_test_evaluation(
            test_case_id=test_case_id,
            event_type="diligence_passed" if is_production_ready else "diligence_failed_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "pdi_score": pdi_score,
                "token_ratio": token_ratio,
                "step_latency_seconds": step_latency_seconds,
                "reasoning_loops": reasoning_loops,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return ProductionDebtReport(
            test_case_id=test_case_id,
            pdi_score=pdi_score,
            token_inflation_multiplier=round(token_ratio, 2),
            step_latency_seconds=round(step_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )


class ProductionDebtMetric:
    """
    DeepEval Metric Wrapper for Production Debt & Technical Due Diligence.
    """

    def __init__(
        self,
        threshold: float = 85.0,
        never_equate_intent_to_approval: bool = True,
    ) -> None:
        self.threshold = threshold
        self.evaluator = ProductionDebtEvaluator(
            never_equate_intent_to_approval=never_equate_intent_to_approval,
            max_acceptable_pdi=max(0.0, 100.0 - threshold),
        )
        self.score: float = 0.0
        self.reason: Optional[str] = None
        self.success: bool = False

    def measure(
        self,
        test_case_id: str = "case_0",
        input_tokens: int = 1000,
        output_tokens: int = 100,
        step_latency: float = 0.8,
        reasoning_loops: int = 0,
        un_gated_mutations: int = 0,
    ) -> float:
        report = self.evaluator.evaluate_test_case(
            test_case_id=test_case_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            step_latency_seconds=step_latency,
            reasoning_loops=reasoning_loops,
            un_gated_mutations=un_gated_mutations,
        )
        self.score = report.production_readiness_index
        self.success = report.is_production_ready and self.score >= self.threshold
        self.reason = (
            f"Production Debt Score {self.score}/100. Smells: {report.critical_smells}"
            if not self.success
            else "Production readiness passed with zero architectural smells."
        )
        return self.score

    def is_successful(self) -> bool:
        return self.success
