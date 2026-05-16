import json
import asyncio
from typing import List, Optional, Union, Dict

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.process_integrity.schema import StepVerdict, ProcessIntegrityResult
from deepeval.metrics.process_integrity.template import ProcessIntegrityTemplate


class ProcessIntegrityMetric(BaseMetric):
    """
    Evaluates whether an agent's reasoning steps are logically consistent
    across a trajectory. Detects contradictions, skipped intermediate logic,
    and irrelevant conclusions inserted mid-trajectory.

    Two-pass LLM pipeline:
    Pass 1 — extract the logical conclusion from each step (non-substantive
              steps like raw tool calls receive verdict SKIP).
    Pass 2 — for each substantive step, check whether its conclusion is
              consistent with all prior conclusions.

    Score = PASS / (PASS + FAIL). SKIP steps excluded from denominator.
    self.score_breakdown is populated with per-step verdicts.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.score_breakdown: Dict = {}

    def measure(self, test_case: LLMTestCase) -> float:
        steps = self._get_steps(test_case)
        conclusions = self._extract_conclusions(steps)
        verdicts = self._check_integrity(steps, conclusions)
        self.score = self._compute_score(verdicts)
        self.score_breakdown = self._build_score_breakdown(verdicts)
        self.reason = self._build_reason(verdicts)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        steps = self._get_steps(test_case)
        conclusions = await self._a_extract_conclusions(steps)
        verdicts = await self._a_check_integrity(steps, conclusions)
        self.score = self._compute_score(verdicts)
        self.score_breakdown = self._build_score_breakdown(verdicts)
        self.reason = self._build_reason(verdicts)
        self.success = self.score >= self.threshold
        return self.score

    def _get_steps(self, test_case: LLMTestCase) -> List[str]:
        if not hasattr(test_case, "steps") or not test_case.steps:
            raise ValueError(
                "ProcessIntegrityMetric requires test_case.steps to be a non-empty list of strings."
            )
        return test_case.steps

    # ── Pass 1: extract conclusions ──────────────────────────────────────────

    def _extract_conclusions(self, steps: List[str]) -> List[dict]:
        results = []
        for i, step in enumerate(steps):
            prompt = ProcessIntegrityTemplate.extract_conclusion(step)
            raw = self._call_model(prompt)
            parsed = self._parse_json(raw)
            results.append({
                "step_index": i,
                "conclusion": parsed.get("conclusion", ""),
                "is_substantive": parsed.get("is_substantive", True),
            })
        return results

    async def _a_extract_conclusions(self, steps: List[str]) -> List[dict]:
        tasks = []
        for i, step in enumerate(steps):
            prompt = ProcessIntegrityTemplate.extract_conclusion(step)
            tasks.append(self._a_call_model(prompt, step_index=i))
        raw_results = await asyncio.gather(*tasks)
        results = []
        for i, raw in enumerate(raw_results):
            parsed = self._parse_json(raw)
            results.append({
                "step_index": i,
                "conclusion": parsed.get("conclusion", ""),
                "is_substantive": parsed.get("is_substantive", True),
            })
        return results

    # ── Pass 2: check integrity ──────────────────────────────────────────────

    def _check_integrity(
        self, steps: List[str], conclusions: List[dict]
    ) -> List[StepVerdict]:
        verdicts = []
        prior_substantive: List[dict] = []

        for i, (step, conclusion_data) in enumerate(zip(steps, conclusions)):
            if not conclusion_data["is_substantive"]:
                verdicts.append(StepVerdict(
                    step_index=i,
                    verdict="SKIP",
                    reason="Non-substantive step (tool call or mechanical action).",
                    conclusion="",
                ))
                continue

            if i == 0 or not prior_substantive:
                verdicts.append(StepVerdict(
                    step_index=i,
                    verdict="PASS",
                    reason="First substantive step — no prior conclusions to check against.",
                    conclusion=conclusion_data["conclusion"],
                ))
                prior_substantive.append({
                    "step_index": i,
                    "conclusion": conclusion_data["conclusion"],
                })
                continue

            prompt = ProcessIntegrityTemplate.check_integrity(
                step=step,
                conclusion=conclusion_data["conclusion"],
                prior_conclusions=prior_substantive,
            )
            raw = self._call_model(prompt)
            parsed = self._parse_json(raw)
            verdict_str = parsed.get("verdict", "PASS")
            verdicts.append(StepVerdict(
                step_index=i,
                verdict=verdict_str,
                reason=parsed.get("reason", ""),
                conclusion=conclusion_data["conclusion"],
            ))
            prior_substantive.append({
                "step_index": i,
                "conclusion": conclusion_data["conclusion"],
            })

        return verdicts

    async def _a_check_integrity(
        self, steps: List[str], conclusions: List[dict]
    ) -> List[StepVerdict]:
        # Sequential for integrity — each step depends on prior verdicts
        return self._check_integrity(steps, conclusions)

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _compute_score(self, verdicts: List[StepVerdict]) -> float:
        pass_count = sum(1 for v in verdicts if v.verdict == "PASS")
        fail_count = sum(1 for v in verdicts if v.verdict == "FAIL")
        denominator = pass_count + fail_count
        if denominator == 0:
            self.reason = "No substantive steps found — all steps were SKIP."
            return 0.0
        return pass_count / denominator

    def _build_score_breakdown(self, verdicts: List[StepVerdict]) -> Dict:
        return {
            f"step_{v.step_index}": {
                "verdict": v.verdict,
                "reason": v.reason,
                "conclusion": v.conclusion,
            }
            for v in verdicts
        }

    def _build_reason(self, verdicts: List[StepVerdict]) -> str:
        fail_verdicts = [v for v in verdicts if v.verdict == "FAIL"]
        if not fail_verdicts:
            return "All substantive steps are logically consistent."
        reasons = "; ".join(
            f"Step {v.step_index}: {v.reason}" for v in fail_verdicts
        )
        return f"{len(fail_verdicts)} integrity failure(s) detected — {reasons}"

    # ── Model call helpers ───────────────────────────────────────────────────

    def _call_model(self, prompt: str) -> str:
        if self.model is None:
            raise ValueError("No model provided. Pass a model= argument to ProcessIntegrityMetric.")
        if isinstance(self.model, str):
            raise ValueError("String model names are not supported directly. Pass a DeepEvalBaseLLM instance.")
        response, _ = self.model.generate(prompt)
        return response

    async def _a_call_model(self, prompt: str, **kwargs) -> str:
        if self.model is None:
            raise ValueError("No model provided.")
        if isinstance(self.model, str):
            raise ValueError("String model names are not supported directly. Pass a DeepEvalBaseLLM instance.")
        response, _ = await self.model.a_generate(prompt)
        return response

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {}

    # ── Required BaseMetric properties ───────────────────────────────────────

    @property
    def __name__(self):
        return "Process Integrity"
