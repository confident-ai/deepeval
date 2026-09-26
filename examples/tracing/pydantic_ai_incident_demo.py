#!/usr/bin/env python3
"""Multi-agent incident response demo with DeepEval instrumentation.

Run:
    python examples/tracing/pydantic_ai_incident_demo.py
    python examples/tracing/pydantic_ai_incident_demo.py --offline
    python examples/tracing/pydantic_ai_incident_demo.py --model openai:gpt-4o-mini --budget 500

Uses OPENAI_API_KEY and optional CONFIDENT_API_KEY from your environment or
.env.local. Without an OpenAI key, runs a scripted offline demonstration.
All incident data and remediation simulations are fictional and local.
Live runs export directly to Confident AI's OTEL endpoint.

Dependencies (Python 3.10+):
    pip install -e . 'pydantic-ai-slim[openai]>=2.40,<3' \
        opentelemetry-exporter-otlp-proto-http

API references:
    https://ai.pydantic.dev/multi-agent-applications/
    https://ai.pydantic.dev/capabilities/instrumentation/
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Scripted model; no model calls or trace uploads.",
    )
    parser.add_argument("--model", default="openai:gpt-4o-mini")
    parser.add_argument(
        "--budget",
        type=int,
        default=500,
        help="Fictional remediation budget in USD.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".deepeval/incident-demo")
    )
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be nonnegative")
    run_demo(args)


def run_demo(args):
    import asyncio
    from dataclasses import dataclass, field
    import json
    import uuid
    from typing import Literal

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env.local", override=False)
    offline = args.offline or not os.getenv("OPENAI_API_KEY")
    if offline:
        os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, ModelRetry, RunContext
    from pydantic_ai.capabilities import Instrumentation
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RunUsage, UsageLimits
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from deepeval.integrations.pydantic_ai import (
        DeepEvalInstrumentationSettings,
    )
    from deepeval.tracing import (
        observe,
        next_agent_span,
        update_current_span,
        flush_traces,
        trace_manager,
    )

    # A small, deterministic incident database. Evidence IDs make claims auditable.
    evidence = {
        "DEPLOY-1": {
            "service": "checkout",
            "detail": "Checkout 2.18 deployed at 10:02. It changed payment retries from 1 to 8.",
        },
        "METRIC-1": {
            "service": "checkout",
            "detail": "10:00: errors 0.2%, p95 220ms. 10:05: errors 18%, p95 4200ms.",
        },
        "LOG-1": {
            "service": "payments",
            "detail": "Payment worker pool exhausted; repeated identical checkout request IDs.",
        },
        "METRIC-2": {
            "service": "payments",
            "detail": "Request volume rose 7.8x after 10:02; completed orders fell 23%.",
        },
        "RUNBOOK-1": {
            "service": "checkout",
            "detail": "Rollback is safe: release 2.18 had no schema migration. Recovery target: error rate below 1%.",
        },
    }
    remedies = {
        "rollback_checkout": {
            "cost_usd": 0,
            "minutes": 5,
            "predicted_error_pct": 0.3,
        },
        "cap_retries": {
            "cost_usd": 0,
            "minutes": 3,
            "predicted_error_pct": 0.6,
        },
        "scale_payment_workers": {
            "cost_usd": 800,
            "minutes": 12,
            "predicted_error_pct": 4.0,
        },
    }
    ActionName = Literal[
        "rollback_checkout", "cap_retries", "scale_payment_workers"
    ]

    class Finding(BaseModel):
        specialist: str
        conclusion: str
        evidence_ids: list[str] = Field(min_length=2)
        confidence: float = Field(ge=0, le=1)

    class Plan(BaseModel):
        actions: list[ActionName] = Field(min_length=1, max_length=3)
        cost_usd: int = Field(ge=0)
        recovery_minutes: int = Field(gt=0)
        rollback_trigger: str
        rationale: str

    class Review(BaseModel):
        approved: bool
        issues: list[str]
        checks: list[str] = Field(min_length=2)

    class IncidentReport(BaseModel):
        incident_id: str
        severity: Literal["P1", "P2", "P3"]
        root_cause: str
        evidence_ids: list[str] = Field(min_length=3)
        plan: Plan
        next_steps: list[str] = Field(min_length=2)

    @dataclass
    class IncidentState:
        budget: int
        incident_id: str = "INC-2042"
        retrieved: set[str] = field(default_factory=set)
        findings: list[Finding] = field(default_factory=list)
        simulations: dict = field(default_factory=dict)
        plan: object = None
        review: object = None
        retry_events: list[str] = field(default_factory=list)

    state = IncidentState(budget=args.budget)
    run_id = str(uuid.uuid4())
    offline_plan = {
        "actions": ["rollback_checkout", "cap_retries"],
        "cost_usd": 0,
        "recovery_minutes": 5,
        "rollback_trigger": "Escalate if errors remain above 1% after five minutes.",
        "rationale": "Remove the retry amplification and restore the known-good release.",
    }
    offline_review = {
        "approved": True,
        "issues": [],
        "checks": [
            "Simulated cost is within budget.",
            "Rollback is safe per RUNBOOK-1.",
        ],
    }

    def scripted_model(role):
        # This drives real Agent runs/tools/validators. Only model responses are
        # scripted; the first coordinator and payments responses trigger retries.
        step = 0

        def respond(messages, info):
            nonlocal step
            step += 1
            calls = []
            payload = None
            if role == "coordinator":
                if step == 1:
                    payload = {
                        "incident_id": "INC-2042",
                        "severity": "P1",
                        "root_cause": "Unverified early guess.",
                        "evidence_ids": ["DEPLOY-1", "LOG-1", "METRIC-1"],
                        "plan": offline_plan,
                        "next_steps": ["Investigate.", "Monitor."],
                    }
                elif step == 2:
                    calls = [("investigate", {})]
                elif step == 3:
                    calls = [("design_remediation", {})]
                elif step == 4:
                    calls = [("review_remediation", {})]
                else:
                    payload = {
                        "incident_id": state.incident_id,
                        "severity": "P1",
                        "root_cause": "Checkout 2.18 amplifies retries and exhausts the payment worker pool.",
                        "evidence_ids": sorted(state.retrieved),
                        "plan": offline_plan,
                        "next_steps": [
                            "Request operator approval for the simulated remediation.",
                            "Monitor error rate and payment throughput for 15 minutes.",
                        ],
                    }
            elif role in ("checkout", "payments"):
                if role == "payments" and step == 1:
                    calls = [("fetch_evidence", {"evidence_id": "LOG-missing"})]
                elif step == 1 or (role == "payments" and step == 2):
                    ids = (
                        ["DEPLOY-1", "METRIC-1", "RUNBOOK-1"]
                        if role == "checkout"
                        else ["LOG-1", "METRIC-2"]
                    )
                    calls = [
                        ("fetch_evidence", {"evidence_id": eid}) for eid in ids
                    ]
                else:
                    ids = (
                        ["DEPLOY-1", "METRIC-1"]
                        if role == "checkout"
                        else ["LOG-1", "METRIC-2"]
                    )
                    payload = {
                        "specialist": role,
                        "conclusion": "Retry amplification after deployment saturates payments.",
                        "evidence_ids": ids,
                        "confidence": 0.96,
                    }
            elif role == "planner":
                if step == 1:
                    calls = [
                        ("simulate_remedy", {"action": action})
                        for action in offline_plan["actions"]
                    ]
                else:
                    payload = offline_plan
            elif role == "reviewer":
                payload = offline_review
            if payload is not None:
                calls = [(info.output_tools[0].name, payload)]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, arguments, tool_call_id=f"{role}-{step}-{i}"
                    )
                    for i, (name, arguments) in enumerate(calls)
                ]
            )

        return FunctionModel(respond, model_name=f"offline-{role}")

    # Share one SDK settings instance/provider across every agent. Capabilities
    # are separate instances because Pydantic AI stores run state on them.
    settings = DeepEvalInstrumentationSettings(
        name="incident-response-demo",
        thread_id=run_id,
        user_id="demo-operator",
        tags=["pydantic-ai", "multi-agent", "offline" if offline else "live"],
        metadata={
            "incident_id": state.incident_id,
            "simulated_remediation": True,
            "run_id": run_id,
            "budget_usd": args.budget,
        },
    )
    provider = otel_trace.get_tracer_provider()
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    # Offline runs use a disabled REST pipeline for local capture only.
    # Live runs have no @observe wrapper, so the integration selects OTLP.
    if offline:
        trace_manager.configure(tracing_enabled=False)

    def make_agent(role, output_type, instructions):
        return Agent(
            scripted_model(role) if offline else args.model,
            name=role,
            deps_type=IncidentState,
            output_type=output_type,
            instructions=instructions,
            retries=3,
            capabilities=[Instrumentation(settings=settings)],
        )

    checkout = make_agent(
        "checkout",
        Finding,
        "Investigate checkout incident INC-2042. Call fetch_evidence for DEPLOY-1, METRIC-1, RUNBOOK-1. Explain timing and rollback safety. Cite retrieved evidence IDs.",
    )
    payments = make_agent(
        "payments",
        Finding,
        "Investigate payments incident INC-2042. Call fetch_evidence for LOG-1 and METRIC-2. Explain downstream saturation. Cite retrieved evidence IDs.",
    )
    planner = make_agent(
        "planner",
        Plan,
        "Design a simulated remediation from the findings. You MUST call simulate_remedy for every selected action. Stay within the supplied budget and target errors below 1%. cost_usd is the sum of selected costs; recovery_minutes is the maximum selected duration. Include a rollback trigger. Prefer rollback_checkout and cap_retries if evidence supports them.",
    )
    reviewer = make_agent(
        "reviewer",
        Review,
        "Independently review the provided findings, plan, simulations, and budget. Check supporting evidence, cost, expected recovery, and rollback safety. Approve only if all constraints are met. Include concrete checks and any issues.",
    )
    coordinator = make_agent(
        "coordinator",
        IncidentReport,
        "You are the incident commander. First call investigate, then design_remediation, then review_remediation. These tools depend on each other; do not call them in parallel. If review rejects the plan, call design_remediation again with the reviewer feedback available in state, then review again. Return a P1/P2/P3 incident report using retrieved evidence and the exact reviewed plan. Propose operator actions; no real infrastructure actions are executed.",
    )

    async def fetch_evidence(
        ctx: RunContext[IncidentState], evidence_id: str
    ) -> dict:
        """Retrieve one incident record by its exact evidence ID."""
        if evidence_id not in evidence:
            ctx.deps.retry_events.append("unknown evidence ID")
            raise ModelRetry(
                f"Unknown evidence ID. Choose from {sorted(evidence)}."
            )
        await asyncio.sleep(0.01)  # Simulate concurrent database reads.
        ctx.deps.retrieved.add(evidence_id)
        update_current_span(
            metadata={
                "evidence_id": evidence_id,
                "source": "local-incident-fixture",
            }
        )
        return {"id": evidence_id, **evidence[evidence_id]}

    checkout.tool(fetch_evidence)
    payments.tool(fetch_evidence)

    async def validate_finding(
        ctx: RunContext[IncidentState], finding: Finding
    ) -> Finding:
        if not set(finding.evidence_ids) <= ctx.deps.retrieved:
            raise ModelRetry(
                "Retrieve every cited evidence ID before submitting findings."
            )
        return finding

    checkout.output_validator(validate_finding)
    payments.output_validator(validate_finding)

    @planner.tool
    async def simulate_remedy(
        ctx: RunContext[IncidentState], action: ActionName
    ) -> dict:
        """Simulate a remedy against fictional metrics; never changes infrastructure."""
        result = {"action": action, **remedies[action], "simulation_only": True}
        ctx.deps.simulations[action] = result
        update_current_span(
            metadata={"action": action, "simulation_only": True}
        )
        return result

    @planner.output_validator
    def validate_plan(ctx: RunContext[IncidentState], plan: Plan) -> Plan:
        if len(plan.actions) != len(set(plan.actions)):
            raise ModelRetry("Do not repeat actions.")
        if any(a not in ctx.deps.simulations for a in plan.actions):
            raise ModelRetry("Simulate every action before proposing it.")
        cost = sum(remedies[a]["cost_usd"] for a in plan.actions)
        duration = max(remedies[a]["minutes"] for a in plan.actions)
        if (
            cost > ctx.deps.budget
            or plan.cost_usd != cost
            or plan.recovery_minutes != duration
        ):
            raise ModelRetry(
                f"Use exact simulated cost {cost} and recovery {duration}; budget is {ctx.deps.budget}. Choose a cheaper plan if needed."
            )
        if not any(
            remedies[a]["predicted_error_pct"] < 1 for a in plan.actions
        ):
            raise ModelRetry(
                "The plan must include a simulated remedy that reduces errors below 1%."
            )
        return plan

    async def delegate(agent, prompt, ctx):
        print(f"  -> {agent.name}", flush=True)
        with next_agent_span(
            metadata={
                "incident_id": ctx.deps.incident_id,
                "delegated_by": ctx.agent.name,
            }
        ):
            result = await agent.run(
                prompt,
                deps=ctx.deps,
                usage=ctx.usage,
                usage_limits=ctx.usage_limits,
            )
        return result.output

    @coordinator.tool
    async def investigate(ctx: RunContext[IncidentState]) -> list[Finding]:
        """Run independent checkout and payment investigations concurrently."""
        ctx.deps.findings = list(
            await asyncio.gather(
                delegate(
                    checkout,
                    "Investigate checkout. Retrieve all three specified records.",
                    ctx,
                ),
                delegate(
                    payments,
                    "Investigate downstream payments. Retrieve both specified records.",
                    ctx,
                ),
            )
        )
        ctx.deps.plan = ctx.deps.review = None
        return ctx.deps.findings

    @coordinator.tool
    async def design_remediation(ctx: RunContext[IncidentState]) -> Plan:
        """Delegate planning after investigation, incorporating previous review feedback."""
        if len(ctx.deps.findings) != 2:
            raise ModelRetry("Call investigate first.")
        payload = {
            "findings": [f.model_dump() for f in ctx.deps.findings],
            "budget_usd": ctx.deps.budget,
            "review_feedback": (
                ctx.deps.review.model_dump() if ctx.deps.review else None
            ),
        }
        ctx.deps.plan = await delegate(planner, json.dumps(payload), ctx)
        ctx.deps.review = None
        return ctx.deps.plan

    @coordinator.tool
    async def review_remediation(ctx: RunContext[IncidentState]) -> Review:
        """Ask an independent reviewer to verify the plan and its simulations."""
        if ctx.deps.plan is None:
            raise ModelRetry("Call design_remediation first.")
        payload = {
            "plan": ctx.deps.plan.model_dump(),
            "findings": [f.model_dump() for f in ctx.deps.findings],
            "simulations": ctx.deps.simulations,
            "evidence": {k: evidence[k] for k in ctx.deps.retrieved},
            "budget_usd": ctx.deps.budget,
        }
        ctx.deps.review = await delegate(reviewer, json.dumps(payload), ctx)
        return ctx.deps.review

    @coordinator.output_validator
    def validate_report(
        ctx: RunContext[IncidentState], report: IncidentReport
    ) -> IncidentReport:
        if (
            not ctx.deps.review
            or not ctx.deps.review.approved
            or ctx.deps.review.issues
        ):
            ctx.deps.retry_events.append(
                "report submitted before approved review"
            )
            raise ModelRetry(
                "Complete investigate, design_remediation, and an approved review_remediation before submitting the report."
            )
        if (
            report.incident_id != ctx.deps.incident_id
            or report.plan != ctx.deps.plan
        ):
            raise ModelRetry("Use the exact incident ID and reviewed plan.")
        if not set(report.evidence_ids) <= ctx.deps.retrieved:
            raise ModelRetry("Only cite retrieved evidence.")
        return report

    async def workflow():
        usage = RunUsage()
        result = await coordinator.run(
            f"Resolve incident {state.incident_id}. Remediation budget: ${args.budget}. Investigate, simulate, review, and produce a final incident report.",
            deps=state,
            usage=usage,
            usage_limits=UsageLimits(
                request_limit=40, tool_calls_limit=50, total_tokens_limit=60000
            ),
        )
        return result.output, usage

    if offline:
        workflow = observe(type="agent", name="incident-response-workflow")(
            workflow
        )

    print(
        f'Incident response demo | {"OFFLINE (scripted model)" if offline else args.model} | {state.incident_id}',
        flush=True,
    )
    if offline and not args.offline:
        print(
            "OPENAI_API_KEY is not set; running offline. Set it to use a real model.",
            flush=True,
        )
    report = None
    try:
        report, usage = asyncio.run(workflow())
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(
            report.model_dump_json(indent=2) + "\n"
        )
        print(report.model_dump_json(indent=2))
        print(
            f"\nModel requests: {usage.requests} | tool calls: {usage.tool_calls} | tokens: {usage.total_tokens}"
        )
        print(f"Retry events: {state.retry_events}")
    finally:
        provider.force_flush(timeout_millis=15000)
        flush_traces(timeout=15)
        spans = memory_exporter.get_finished_spans()
        args.output.mkdir(parents=True, exist_ok=True)
        trace_json = [
            {
                "name": s.name,
                "trace_id": format(s.context.trace_id, "032x"),
                "span_id": format(s.context.span_id, "016x"),
                "parent_span_id": (
                    format(s.parent.span_id, "016x") if s.parent else None
                ),
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes),
            }
            for s in spans
        ]
        (args.output / "spans.json").write_text(
            json.dumps(trace_json, indent=2, default=str) + "\n"
        )
        print(
            f"Captured {len(spans)} OpenTelemetry spans. Artifacts: {args.output.resolve()}"
        )
        provider.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled.")
