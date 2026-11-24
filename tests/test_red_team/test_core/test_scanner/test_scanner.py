

import os
import uuid

import pytest
from deepeval.metrics.utils import initialize_model
from deepeval.tracing.types import AgentSpan, LlmSpan, Trace, TraceSpanStatus

from deepteam.trace_scanner.trace_scanner import TraceScanner
from deepteam.vulnerabilities.bias.bias import Bias
from deepteam.vulnerabilities.bias.template import BiasTemplate

def load_trace_fixture(filename: str) -> Trace:
    test_dir = os.path.dirname(__file__)
    filepath = os.path.join(test_dir, "traces", f"{filename}.json")
    with open(filepath, "r") as f:
        return Trace.model_validate_json(f.read())


@pytest.fixture(scope="module")
def evaluation_model():
    model, using_native = initialize_model("gpt-4o")
    return model, using_native


@pytest.fixture
def bias_vulnerability(evaluation_model):
    model, _ = evaluation_model
    return Bias(
        evaluation_model=model,
        async_mode=False
    )


@pytest.fixture
def batch_evaluator_factory(evaluation_model):
    model, using_native = evaluation_model

    def _factory(limit: int) -> TraceScanner:
        return TraceScanner(
            model=model,
            template=BiasTemplate,
            limit=limit,
        )

    return _factory



class TestTraceScanner:
    
    def test_sync_materialized_bias_finding(self, bias_vulnerability):
        trace = load_trace_fixture("unsafe_bias_trace")

        findings = bias_vulnerability._assess_trace(trace)

        assert bias_vulnerability.vulnerable is True
        assert len(findings) >= 1
        assert any(f.vulnerabilityType == "gender" and f.outcome == "materialized" for f in findings)

    def test_guardrail_mitigation_no_root_breach(self, bias_vulnerability):
        trace = load_trace_fixture("guardrail_mitigated_trace")

        findings = bias_vulnerability._assess_trace(trace)

        assert len(findings) > 0
        assert all(f.outcome == "mitigated" for f in findings)
        
        assert bias_vulnerability.vulnerable is False

    @pytest.mark.asyncio
    async def test_async_deep_trace_empty_findings(self, bias_vulnerability):
        trace = load_trace_fixture("safe_trace")
        
        findings = await bias_vulnerability._a_assess_trace(trace)

        assert findings == []
        assert bias_vulnerability.vulnerable is False

    def test_io_collapsing_wrapper_edge_case(self, batch_evaluator_factory):
        trace_uuid = str(uuid.uuid4())
        agent_uuid = str(uuid.uuid4())
        llm_uuid = str(uuid.uuid4())
        
        shared_input = "Write a controversial opinion."
        shared_output = "Here is a highly controversial and biased opinion."

        llm_span = LlmSpan(
            uuid=llm_uuid,
            name="actual_generation",
            status=TraceSpanStatus.SUCCESS,
            trace_uuid=trace_uuid,
            parent_uuid=agent_uuid,
            start_time=1_700_000_000.0,
            end_time=1_700_000_001.0,
            input=shared_input,
            output=shared_output
        )

        agent_span = AgentSpan(
            uuid=agent_uuid,
            name="wrapper_agent",
            status=TraceSpanStatus.SUCCESS,
            trace_uuid=trace_uuid,
            parent_uuid=None,
            start_time=1_700_000_000.0,
            end_time=1_700_000_001.0,
            children=[llm_span],
            input=shared_input,
            output=shared_output,
            available_tools=[],
            agent_handoffs=[]
        )

        evaluator = batch_evaluator_factory(limit=40_000)
        extracted_parent = evaluator._extract_span_with_findings(agent_span, child_findings=[])

        assert extracted_parent.spanUuid == agent_uuid
        assert extracted_parent.input is None
        assert extracted_parent.output is None