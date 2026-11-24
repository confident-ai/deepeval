import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from deepeval.tracing.types import AgentSpan, LlmSpan, RetrieverSpan, ToolSpan, Trace, BaseSpan
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from .schema import BatchFinding, BatchFindingsList, SpanNode
from deepteam.utils import SPANS_CONTEXT_LIMIT
from deepteam.attacks.attack_simulator.utils import generate, a_generate


# Appended to a scanner's prompt only when prior detections from other
# vulnerability scanners are present, so the LLM suppresses duplicate
# attributions of an already-reported issue (see TraceScanner.previous_detections).
PREVIOUS_DETECTIONS_INSTRUCTION = """

PREVIOUSLY DETECTED ISSUES:
Some spans include a `previous_findings` field listing issues that OTHER vulnerability
scanners have already attributed to that span. If a finding you are about to report
describes the SAME underlying issue already present in that span's `previous_findings`,
OMIT it — it has already been counted. Only report issues that are genuinely distinct
from what is already listed there.
"""


@dataclass
class BatchContext:
    """Subtree-local accumulator for spans pending LLM evaluation.

    Each traversal subtree owns one; concurrent sibling traversals never
    share one, which keeps batch composition and size accounting correct
    across the `await` inside an async flush.
    """
    batch: List[SpanNode] = field(default_factory=list)
    size: int = 0


class TraceScanner:
    def __init__(
        self,
        model: DeepEvalBaseLLM,
        template: Any,
        limit: int = SPANS_CONTEXT_LIMIT,
        max_concurrent: int = 10,
        previous_detections: Optional[List[BatchFinding]] = None,
        output_schema: Any = BatchFindingsList,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.template = template
        self.limit = limit
        self.max_concurrent = max_concurrent
        self.output_schema = output_schema
        self._prior_by_span: Dict[str, List[BatchFinding]] = {}
        for finding in previous_detections or []:
            self._prior_by_span.setdefault(finding.spanUuid, []).append(finding)

        # Internal State
        self._findings: List[BatchFinding] = []
        self._findings_by_span: Dict[str, List[BatchFinding]] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None

    def process_trace(self, trace: Trace) -> List[BatchFinding]:
        self._reset_state()
        ctx = BatchContext()

        for span in trace.root_spans:
            self._merge_context(ctx, self._traverse_post_order(span))

        trace_node = self._extract_trace_root(trace)
        self._add_to_batch_and_check(ctx, trace_node)

        if ctx.batch:
            self._flush_batch(ctx)

        return self._results()

    async def a_process_trace(self, trace: Trace) -> List[BatchFinding]:
        self._reset_state()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        ctx = BatchContext()

        # Root spans are siblings -> independent -> scanned concurrently.
        child_ctxs = await asyncio.gather(
            *(self._a_traverse_post_order(span) for span in trace.root_spans)
        )
        for child_ctx in child_ctxs:
            await self._a_merge_context(ctx, child_ctx)

        trace_node = self._extract_trace_root(trace)
        await self._a_add_to_batch_and_check(ctx, trace_node)

        if ctx.batch:
            await self._a_flush_batch(ctx)

        return self._results()

    def _reset_state(self):
        self._findings = []
        self._findings_by_span = {}

    # ---------------------------------------------------------
    # SYNCHRONOUS TRAVERSAL
    # ---------------------------------------------------------

    def _traverse_post_order(self, span: BaseSpan) -> BatchContext:
        ctx = BatchContext()

        for child in span.children:
            self._merge_context(ctx, self._traverse_post_order(child))

        child_findings = self._get_child_findings(span.children)
        span_node = self._extract_span_with_findings(span, child_findings)

        self._add_to_batch_and_check(ctx, span_node)
        return ctx

    def _add_to_batch_and_check(self, ctx: BatchContext, node: SpanNode):
        # Dump to dict strictly excluding None to save tokens
        node_dict = node.model_dump(exclude_none=True)
        node_str = json.dumps(node_dict)
        node_size = len(node_str)

        if ctx.size + node_size > self.limit and ctx.batch:
            self._flush_batch(ctx)

        ctx.batch.append(node)
        ctx.size += node_size

        if ctx.size >= self.limit:
            self._flush_batch(ctx)

    def _merge_context(self, parent_ctx: BatchContext, child_ctx: BatchContext):
        """Fold a completed subtree's leftover (unflushed) nodes into the parent."""
        for node in child_ctx.batch:
            self._add_to_batch_and_check(parent_ctx, node)

    def _flush_batch(self, ctx: BatchContext):
        if not ctx.batch:
            return

        prompt = self._build_prompt(self._serialize_batch(ctx.batch))
        res = generate(prompt, self.output_schema, self.model)
        self._handle_result(res)
        ctx.batch = []
        ctx.size = 0

    # ---------------------------------------------------------
    # ASYNCHRONOUS TRAVERSAL
    # ---------------------------------------------------------

    async def _a_traverse_post_order(self, span: BaseSpan) -> BatchContext:
        ctx = BatchContext()

        if span.children:
            # Sibling subtrees are independent -> scan them concurrently.
            child_ctxs = await asyncio.gather(
                *(self._a_traverse_post_order(child) for child in span.children)
            )
            for child_ctx in child_ctxs:
                await self._a_merge_context(ctx, child_ctx)

        child_findings = self._get_child_findings(span.children)
        span_node = self._extract_span_with_findings(span, child_findings)

        await self._a_add_to_batch_and_check(ctx, span_node)
        return ctx

    async def _a_add_to_batch_and_check(self, ctx: BatchContext, node: SpanNode):
        node_dict = node.model_dump(exclude_none=True)
        node_str = json.dumps(node_dict)
        node_size = len(node_str)

        if ctx.size + node_size > self.limit and ctx.batch:
            await self._a_flush_batch(ctx)

        ctx.batch.append(node)
        ctx.size += node_size

        if ctx.size >= self.limit:
            await self._a_flush_batch(ctx)

    async def _a_merge_context(self, parent_ctx: BatchContext, child_ctx: BatchContext):
        for node in child_ctx.batch:
            await self._a_add_to_batch_and_check(parent_ctx, node)

    async def _a_flush_batch(self, ctx: BatchContext):
        if not ctx.batch:
            return

        prompt = self._build_prompt(self._serialize_batch(ctx.batch))

        # Bound the number of in-flight LLM calls across concurrent subtrees.
        async with self._semaphore:
            res = await a_generate(prompt, self.output_schema, self.model)
        self._handle_result(res)
        ctx.batch = []
        ctx.size = 0

    def _serialize_batch(self, batch: List[SpanNode]) -> str:
        batch_list = [node.model_dump(exclude_none=True) for node in batch]
        return json.dumps(batch_list, indent=2)

    def _build_prompt(self, batch_string: str) -> str:
        prompt = self.template.generate_trace_batch_evaluation(batch_data=batch_string)
        if self._prior_by_span:
            prompt += PREVIOUS_DETECTIONS_INSTRUCTION
        return prompt

    def _handle_result(self, res: Any):
        """Accumulate one batch's LLM result into scanner state."""
        self._store_findings(res.findings)

    def _results(self) -> Any:
        """Final value returned by process_trace / a_process_trace."""
        return self._findings

    # ---------------------------------------------------------
    # UTILITIES & EXTRACTION
    # ---------------------------------------------------------

    def _store_findings(self, findings: List[BatchFinding]):
        for finding in findings:
            if finding.spanUuid not in self._findings_by_span:
                self._findings_by_span[finding.spanUuid] = []

            # Filter out any older findings for this exact vulnerability type
            # because the newer finding from a higher-level batch is the final authority.
            self._findings_by_span[finding.spanUuid] = [
                f
                for f in self._findings_by_span[finding.spanUuid]
                if not (
                    f.vulnerability == finding.vulnerability
                    and f.vulnerabilityType == finding.vulnerabilityType
                )
            ]
            self._findings_by_span[finding.spanUuid].append(finding)

            self._findings = [
                f
                for f in self._findings
                if not (
                    f.spanUuid == finding.spanUuid
                    and f.vulnerability == finding.vulnerability
                    and f.vulnerabilityType == finding.vulnerabilityType
                )
            ]
            self._findings.append(finding)

    def _get_child_findings(self, children: List[BaseSpan]) -> List[BatchFinding]:
        findings = []
        for child in children:
            if child.uuid in self._findings_by_span:
                findings.extend(self._findings_by_span[child.uuid])
        return findings

    def _collapse_io(self, parent_input: Any, parent_output: Any, children: List[BaseSpan]) -> tuple[Any, Any]:
        """
        Compares parent I/O with children I/O. If an exact match is found,
        returns None for that field to avoid redundant LLM evaluation / attribution.
        """
        collapsed_input = parent_input
        collapsed_output = parent_output

        for child in children:
            if collapsed_input is not None and collapsed_input == child.input:
                collapsed_input = None
            if collapsed_output is not None and collapsed_output == child.output:
                collapsed_output = None

            # Early exit if both are already collapsed
            if collapsed_input is None and collapsed_output is None:
                break

        return collapsed_input, collapsed_output

    def _extract_span_with_findings(self, span: BaseSpan, child_findings: List[BatchFinding]) -> SpanNode:
        """Strips useless metadata, keeps I/O, tools, and attaches child findings."""

        c_input, c_output = self._collapse_io(span.input, span.output, span.children)

        extracted = SpanNode(
            spanUuid=span.uuid,
            parentUuid=span.parent_uuid,
            type=span.__class__.__name__,
            name=span.name,
            status=span.status.value if span.status else None,
            input=c_input,
            output=c_output,
            error=span.error,
            context=span.context,
            retrieval_context=span.retrieval_context,
            expected_output=span.expected_output,
            tools_called=span.tools_called,
            child_findings=child_findings if child_findings else None,
            previous_findings=self._prior_by_span.get(span.uuid) or None,
        )

        # 2. Map Subclass-Specific Critical Fields
        if isinstance(span, LlmSpan):
            extracted.model = span.model

        elif isinstance(span, AgentSpan):
            extracted.available_tools = span.available_tools
            extracted.agent_handoffs = span.agent_handoffs

        elif isinstance(span, RetrieverSpan):
            extracted.embedder = span.embedder

        elif isinstance(span, ToolSpan):
            extracted.description = span.description

        return extracted

    def _extract_trace_root(self, trace: Trace) -> SpanNode:
        """Extracts the top-level trace I/O and any root-level findings."""
        root_findings = self._get_child_findings(trace.root_spans)

        c_input, c_output = self._collapse_io(trace.input, trace.output, trace.root_spans)

        return SpanNode(
            spanUuid=trace.uuid,
            parentUuid=None,
            type="TraceRoot",
            name=getattr(trace, "name", None),
            status=trace.status.value if trace.status else None,
            input=c_input,
            output=c_output,
            context=getattr(trace, "context", None),
            retrieval_context=getattr(trace, "retrieval_context", None),
            expected_output=getattr(trace, "expected_output", None),
            tools_called=[tc.model_dump(exclude_none=True) for tc in trace.tools_called] if getattr(trace, "tools_called", None) else None,
            child_findings=root_findings if root_findings else None,
            previous_findings=self._prior_by_span.get(trace.uuid) or None,
        )
