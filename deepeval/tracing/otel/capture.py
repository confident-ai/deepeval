"""Synchronous, ownership-aware ingestion of OTEL spans.

This layer accepts standard SDK spans (including confident-trace emissions).
It never sends spans through a batch exporter to discover evaluation context.
"""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Optional, Tuple
from weakref import WeakSet

from deepeval.contextvars import get_current_golden
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.types import BaseSpan, EvalSession, Trace, TraceSpanStatus
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.otel.utils import to_hex_string

_captures = WeakSet()
current_eval_owner = ContextVar("deepeval_otel_eval_owner", default=None)


@dataclass
class Owner:
    trace: Trace
    session: Optional[EvalSession]
    owns_trace: bool
    active: set = field(default_factory=set)
    spans: dict = field(default_factory=dict)


@dataclass
class Binding:
    owner: Optional[Owner]
    placeholder: Optional[BaseSpan] = None
    trace_context: Optional[Trace] = None
    cleanup: object = None
    transport: object = None
    api_key: Optional[str] = field(default=None, repr=False)


def capture_is_configured():
    return bool(_captures)


def promote_context(placeholder):
    for capture in list(_captures):
        if capture.promote(placeholder):
            return


def finish_capture(session, trace_uuid=None):
    """Close unfinished local spans at the evaluation boundary, never as success."""
    for capture in list(_captures):
        capture.finish(session, trace_uuid)


class SpanCapture:
    def __init__(self, exporter):
        self.exporter = exporter
        self.bindings: Dict[Tuple[int, int], Binding] = {}
        self.lock = RLock()
        _captures.add(self)

    @staticmethod
    def key(span):
        ctx = span.get_span_context()
        return ctx.trace_id, ctx.span_id

    def start(self, span, local, api_key=None):
        with self.lock:
            key = self.key(span)
            contextual_session = current_eval_owner.get()
            if (
                contextual_session is not None
                and contextual_session is not trace_manager.eval_session
            ):
                # Inherited context from a closed iterator must never join the
                # current run or escape through production OTLP.
                return
            if key in self.bindings:
                return
            parent_key = (key[0], span.parent.span_id) if span.parent else None
            parent = self.bindings.get(parent_key)
            if parent is not None:
                owner = parent.owner
            elif local:
                ctx = current_trace_context.get()
                explicit = ctx is not None and not ctx._is_otel_implicit
                target = (
                    trace_manager.get_trace_by_uuid(ctx.uuid) if ctx else None
                )
                owns_trace = target is None or not explicit
                if target is None:
                    target = trace_manager.start_new_trace(
                        trace_uuid=ctx.uuid
                        if ctx
                        else to_hex_string(key[0], 32),
                        _trace=ctx
                        if ctx is not None and ctx._is_otel_implicit
                        else None,
                    )
                if api_key is not None:
                    target.confident_api_key = api_key
                session = trace_manager.eval_session
                owner = Owner(
                    target,
                    session if session.is_evaluating else None,
                    owns_trace,
                )
                golden = get_current_golden()
                if owner.session is not None and golden is not None:
                    owner.session.trace_uuid_to_golden[target.uuid] = golden
                    if target.input is None:
                        target.input = golden.input
            else:
                owner = None
            if owner is None:
                live = current_span_context.get()
                if live is not None:
                    live.parent_uuid = (
                        to_hex_string(span.parent.span_id, 16)
                        if span.parent
                        else None
                    )
                self.bindings[key] = Binding(
                    None,
                    current_span_context.get(),
                    current_trace_context.get(),
                    transport=parent.transport if parent else None,
                    api_key=api_key,
                )
                return

            placeholder = current_span_context.get()
            sid = to_hex_string(key[1], 16)
            if placeholder is None or placeholder.uuid != sid:
                from deepeval.tracing import perf_epoch_bridge as peb

                placeholder = BaseSpan(
                    uuid=sid,
                    trace_uuid=owner.trace.uuid,
                    start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
                    status=TraceSpanStatus.IN_PROGRESS,
                )
            placeholder._otel_bridge = True
            placeholder.trace_uuid = owner.trace.uuid
            placeholder.parent_uuid = (
                parent.placeholder.uuid
                if parent and parent.placeholder
                else (span.attributes or {}).get("confident.span.parent_uuid")
            )
            if placeholder.parent_uuid is None and span.parent:
                placeholder.parent_uuid = to_hex_string(span.parent.span_id, 16)
            binding = Binding(owner, placeholder, current_trace_context.get())
            self.bindings[key] = binding
            owner.active.add(key)
            owner.spans[key] = placeholder
            trace_manager._otel_trace_ids.add(owner.trace.uuid)
            trace_manager.add_span(placeholder)
            if (
                parent is not None
                and parent.owner is owner
                and parent_key in owner.spans
            ):
                owner.spans[parent_key].children.append(placeholder)
            else:
                trace_manager.add_span_to_trace(placeholder)
            try:
                task = asyncio.current_task()
            except RuntimeError:
                task = None
            if task is not None:
                task_binding = trace_manager.task_bindings.setdefault(task, {})
                task_binding.setdefault("trace_uuid", owner.trace.uuid)
                task_binding.setdefault("root_span_uuid", placeholder.uuid)

    def promote(self, placeholder):
        """A native @observe nested in an OTEL entry opts that entry into capture."""
        with self.lock:
            match = next(
                (
                    b
                    for b in self.bindings.values()
                    if b.placeholder is placeholder and b.owner is None
                ),
                None,
            )
            if match is None or match.trace_context is None:
                return False
            target = trace_manager.start_new_trace(
                trace_uuid=match.trace_context.uuid, _trace=match.trace_context
            )
            target._is_otel_implicit = False
            target.confident_api_key = match.api_key
            session = trace_manager.eval_session
            owner = Owner(
                target, session if session.is_evaluating else None, True
            )
            for key, binding in self.bindings.items():
                if (
                    binding.owner is None
                    and binding.trace_context is target
                    and binding.placeholder is not None
                ):
                    binding.owner = owner
                    binding.placeholder._otel_bridge = True
                    owner.active.add(key)
                    owner.spans[key] = binding.placeholder
                    trace_manager.add_span(binding.placeholder)
                    trace_manager.add_span_to_trace(binding.placeholder)
            trace_manager._otel_trace_ids.add(target.uuid)
            return True

    def end(self, span):
        """Return True for captured/dropped spans; False selects production OTLP."""
        with self.lock:
            key = self.key(span)
            binding = self.bindings.get(key)
            # A processor attached after span start cannot safely attribute it.
            if binding is None:
                return True
            owner = binding.owner
            if owner is None:
                self.bindings.pop(key, None)
                return False
            if key not in owner.active:
                return True
            if (
                owner.session is not None
                and owner.session is not trace_manager.eval_session
            ):
                self._discard(owner)
                return True

            wrapper = self.exporter._convert_readable_span_to_base_span(
                span, _live_metrics=True
            )
            converted = wrapper.base_span
            old = binding.placeholder
            converted.trace_uuid = owner.trace.uuid
            converted.parent_uuid = old.parent_uuid
            converted.children = old.children
            # Python metric objects and explicit empty values never cross OTLP.
            for name in old.model_fields_set:
                if name in {
                    "uuid",
                    "trace_uuid",
                    "parent_uuid",
                    "children",
                    "start_time",
                    "end_time",
                    "status",
                    "error",
                }:
                    continue
                value = getattr(old, name)
                if name not in type(converted).model_fields:
                    continue
                if value is not None and (name != "name" or value):
                    setattr(converted, name, value)
            self._replace(owner.trace.root_spans, old, converted)
            owner.spans[key] = converted
            # Root completion applies final trace metadata after child updates.
            self.exporter._set_current_trace_attributes_from_base_span_wrapper(
                owner.trace, wrapper
            )
            if converted.status == TraceSpanStatus.ERRORED:
                owner.trace.status = TraceSpanStatus.ERRORED
            owner.active.remove(key)
            trace_manager.remove_span(old.uuid)
            if not owner.active:
                if owner.owns_trace:
                    trace_manager.end_trace(owner.trace.uuid)
                self._release(owner)
            return True

    @staticmethod
    def _replace(roots, old, new):
        for i, item in enumerate(roots):
            if item is old:
                roots[i] = new
                return True
            if SpanCapture._replace(item.children, old, new):
                return True
        return False

    def _release(self, owner):
        for key in owner.spans:
            binding = self.bindings.pop(key, None)
            if binding is not None and binding.cleanup is not None:
                binding.cleanup()

    def _discard(self, owner):
        # A closed evaluation must not be finalized under a later session's
        # routing policy (especially OFF, which would upload it as production).
        trace_manager._otel_pending_ends.discard(owner.trace.uuid)
        trace_manager._otel_trace_ids.discard(owner.trace.uuid)
        if trace_manager.active_traces.get(owner.trace.uuid) is owner.trace:
            trace_manager.active_traces.pop(owner.trace.uuid, None)
        trace_manager.traces[:] = [
            t for t in trace_manager.traces if t is not owner.trace
        ]
        for key in owner.active:
            trace_manager.remove_span(owner.spans[key].uuid)
        self._release(owner)

    def finish(self, session, trace_uuid=None):
        with self.lock:
            owners = {
                id(b.owner): b.owner
                for b in self.bindings.values()
                if b.owner is not None
            }
            for owner in owners.values():
                if owner.session is not session or (
                    trace_uuid and owner.trace.uuid != trace_uuid
                ):
                    continue
                owner.trace.status = TraceSpanStatus.ERRORED
                for key in list(owner.active):
                    pending = owner.spans[key]
                    pending.status = TraceSpanStatus.ERRORED
                    pending.error = "OTEL span did not finish before the evaluation scope closed"
                    pending.end_time = time.perf_counter()
                    trace_manager.remove_span(pending.uuid)
                owner.active.clear()
                if owner.owns_trace:
                    trace_manager.end_trace(owner.trace.uuid)
                self._release(owner)

    def drained(self):
        with self.lock:
            return not any(
                b.owner is not None and b.owner.active
                for b in self.bindings.values()
            )
