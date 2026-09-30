"""Provenance capture for judge LLM calls.

A metric opens a capture scope around each judge call (``capture_judge_exchanges``);
model implementations that can see the provider's raw response call
``record_openai_exchange`` inside it. The metric then stores the resulting
``JudgeExchange`` next to its captured prompt, so an archived ``MetricData``
holds the HTTP entity body, logprobs and provider metadata, and says whether
the capture came from a live call or was replayed from the cache.
"""

from __future__ import annotations

import base64
import contextvars
import functools
import hashlib
import inspect
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional

from pydantic import BaseModel, Field

CaptureStatus = Literal["live", "replay"]

LIVE: CaptureStatus = "live"
REPLAY: CaptureStatus = "replay"


class JudgeExchange(BaseModel):
    status: CaptureStatus = LIVE
    provider: Optional[str] = None
    request_model: Optional[str] = None
    served_model: Optional[str] = None
    request_id: Optional[str] = None
    response_id: Optional[str] = None
    system_fingerprint: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    # Assistant message content, as a string from the parsed response JSON.
    response_content: Optional[str] = None
    # HTTP entity body (after any Content-Encoding is undone), decoded to text
    # with the response charset: the same definition as the pilot ledger's
    # `response_body`. Not a byte-level record; see the two fields below.
    raw_response_body: Optional[str] = None
    # The entity body's bytes, base64-encoded, and their SHA-256 digest.
    raw_response_body_base64: Optional[str] = None
    raw_response_body_sha256: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None
    http_status: Optional[int] = None
    captured_at: Optional[str] = None
    # Earlier attempts (retries) that reached the provider for this judge call
    # but were not the one the metric kept.
    discarded_attempts: int = 0


_scope: contextvars.ContextVar[Optional[List[JudgeExchange]]] = (
    contextvars.ContextVar("deepeval_judge_capture_scope", default=None)
)


@contextmanager
def capture_judge_exchanges() -> Iterator[List[JudgeExchange]]:
    """Collect exchanges recorded by the model during one judge call."""
    exchanges: List[JudgeExchange] = []
    token = _scope.set(exchanges)
    try:
        yield exchanges
    finally:
        _scope.reset(token)


def judge_capture_active() -> bool:
    return _scope.get() is not None


def kept_exchange(exchanges: List[JudgeExchange]) -> Optional[JudgeExchange]:
    """The last recorded attempt is the one whose result the metric used."""
    if not exchanges:
        return None
    kept = exchanges[-1]
    kept.discarded_attempts = len(exchanges) - 1
    return kept


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def record_openai_exchange(
    raw_response: Any, request_model: Optional[str], provider: str = "openai"
) -> None:
    """Record an OpenAI SDK ``with_raw_response`` result, before it is parsed.

    Safe to call outside a capture scope (it does nothing) and never raises.
    """
    exchanges = _scope.get()
    if exchanges is None:
        return
    try:
        http_response = raw_response.http_response
        body_bytes = http_response.content
        body = http_response.text
        exchange = JudgeExchange(
            provider=provider,
            request_model=request_model,
            request_id=http_response.headers.get("x-request-id"),
            raw_response_body=body,
            raw_response_body_base64=base64.b64encode(body_bytes).decode(),
            raw_response_body_sha256=hashlib.sha256(body_bytes).hexdigest(),
            http_status=http_response.status_code,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if isinstance(data, dict):
            exchange.served_model = data.get("model")
            exchange.response_id = data.get("id")
            exchange.system_fingerprint = data.get("system_fingerprint")
            exchange.usage = data.get("usage")
            choices = data.get("choices") or []
            if choices:
                choice = choices[0]
                exchange.finish_reason = choice.get("finish_reason")
                exchange.response_content = (choice.get("message") or {}).get(
                    "content"
                )
                exchange.logprobs = choice.get("logprobs")
        exchanges.append(exchange)
    except Exception:
        pass


def record_judge_call(
    metric: Any,
    prompt: Any,
    response: Any,
    exchanges: Optional[List[JudgeExchange]] = None,
) -> None:
    """Append one judge call to the metric's capture lists.

    ``judge_prompts``, ``judge_responses`` and ``judge_exchanges`` stay
    index-aligned. When the provider exchange was captured, the response is the
    message content from the captured response body rather than a rendering of
    the parsed object.
    """
    exchange = kept_exchange(exchanges or [])
    if exchange is not None and exchange.response_content is not None:
        response_text = exchange.response_content
    elif hasattr(response, "model_dump_json"):
        response_text = response.model_dump_json()
    else:
        response_text = str(response)
    if metric.judge_prompts is None:
        metric.judge_prompts = []
    if metric.judge_responses is None:
        metric.judge_responses = []
    if metric.judge_exchanges is None:
        metric.judge_exchanges = []
    metric.judge_prompts.append(str(prompt))
    metric.judge_responses.append(response_text)
    metric.judge_exchanges.append(
        exchange.model_dump() if exchange is not None else None
    )
    metric.judge_capture_status = LIVE


def reset_judge_capture(metric: Any) -> None:
    metric.judge_prompts = None
    metric.judge_responses = None
    metric.judge_exchanges = None
    metric.judge_capture_status = None


def load_replayed_capture(metric: Any, metric_data: Any) -> None:
    """Put a cache hit's (replay-marked) captures on the metric object."""
    metric.judge_prompts = metric_data.judge_prompts
    metric.judge_responses = metric_data.judge_responses
    metric.judge_exchanges = metric_data.judge_exchanges
    metric.judge_capture_status = metric_data.judge_capture_status


def mark_metric_data_replay(metric_data: Any) -> Any:
    """Return a copy of a cached ``MetricData`` that says no judge was called.

    Cache hits reuse captures from an earlier live call; without this marker
    they are indistinguishable from fresh ones.
    """
    exchanges = None
    if metric_data.judge_exchanges is not None:
        exchanges = [
            dict(exchange, status=REPLAY) if exchange is not None else None
            for exchange in metric_data.judge_exchanges
        ]
    return metric_data.model_copy(
        update={
            "judge_capture_status": REPLAY,
            "judge_exchanges": exchanges,
            "evaluation_cost": 0,
        },
        deep=True,
    )


def rebuild_chat_completion(exchange: Dict[str, Any]):
    """Rebuild the provider's ChatCompletion from an archived exchange.

    Lets a reader recompute a logprob-dependent score (e.g. GEval's weighted
    score) from the archive alone.
    """
    from openai.types.chat import ChatCompletion

    encoded = exchange.get("raw_response_body_base64")
    if encoded:
        return ChatCompletion.model_validate_json(base64.b64decode(encoded))
    return ChatCompletion.model_validate_json(exchange["raw_response_body"])


def provenance_mode_enabled() -> bool:
    from deepeval.config.settings import get_settings

    return bool(get_settings().DEEPEVAL_JUDGE_PROVENANCE)


# Attributes a metric fills in during measure() from judge output rather than
# from its configuration. Fingerprinting them would make a fresh instance
# (warm lookup) and a used one (cache write) disagree. GEval's generated steps
# are still compared by MetricConfiguration.evaluation_steps / criteria.
_DERIVED_ATTRIBUTES = frozenset({"evaluation_steps"})


def _derived_attributes(metric: Any) -> frozenset:
    derived = set(_DERIVED_ATTRIBUTES)
    # TaskCompletionMetric infers `task` from the trace unless it was given.
    if getattr(metric, "_is_task_provided", True) is False:
        derived.add("task")
    return frozenset(derived)


def _function_identity(fn: Any) -> Any:
    fn = inspect.unwrap(getattr(fn, "__func__", fn))
    try:
        source = inspect.getsource(fn)
    except (TypeError, OSError):
        code = getattr(fn, "__code__", None)
        source = code.co_code.hex() if code is not None else repr(fn)
    cells = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            cells.append(_canonical(cell.cell_contents, depth=1))
        except ValueError:  # empty cell
            cells.append(None)
    return {
        "name": getattr(fn, "__qualname__", None),
        "source": source,
        "closure": cells,
    }


def _class_identity(cls: type) -> Any:
    """Name and body of every class in ``cls``'s MRO.

    Two template classes defined in the same (unchanged) file get different
    identities, and so does a subclass that overrides one prompt method.
    """
    identity = []
    for klass in cls.__mro__:
        if klass is object:
            continue
        members = {}
        for name, value in sorted(vars(klass).items()):
            if name.startswith("__"):
                continue
            if isinstance(value, (staticmethod, classmethod)) or callable(
                value
            ):
                if inspect.isclass(value):
                    members[name] = f"{value.__module__}.{value.__qualname__}"
                else:
                    members[name] = _function_identity(value)
            else:
                members[name] = _canonical(value, depth=1)
        identity.append(
            {"class": f"{klass.__module__}.{klass.__qualname__}", "members": members}
        )
    return identity


def _canonical(value: Any, depth: int = 0) -> Any:
    """A stable, JSON-serialisable view of a configuration value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _canonical(value.value, depth)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(v, depth) for v in value]
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
        return items
    if isinstance(value, dict):
        return {
            str(k): _canonical(v, depth)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"), depth)
    if inspect.isclass(value):
        if depth > 0:
            return f"{value.__module__}.{value.__qualname__}"
        return _class_identity(value)
    if callable(getattr(value, "get_model_name", None)):
        # A judge model: its identity and sampling settings.
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.get_model_name(),
            "temperature": _canonical(getattr(value, "temperature", None), 1),
            "generation_kwargs": _canonical(
                getattr(value, "generation_kwargs", None), 1
            ),
        }
    # Anything else contributes only its type: never a repr with an address.
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def metric_configuration_snapshot(metric: Any) -> Dict[str, Any]:
    """The metric's prompt-affecting configuration.

    Every constructor parameter of the metric's class (and its bases) that is
    stored on the instance under the same name, e.g. GEval's ``criteria``,
    ``rubric``, ``top_logprobs``, ``evaluation_template`` and ``model``,
    minus attributes the metric derives during measure().
    """
    params = set()
    for klass in type(metric).__mro__:
        init = vars(klass).get("__init__")
        if init is None:
            continue
        try:
            signature = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, param in signature.parameters.items():
            if name != "self" and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                params.add(name)
    params -= _derived_attributes(metric)
    instance = vars(metric)
    return {
        name: _canonical(instance[name])
        for name in sorted(params)
        if name in instance
    }


def _template_files(metric: Any) -> List[str]:
    import deepeval.templates

    suffixes = (".py", ".txt", ".json", ".md", ".jinja", ".j2")
    dirs = [os.path.dirname(deepeval.templates.__file__)]
    try:
        dirs.append(os.path.dirname(inspect.getfile(type(metric))))
    except (TypeError, OSError):
        pass
    template = getattr(metric, "evaluation_template", None)
    paths = set()
    if inspect.isclass(template):
        try:
            paths.add(inspect.getfile(template))
        except (TypeError, OSError):
            pass
    for directory in dirs:
        for root, _, files in os.walk(directory):
            for name in files:
                if name.endswith(suffixes):
                    paths.add(os.path.join(root, name))
    return sorted(paths)


def template_fingerprint(metric: Any) -> str:
    """Hash everything a metric's judge prompts are built from.

    - the deepeval version;
    - the template files: the runtime bundle ``deepeval/templates`` (including
      the compiled ``templates.json`` prompts are rendered from), the metric's
      package directory, and the file defining a custom ``evaluation_template``;
    - the metric's configuration (``metric_configuration_snapshot``), which
      includes the selected ``evaluation_template`` class by name and source.

    It errs towards cache misses: a configuration value it cannot represent
    contributes its type only, and a metric that rewrites one of its
    constructor arguments during measure() will miss rather than go stale.
    Prompts are rendered from the bundle as loaded into memory: after editing
    ``templates.json`` in a running process, call
    ``deepeval.templates.resolver.clear_metric_template_cache()`` or the new
    fingerprint will not match the prompts actually sent.
    """
    from deepeval import __version__

    digest = hashlib.sha256()
    digest.update(f"deepeval=={__version__}\n".encode())
    for path in _template_files(metric):
        relative = path.rsplit(os.sep + "deepeval" + os.sep, 1)[-1]
        digest.update(relative.encode() + b"\0")
        with open(path, "rb") as f:
            digest.update(hashlib.sha256(f.read()).digest())
    digest.update(
        json.dumps(
            metric_configuration_snapshot(metric),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


def wrap_measure_entry_points(cls: type) -> None:
    """Reset a metric's captures when a public measure call starts.

    Only the outermost call resets, so ``measure()`` delegating to
    ``a_measure()`` (or a subclass calling ``super().measure()``) keeps one
    run's captures together, and a second ``measure()`` on the same instance
    starts from empty instead of appending to the first run's.
    """
    for name in ("measure", "a_measure"):
        fn = vars(cls).get(name)
        if fn is None or getattr(fn, "_resets_judge_capture", False):
            continue
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(self, *args, __fn=fn, **kwargs):
                _enter_measure(self)
                try:
                    return await __fn(self, *args, **kwargs)
                finally:
                    _exit_measure(self)

        else:

            @functools.wraps(fn)
            def wrapper(self, *args, __fn=fn, **kwargs):
                _enter_measure(self)
                try:
                    return __fn(self, *args, **kwargs)
                finally:
                    _exit_measure(self)

        wrapper._resets_judge_capture = True
        setattr(cls, name, wrapper)


def _enter_measure(metric: Any) -> None:
    depth = getattr(metric, "_judge_capture_depth", 0)
    if depth == 0:
        reset_judge_capture(metric)
    metric._judge_capture_depth = depth + 1


def _exit_measure(metric: Any) -> None:
    metric._judge_capture_depth = getattr(metric, "_judge_capture_depth", 1) - 1
