import pytest

from deepeval.tracing import observe, trace_manager
from deepeval.tracing.context import current_span_context, current_trace_context


@pytest.fixture(autouse=True)
def clean_trace_state():
    trace_manager.clear_traces()
    trace_manager.tracing_enabled = False
    current_span_context.set(None)
    current_trace_context.set(None)
    yield
    trace_manager.clear_traces()
    trace_manager.tracing_enabled = True
    current_span_context.set(None)
    current_trace_context.set(None)


@observe(type="agent")
def app_that_drops_trace():
    current_trace_context.drop()
    return "done"


@observe(type="agent")
def app_that_does_not_drop():
    return "done"


def test_drop_trace_sets_flag():
    """Calling current_trace_context.drop() sets trace.drop = True."""
    app_that_drops_trace()

    assert len(trace_manager.traces) == 1
    assert trace_manager.traces[0].drop is True


def test_trace_not_dropped_by_default():
    """Traces are not dropped by default."""
    app_that_does_not_drop()

    assert len(trace_manager.traces) == 1
    assert trace_manager.traces[0].drop is False


@observe(type="agent")
def app_drop_then_update():
    current_trace_context.drop()
    from deepeval.tracing import update_current_trace

    update_current_trace(name="updated_name")
    return "done"


def test_drop_persists_after_update_current_trace():
    """Once dropped, updating the trace does not undo the drop."""
    app_drop_then_update()

    trace = trace_manager.traces[0]
    assert trace.drop is True
    assert trace.name == "updated_name"


@observe(type="agent")
def app_with_dropped_span():
    @observe(type="tool")
    def kept_span():
        return "kept"

    @observe(type="tool")
    def dropped_span():
        current_span_context.drop()
        return "dropped"

    kept_span()
    dropped_span()
    return "done"


def test_drop_span_sets_flag():
    """Calling current_span_context.drop() sets span.drop = True."""
    app_with_dropped_span()

    trace = trace_manager.traces[0]
    root = trace.root_spans[0]
    children = root.children

    assert len(children) == 2
    dropped = [c for c in children if c.drop is True]
    kept = [c for c in children if c.drop is False]
    assert len(dropped) == 1
    assert len(kept) == 1
    assert dropped[0].name == "dropped_span"
    assert kept[0].name == "kept_span"


def test_dropped_span_excluded_from_trace_api():
    """Dropped spans are excluded when converting to the API payload."""
    app_with_dropped_span()

    trace = trace_manager.traces[0]
    trace_api = trace_manager.create_trace_api(trace)

    all_span_names = set()
    for span_list in [
        trace_api.base_spans or [],
        trace_api.agent_spans or [],
        trace_api.llm_spans or [],
        trace_api.retriever_spans or [],
        trace_api.tool_spans or [],
    ]:
        for span in span_list:
            all_span_names.add(span.name)

    assert "kept_span" in all_span_names
    assert "dropped_span" not in all_span_names


@observe(type="agent")
def app_drop_span_not_trace():
    @observe(type="tool")
    def child():
        current_span_context.drop()
        return "x"

    child()
    return "done"


def test_drop_span_does_not_drop_trace():
    """Dropping a span should not affect the parent trace's drop flag."""
    app_drop_span_not_trace()

    trace = trace_manager.traces[0]
    assert trace.drop is False
    assert trace.root_spans[0].children[0].drop is True


@observe(type="agent")
def app_drop_trace_not_span():
    current_trace_context.drop()
    return "done"


def test_drop_trace_does_not_set_span_drop():
    """Dropping a trace does not set drop on its spans."""
    app_drop_trace_not_span()

    trace = trace_manager.traces[0]
    assert trace.drop is True
    assert trace.root_spans[0].drop is False
