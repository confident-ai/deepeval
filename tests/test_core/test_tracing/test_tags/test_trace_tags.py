from deepeval.tracing import observe, update_current_trace
from tests.test_core.test_tracing.conftest import trace_test


@observe()
def trace_with_tags(data: str) -> str:
    update_current_trace(tags=["production", "v2", "ai-assistant"])
    return f"Tagged: {data}"


@observe()
def trace_with_env_tags(data: str, env: str = "dev") -> str:
    update_current_trace(tags=[env, "api", "traced"])
    return f"[{env}] {data}"


@observe()
def trace_with_feature_tags(query: str, features: list = None) -> str:
    tags = ["search"]
    if features:
        tags.extend(features)
    update_current_trace(tags=tags)
    return f"Search: {query}"


@observe()
def trace_with_name_and_tags(data: str) -> str:
    update_current_trace(
        name="custom_workflow", tags=["workflow", "custom", "test"]
    )
    return data


class TestTraceTags:

    @trace_test("tags/basic_tags_schema.json")
    def test_basic_tags(self):
        trace_with_tags("test")

    @trace_test("tags/env_tags_schema.json")
    def test_environment_tags(self):
        trace_with_env_tags("data", env="staging")

    @trace_test("tags/feature_tags_schema.json")
    def test_feature_tags(self):
        trace_with_feature_tags("AI query", features=["semantic", "reranking"])

    @trace_test("tags/name_and_tags_schema.json")
    def test_name_and_tags(self):
        trace_with_name_and_tags("test data")
