from deepeval.tracing.tracing import Observer


from deepeval.tracing.tracing import current_span_context
from deepeval.tracing.types import RetrieverSpan, RetrieverAttributes


def patch_build_context_for_task():
    from crewai.memory.contextual.contextual_memory import ContextualMemory
    from crewai.task import Task

    original_build_context_for_task = ContextualMemory.build_context_for_task

    def patched_build_context_for_task(*args, **kwargs):
        observer_kwargs = {
            "observe_kwargs": {
                "span_type": "retriever",
                "embedder": "contextual",
            }
        }
        with Observer(
            func_name="build_context_for_task",
            span_type="retriever",
            **observer_kwargs
        ) as observer:
            embedding_input = "No input"
            if isinstance(args[1], Task):
                embedding_input = args[1].prompt()

            result = original_build_context_for_task(*args, **kwargs)

            retrieval_context = []
            if isinstance(result, str):
                retrieval_context = [result]

            current_span = current_span_context.get()
            current_span.set_attributes(
                RetrieverAttributes(
                    embedding_input=embedding_input,
                    retrieval_context=retrieval_context,
                )
            )

        return result

    ContextualMemory.build_context_for_task = patched_build_context_for_task
