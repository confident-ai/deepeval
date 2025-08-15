from deepeval.tracing import observe, update_current_span, update_current_trace


@observe()
async def llm_app(query: str):
    # Add span-level metadata
    update_current_span(
        metadata={"source": "knowledge_base_1", "retrieved_documents": 3}
    )

    # Add trace-level metadata
    update_current_trace(
        metadata={
            "user_id": "user-456",
            "app_version": "1.2.3",
        }
    )


llm_app("Test Metadata")
