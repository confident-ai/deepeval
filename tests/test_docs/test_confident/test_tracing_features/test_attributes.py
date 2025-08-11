from deepeval.tracing import observe, update_current_span, RetrieverAttributes
 
@observe(type="custom")
def outer_function():
 
    @observe(type="retriever")
    def inner_function():
 
        # Here, update_current_span() will update the Retriever span
        update_current_span(
            attribtues=RetrieverAttributes(
                embedding_input=query, 
                retrieval_context=fetched_documents
            )
        )