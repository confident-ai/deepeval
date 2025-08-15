from deepeval.tracing import observe, update_llm_span


@observe(type="llm", model="gpt-4.1")
def generate_response(prompt):
    output = "Generated response to: " + prompt
    update_llm_span(
        input_token_count=10,
        output_token_count=25,
        cost_per_input_token=0.01,
        cost_per_output_token=0.01,
    )
    return output


generate_response("What is the capital of France?")

############################################

from deepeval.tracing import observe, update_retriever_span


@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query):
    fetched_documents = ["doc1", "doc2"]
    update_retriever_span(
        embedder="text-embedding-ada-002",
        chunk_size=10,
        top_k=5,
    )
    return fetched_documents


retrieve_documents("What is the capital of France?")
