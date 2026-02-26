# from deepeval.tracing import (
#     observe,
#     update_current_span,
#     update_current_trace,
#     update_llm_span,
# )
# import asyncio
# from deepeval.prompt import Prompt

# prompt = Prompt(alias="Message Prompt")
# prompt.pull()


# @observe(
#     type="llm",
#     model="gemini-2.5-flash",
#     cost_per_input_token=0.0000003,
#     cost_per_output_token=0.0000025,
#     metric_collection="default123",
# )
# async def meta_agent(query: str):
#     update_current_span(
#         input=query,
#         output=query,
#         expected_output=query,
#         retrieval_context=[query],
#         metadata={"query": query},
#         name="meta_agent",
#     )
#     update_llm_span(
#         input_token_count=10,
#         output_token_count=10,
#         # prompt=prompt,
#     )
#     update_current_trace(
#         name="meta_agent",
#         thread_id="fetch 11",
#         input=query,
#         output=query,
#         expected_output=query,
#         retrieval_context=[query],
#         user_id="clickhouse user",
#         tags=["test", "example"],
#         metadata={"query": {"query": query}},
#         test_case_id="ffb10651-010e-4d37-9947-b75324c31971",
#     )
#     return query


# async def run_parallel_examples():
#     tasks = [
#         meta_agent("How tall is Mount Everest?"),
#         # meta_agent("What's the capital of Brazil?"),
#         # meta_agent("Who won the last World Cup?"),
#         # meta_agent("Explain quantum entanglement."),
#         # meta_agent("What's the latest iPhone model?"),
#         # meta_agent("How do I cook a perfect steak?"),
#         # meta_agent("Tell me a joke about robots."),
#         # meta_agent("What causes lightning?"),
#         # meta_agent("Who painted the Mona Lisa?"),
#         # meta_agent("What's the population of Japan?"),
#         # meta_agent("How do vaccines work?"),
#         # meta_agent("Recommend a good sci-fi movie."),
#     ]
#     await asyncio.gather(*tasks)


# asyncio.run(run_parallel_examples())

# # # ############################################################
# # # ############################################################
# # # ############################################################

# # from deepeval import evaluate
# # from deepeval.metrics import AnswerRelevancyMetric
# # from deepeval.test_case import LLMTestCase

# # evaluate(
# #     test_cases=[
# #         LLMTestCase(
# #             input="How tall is Mount Everest?",
# #             actual_output="Mount Everest is 8,848 meters tall.",
# #             expected_output="Mount Everest is 8,848 meters tall.",
# #         ),
# #     ],
# #     metrics=[AnswerRelevancyMetric()],
# # )

# # ############################################################
# # ############################################################
# # ############################################################

# # from deepeval import evaluate
# # from deepeval.metrics import TurnRelevancyMetric
# # from deepeval.test_case import ConversationalTestCase, Turn

# # evaluate(
# #     test_cases=[
# #         ConversationalTestCase(
# #             turns=[
# #                 Turn(
# #                     role="user",
# #                     content="Hello, how are you?",
# #                 ),
# #                 Turn(
# #                     role="assistant",
# #                     content="I'm doing well, thanks for asking! How can I help you today?",
# #                 ),
# #                 Turn(
# #                     role="user",
# #                     content="Can you explain what answer relevancy means?",
# #                 ),
# #                 Turn(
# #                     role="assistant",
# #                     content=(
# #                         "Answer relevancy measures how well a response directly addresses "
# #                         "the user's question, focusing on usefulness and alignment with intent."
# #                     ),
# #                 ),
# #             ],
# #         ),
# #     ],
# #     metrics=[TurnRelevancyMetric()],
# # )

# ######


# # from deepeval.tracing import observe, update_current_trace
# # from deepeval.metrics import AnswerRelevancyMetric
# # from deepeval.dataset import EvaluationDataset
# # from openai import OpenAI


# # @observe()
# # def llm_app(query: str) -> str:

# #     @observe()
# #     def retriever(query: str) -> list[str]:
# #         chunks = ["List", "of", "text", "chunks"]
# #         update_current_trace(retrieval_context=chunks)
# #         return chunks

# #     @observe(type="llm")
# #     def generator(query: str, text_chunks: list[str]) -> str:
# #         res = (
# #             OpenAI()
# #             .chat.completions.create(
# #                 model="gpt-4o", messages=[{"role": "user", "content": query}]
# #             )
# #             .choices[0]
# #             .message.content
# #         )
# #         update_current_trace(input=query, output=res)
# #         return res

# #     return generator(query, retriever(query))


# # dataset = EvaluationDataset()
# # dataset.pull(alias="Playground Dataset")

# # for golden in dataset.evals_iterator(metrics=[AnswerRelevancyMetric()]):
# #     llm_app(golden.input)

#########################################

from pydantic import BaseModel
from deepeval.tracing import observe, trace
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from time import sleep
import json

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@observe()
def llm_app(query: str) -> str:
    print("in llm_app")
    print("done processing query")
    return {"response": f"My Received query: {query}"}


@observe()
def llm_app_yield(query: str):
    print("in llm_app_yield")
    yield {"request": query}

    yield {"response": f"Received query yield: {query}"}

    yield {"metadata": {"yield query_length": len(query)}}

    print("done processing query")
    return {"response": f"My Received query with yield: {query}"}


@app.post("/query-yield")
def run_query_yield(payload: QueryRequest):
    print("Received query: ", payload.query)
    gen = llm_app_yield(payload.query)
    def stream_wrapper():
        # We wrap the generator to ensure it yields strings for the HTTP response
        for item in gen:
            yield json.dumps(item) + "\n"

    # The endpoint returns IMMEDIATELY after creating this object
    # This is where @observe() usually loses track of the 'output'
    return StreamingResponse(stream_wrapper(), media_type="application/x-ndjson")


@app.post("/query")
def run_query(payload: QueryRequest):
    print("Received query: ", payload.query)
    gen = llm_app(payload.query)
    print("Final return:", gen)
    return {"status": "ok", "result": gen}


if __name__ == "__main__":
    print("Running app directly")
    gen = llm_app("What is the capital of France?")
    ret = None

    print("Final return:", gen)

#########################################

# from deepeval.tracing import observe, update_retriever_span, update_current_trace
# from deepeval.metrics import AnswerRelevancyMetric

# from deepeval.dataset import EvaluationDataset, Golden


# @observe(type="retriever", name="Retriever")
# def retrieve(input):
#     update_retriever_span(
#         embedder="openai-embedder",
#         top_k=10,
#         chunk_size=20
#     )
#     return ["This is some context"]

# @observe(
#     type="tool",
#     name="Tool Span"
# )
# def tool_called(input):
#     retrieve(input)
#     return {
#         "output": "yes"
#     }

# @observe(
#     type="llm",
#     name="LLM Call",
#     metrics=[AnswerRelevancyMetric(threshold=0.8)],
# )
# def my_llm(input):
#     tool_called(input)
#     update_current_trace(
#         name="New Trace"
#     )
#     return f"Response: {input}"


# dataset = EvaluationDataset(
#     goldens=[Golden(input="hello")]
# )
# # dataset.pull(alias="Dataset for testing")

# for golden in dataset.evals_iterator(identifier="Some name"):
#     my_llm(golden.input)

# from time import sleep

# sleep(10)
