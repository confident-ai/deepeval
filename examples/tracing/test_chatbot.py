from deepeval.tracing import trace, TraceType
import openai


class Chatbot:
    def __init__(self):
        pass

    @trace(type=TraceType.LLM, name="OpenAI", model="gpt-4")
    def llm(self, input):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": input},
            ],
        )
        return response.choices[0].message.content

    @trace(
        type=TraceType.EMBEDDING,
        name="Embedding",
        model="text-embedding-ada-002",
    )
    def get_embedding(self, input):
        response = openai.Embedding.create(
            input=input, model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]

    @trace(type=TraceType.RETRIEVER, name="Retriever")
    def retriever(self, input=input):
        embedding = self.get_embedding(input)

        # Replace this with an actual vector search that uses embedding
        list_of_retrieved_nodes = ["Retrieval Node 1", "Retrieval Node 2"]
        return list_of_retrieved_nodes

    @trace(type=TraceType.TOOL, name="Search")
    def search(self, input):
        # Replace this with an actual function that searches the web
        title_of_the_top_search_results = "Search Result: " + input
        return title_of_the_top_search_results

    @trace(type=TraceType.TOOL, name="Format")
    def format(self, retrieval_nodes, input):
        prompt = "You are a helpful assistant, based on the following information: \n"
        for node in retrieval_nodes:
            prompt += node + "\n"
        prompt += "Generate an unbiased response for " + input + "."
        return prompt

    @trace(type=TraceType.AGENT, name="Chatbot")
    def query(self, user_input=input):
        top_result_title = self.search(user_input)
        retrieval_results = self.retriever(top_result_title)
        prompt = self.format(retrieval_results, top_result_title)
        return self.llm(prompt)


import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.evaluator import assert_test

chatbot = Chatbot()


def test_factual_consistency():
    context = [
        "Be a natural-born citizen of the United States.",
        "Be at least 35 years old.",
        "Have been a resident of the United States for 14 years.",
    ]
    input = "What are the requimrents to be president?"

    metric = FactualConsistencyMetric(minimum_score=0.8)
    test_case = LLMTestCase(
        input=input,
        actual_output=chatbot.query(user_input=input),
        context=context,
    )
    assert_test(test_case, [metric])
