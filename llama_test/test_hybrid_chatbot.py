from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import set_global_handler

from typing import Any
from openai import AsyncOpenAI
import asyncio
import os

from deepeval.tracing import Tracer, TraceType, LlmMetadata
from deepeval.integrations.llama_index import LlamaIndexCallbackHandler

########################################################
### Integration ########################################
########################################################


def deepeval_callback_handler(**eval_params: Any) -> BaseCallbackHandler:
    return LlamaIndexCallbackHandler(**eval_params)


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""
    if eval_mode == "deepeval":
        handler = deepeval_callback_handler(**eval_params)
        import llama_index.core

        llama_index.core.global_handler = handler


set_global_handler("deepeval")

###########################################################
# set up RAG pipeline
###########################################################


class RAGPipeline:
    def __init__(
        self,
        model_name="gpt-4-turbo-preview",
        top_k=5,
        chunk_size=200,
        chunk_overlap=20,
        min_similarity=0.5,
        data_dir="data",
    ):
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "OpenAI API key not found in environment variables."
            )

        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        documents = SimpleDirectoryReader(data_dir).load_data()
        node_parser = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(
            documents, show_progress=True
        )

        self.index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        self.retriever = self.index.as_retriever(
            similarity_top_k=top_k, similarity_cutoff=min_similarity
        )
        self.model_name = model_name

    def format_nodes(self, query):
        with Tracer(trace_type="Custom Type") as llama_wrapper_trace:
            nodes = self.retriever.retrieve(query)
            combined_nodes = "\n".join([node.get_content() for node in nodes])

            # set parameters
            # llama_wrapper_trace.set_parameters(combined_nodes)
            return combined_nodes

    async def generate_completion(self, prompt, context):
        with Tracer(trace_type=TraceType.LLM) as llm_trace:
            full_prompt = f"Context: {context}\n\nQuery: {prompt}\n\nResponse:"
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            output = response.choices[0].message.content

            # set parameters
            llm_trace.set_parameters(
                output=output, metadata=LlmMetadata(model="gpt-4-turbo-preview")
            )
            return output

    async def aquery(self, query_text):
        with Tracer(trace_type=TraceType.QUERY) as query_trace:
            context = self.format_nodes(query_text)
            response = await self.generate_completion(query_text, context)

            # set parameters and track event
            print("WHAT")
            query_trace.set_parameters(response)
            query_trace.track(
                input=query_text,
                response=response,
                model="gpt-4-turbo-preview",
            )
            return response


#############################################################
### test chatbot event tracking
#############################################################

user_inputs = [
    "what does your company do",
    # "when were you incorporated",
    # "what is your company name",
    # "what are your products",
    # "do you offer support",
    # "what are your services"
]


async def query_and_print(engine: RAGPipeline, query: str):
    print(query)
    res = await engine.aquery(query)
    print("end of " + str(query))


async def main():
    engine = RAGPipeline(data_dir="data")

    tasks = [query_and_print(engine, query) for query in user_inputs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
