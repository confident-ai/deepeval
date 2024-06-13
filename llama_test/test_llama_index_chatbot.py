from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import set_global_handler, Settings
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Any, List
import asyncio

from deepeval.tracing import Tracer, TraceType
from deepeval.integrations.llama_index import LlamaIndexCallbackHandler
import deepeval.tracing

###########################################################
# set up integration
###########################################################


# set llama index global handler
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
# test chatbot
###########################################################

Settings.llm = OpenAI(model="gpt-4-turbo-preview")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Generate nodes from documents
documents = SimpleDirectoryReader("data").load_data()
node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

# Define embedding model
index = VectorStoreIndex(nodes)


async def chatbot(input):
    with Tracer(trace_type="Chatbot") as chatbot_trace:
        # LLM

        # Build query engine
        query_engine = index.as_query_engine(similarity_top_k=5)
        res = query_engine.query(input).response

        chatbot_trace.set_parameters(
            output=res,
        )

        chatbot_trace.track(
            input=input,
            response=res,
            model="gpt-4-turbo-preview",
        )

        return res


#############################################################
### test chatbot event tracking
#############################################################

user_inputs = [
    "what does your company do",
    "when were you incorporated",
    # "what is your company name",
    # "what are your products",
    # "do you offer support",
    # "what are your services"
]


async def query_and_print(query: str):
    res = await chatbot(query)
    print("end of " + str(query))


async def main():
    tasks = [query_and_print(query) for query in user_inputs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
