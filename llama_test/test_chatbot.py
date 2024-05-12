from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import set_global_handler, Settings
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Any, List
import asyncio

from deepeval.integrations.llama_index import DeepEvalToxicityEvaluator, LlamaIndexCallbackHandler
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import deepeval
 
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

# LLM
Settings.llm = OpenAI(model="gpt-4-turbo-preview")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Generate nodes from documents
documents = SimpleDirectoryReader("data").load_data()
node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(
    documents, show_progress=True
)

# Define embedding model
index = VectorStoreIndex(nodes)

# Build query engine
query_engine = index.as_query_engine(similarity_top_k=5)

# ###########################################################
# # test chatbot
# ###########################################################
# from llama_index.core.tools import FunctionTool
# from llama_index.core.agent import ReActAgent

# # define sample Tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two integers and returns the result integer"""
#     return a * b

# multiply_tool = FunctionTool.from_defaults(fn=multiply)

# # initialize llm
# llm = OpenAI(model="gpt-3.5-turbo-0613")

# # initialize ReAct agent
# input = "What is 2123 * 215123"
# agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
# output = agent.chat(input)
# test_case_2 = LLMTestCase(
#     input=input,
#     actual_output=str(output),
# )


user_inputs = [
    "what does your company do",
    "when where you incoporated",
    "what is your company name",
    "what are your products",
    "do you offer support",
    "what are your services"
]
for input in user_inputs:
    res = query_engine.query(input)
    
async def main():
    while True:
        user_input = input("Enter a query (or type 'exit' to finish): ")
        if user_input.lower() == 'exit':
            break

        res = query_engine.query(user_input)

        # Track with the desired asynchronous or synchronous mode
        await deepeval.track(
            event_name="single_query_event",
            model="gpt-4-turbo-preview",
            input=user_input,
            response=res.response,
            retrieval_context=None,
            completion_time=None,
            token_usage=None,
            token_cost=None,
            distinct_id=None,
            conversation_id=None,
            additional_data={},
            hyperparameters={},
            fail_silently=False,
            raise_expection=True,
            run_async=True  # Set to False to use synchronous behavior
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
