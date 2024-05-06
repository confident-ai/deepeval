from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import set_global_handler, Settings
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Any

from deepeval.integrations.llama_index import DeepEvalToxicityEvaluator, LlamaIndexCallbackHandler
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

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

user_input = "what does your company do"
res = query_engine.query(user_input)
test_case = LLMTestCase(
    input=user_input,
    actual_output=res.response,
)

# evaluator = DeepEvalToxicityEvaluator()
# result = evaluator.evaluate_response(query=user_input, response=res)
# print(result)

metric = ToxicityMetric(threshold=0.5)
evaluate([test_case], [metric])

