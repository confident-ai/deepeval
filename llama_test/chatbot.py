from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, set_global_handler
from deepeval.integrations.llama_index import DeepEvalToxicityEvaluator

set_global_handler("deepeval")
# service_context = ServiceContext.from_defaults(chunk_size=500)
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)


def query(user_input):
    res = query_engine.query(user_input)
    # evaluator = DeepEvalToxicityEvaluator()
    # result = evaluator.evaluate_response(query=user_input, response=res)
    # print(result)
    return res.response
