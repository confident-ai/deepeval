from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
import llama_index

# llama_index.set_global_handler("deepeval")
service_context = ServiceContext.from_defaults(chunk_size=500)
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)


def query(user_input):
    res = query_engine.query(user_input)
    # evaluator = ToxicityEvaluator()
    # result = evaluator.evaluate_response(query=user_input, response=res)
    return res.response
