from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
import llama_index

llama_index.set_global_handler("deepeval")

service_context = ServiceContext.from_defaults(chunk_size=1000)
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)


def query(user_input):
    return query_engine.query(user_input).response
