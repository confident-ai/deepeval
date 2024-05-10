from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import set_global_handler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Any
from openai import AsyncOpenAI
import asyncio
import os

from deepeval.integrations.llama_index import LlamaIndexCallbackHandler

###########################################################
# Set up Llama-index Integration
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
# RAg Pipeline
###########################################################

class RAGPipeline:
    def __init__(self, model_name="gpt-4-turbo-preview", top_k=5, chunk_size=200, chunk_overlap=20, data_dir="data"):
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        
        # Initialize OpenAI client and embedding model
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Generate nodes and create the index
        documents = SimpleDirectoryReader(data_dir).load_data()
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        self.query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        self.model_name = model_name

    def retrieve_context(self, query):
        nodes = self.query_engine.query(query)
        return nodes.response

    async def generate_completion(self, prompt, context):
        full_prompt = f"Context: {context}\n\nQuery: {prompt}\n\nResponse:"
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content

    async def aquery(self, query_text):
        context = self.retrieve_context(query_text)
        return await self.generate_completion(query_text, context)

# ###########################################################
# # test chatbot
# ###########################################################

user_inputs = [
    "what does your company do",
    "when were you incorporated",
    "what is your company name",
    "what are your products",
    "do you offer support",
    "what are your services"
]

async def query_and_print(engine: RAGPipeline, query: str):
    print("start_query")
    
    # implement tracing tracking logic here

    res = await engine.aquery(query)

    # implement tracing tracking logic here

    if res:
        print(f"Result for '{query}': {res}")

async def main():
    engine = RAGPipeline(data_dir="data")

    tasks = [query_and_print(engine, query) for query in user_inputs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())