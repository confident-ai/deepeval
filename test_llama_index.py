import os
import asyncio
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import llama_index.core.instrumentation as instrument

import deepeval
from deepeval.integrations import LLamaIndexEventHandler
from deepeval.tracing.tracing import observe

deepeval.login_with_confident_api_key("<confident_api_key>")

os.environ["OPENAI_API_KEY"] = "<openai_api_key>"

dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(LLamaIndexEventHandler())

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("./examples/llama_index_sample_data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with both tools
agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
@observe()
async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(20) # wait for queue to be processed