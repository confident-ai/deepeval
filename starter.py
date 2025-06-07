from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
from deepeval.integrations.llama_index.handler import my_event_handler
import time
import llama_index.core.instrumentation as instrument

import deepeval

deepeval.login_with_confident_api_key("jm2oYpcAJu/125pqmzAkrlTr2+iG+qJffjGPdaFcn2A=")

dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(my_event_handler)

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("./data").load_data()
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
async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)
    time.sleep(60)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())