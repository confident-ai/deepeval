import os
import llama_index.core.instrumentation as instrument
import asyncio
import time

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"

from deepeval.integrations.llama_index import ReActAgent
from deepeval.integrations.llama_index import instrument_llama_index
from llama_index.llms.openai import OpenAI

import deepeval

deepeval.login("<CONFIDENT_API_KEY>")
instrument_llama_index(instrument.get_dispatcher())

agent = ReActAgent(
    llm=OpenAI(model="gpt-4o-mini"),
    metric_collection="test_collection_1",
)


async def main():
    response = await agent.run("What is the capital of France?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(7)
