import os
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
import time
 
import deepeval
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import Golden
from deepeval.evaluate import dataset, test_run
 
# Don't forget to setup tracing
instrument_llama_index(instrument.get_dispatcher())
 
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

answer_relevancy_metric = AnswerRelevancyMetric()
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[answer_relevancy_metric]
)

goldens = [
    Golden(input="How do I perform 7 * 8?"),
]

def main():
    for golden in dataset(goldens=goldens):
        # Create an async function that properly awaits the agent.run()
        async def run_agent(input_text):
            return await agent.run(input_text)
        
        task = asyncio.create_task(run_agent(golden.input))
        test_run.append(task)


if __name__ == "__main__":
    main()
    time.sleep(7)