from deepeval.integrations.llama_index import instrument_llama_index
from deepeval.integrations.llama_index import FunctionAgent
import llama_index.core.instrumentation as instrument
from llama_index.llms.openai import OpenAI
import asyncio
 
from deepeval.metrics import TaskCompletionMetric
from deepeval.evaluate import dataset, test_run
from deepeval.dataset import Golden
 
instrument_llama_index(instrument.get_dispatcher())
 
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

agent = FunctionAgent(  
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[TaskCompletionMetric()]
)

goldens = [
    Golden(input="What's 7 * 8?"),
    Golden(input="What's 7 * 6?")
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