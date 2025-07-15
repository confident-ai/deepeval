from tracing_tests.new_tracing.test_async_traceable_eval import (
    meta_agent as async_meta_agent,
)
from tracing_tests.new_tracing.test_sync_traceable_eval import (
    meta_agent as sync_meta_agent,
)
from deepeval.evaluate import dataset, test_run, AsyncConfig
from deepeval.dataset import Golden
import asyncio


goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
    Golden(input="How tall is Mount Everest?"),
    # Golden(input="Who painted the Mona Lisa?"),
    # Golden(input="What's the population of Japan?"),
    # Golden(input="How do vaccines work?"),
    # Golden(input="Recommend a good sci-fi movie."),
]

# for golden in dataset(goldens=goldens, async_config=AsyncConfig(run_async=False)):
#     sync_meta_agent(golden.input)

for golden in dataset(alias="Expanded QA Dataset"):
    task = asyncio.create_task(async_meta_agent(golden.input))
    test_run.append(task)
