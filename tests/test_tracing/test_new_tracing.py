from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
    LlmAttributes,
    RetrieverAttributes,
    ToolAttributes,
    AgentAttributes,
    trace_manager,
)
import random
from deepeval.metrics import AnswerRelevancyMetric
from asyncio import sleep

from deepeval.tracing.types import Feedback

#######################################################
## Example ############################################
#######################################################


# Example with explicit attribute setting
@observe(type="llm", model="gpt-4o")
async def generate_text(prompt: str):
    generated_text = f"Generated text for: {prompt}"
    attributes = LlmAttributes(
        input=prompt,
        output=generated_text,
        input_token_count=len(prompt.split()),
        output_token_count=len(generated_text.split()),
    )
    update_current_span(attributes=attributes)
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return generated_text


async def embed_query(text: str):
    embedding = [0.1, 0.2, 0.3]
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return embedding


# Example of a retrieval node with embedded embedder
@observe(type="retriever", embedder="text-embedding-ada-002")
async def retrieve_documents(query: str, top_k: int = 3):
    embedding = await embed_query(query)
    documents = [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
        f"Document 3 about {query}",
    ]
    update_current_span(
        attributes=RetrieverAttributes(
            embedding_input=query,
            retrieval_context=documents,
        )
    )
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return documents


# Simple example of using the observe decorator
@observe(type="tool")
async def get_weather(city: str):
    """Get the weather for a city."""
    # Simulate API call
    weather = f"Sunny in {city}"

    # Create attributes
    attributes = ToolAttributes(
        input_parameters={"asdfsaf": city}, output=weather
    )

    # Set attributes using the helper function
    update_current_span(attributes=attributes)

    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))

    # The function returns just the weather result
    return weather


# First get the location
@observe(type="tool")
async def get_location(query: str):
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return "San Francisco"  # Simulated location lookup


# Example of an agent that uses both retrieval and LLM
@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
)
async def random_research_agent(user_query: str, testing: bool = False):
    documents = await retrieve_documents(user_query, top_k=3)
    analysis = await generate_text(user_query)
    # set_current_span_attributes(
    #     AgentAttributes(
    #         input=user_query,
    #         output=analysis,
    #     )
    # )
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return analysis


# Example of a complex agent with multiple tool uses
@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
    name="weather_research_agent",
)
async def weather_research_agent(user_query: str):
    location = await get_location(user_query)
    weather = await get_weather(location)
    documents = await retrieve_documents(f"{weather} in {location}", top_k=2)
    response = f"In {location}, it's currently {weather}. Additional context: {documents[0]}"
    update_current_span(
        attributes=AgentAttributes(
            input=user_query,
            output=response,
        )
    )
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return response


@observe(
    type="agent",
    agent_handoffs=["random_research_agent", "weather_research_agent"],
)
async def supervisor_agent(user_query: str):
    research = await random_research_agent(user_query)
    weather_research = await weather_research_agent(user_query)
    update_current_span(
        attributes=AgentAttributes(
            input=user_query,
            output=research + weather_research,
        )
    )

    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))

    return research + weather_research


# # Example usage
# if __name__ == "__main__":
#     # Call the research agent
#     research = supervisor_agent("What's the weather like in San Francisco?")
#     print(f"Research result: {research}")

#     # # Call the complex weather research agent
#     # weather_research = weather_research_agent("What's the weather like?")
#     # print(f"Weather research: {weather_research}")

#     # Get all traces
#     traces = trace_manager.get_all_traces_dict()
#     print(f"Traces: {traces}")


@observe("CustomEmbedder")
async def custom_embed(text: str, model: str = "custom-model"):
    embedding = [0.1, 0.2, 0.3]
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return embedding


@observe("CustomRetriever", name="custom retriever")
async def custom_retrieve(query: str, embedding_model: str = "custom-model"):
    embedding = await custom_embed(query, embedding_model)
    documents = [
        f"Custom doc 1 about {query}",
        f"Custom doc 2 about {query}",
    ]
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    # return documents


@observe("CustomLLM")
async def custom_generate(prompt: str, model: str = "custom-model"):
    # print(final_response)
    update_current_span(metadata={"user_id": "11111", "date": "1/1/11"})
    response = f"Custom response for: {prompt}"
    # Add sleep of 1-3 seconds
    await sleep(random.uniform(1, 3))
    return response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
async def custom_research_agent(query: str):
    # print(final_response)
    update_current_span(metadata={"user_id": "11111", "date": "1/1/11"})
    if random.random() < 0.5:
        docs = await custom_retrieve(query)
        analysis = await custom_generate(str(docs))
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return analysis
    else:
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(type="agent", available_tools=["get_weather", "get_location"])
async def weather_agent(query: str):
    # print(final_response)
    update_current_span(metadata={"user_id": "11111", "date": "1/1/11"})
    if random.random() < 0.5:
        location = await get_location(query)
        if random.random() < 0.5:
            weather = await get_weather(location)
            # Add sleep of 1-3 seconds
            await sleep(random.uniform(1, 3))
            return f"Weather in {location}: {weather}"
        else:
            # Add sleep of 1-3 seconds
            await sleep(random.uniform(1, 3))
            return f"Weather in {location}"
    else:
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return "Weather information unavailable"


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
async def research_agent(query: str):
    # print(final_response)
    update_current_span(metadata={"user_id": "11111", "date": "1/1/11"})
    if random.random() < 0.5:
        docs = await retrieve_documents(query)
        analysis = await generate_text(str(docs))
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return analysis
    else:
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
    metric_collection="My Metrics",
)
async def meta_agent(query: str):
    # print(query)
    # raise ValueError("Test Error")
    # 50% probability of executing the function
    # weather_info = await weather_agent(query)
    # research_info = await research_agent(query)
    # custom_info = await custom_research_agent(query)
    # final_response = f"""
    # Weather: {weather_info}
    # Research: {research_info}
    # Custom Analysis: {custom_info}
    # """
    # print(final_response)
    update_current_span(
        # metadata={"user_id": "11111", "date": "1/1/11"},
        test_case=LLMTestCase(
            input="What is this again?",
            actual_output="this is a latte",
            expected_output="this is a mocha",
            retrieval_context=["I love coffee"],
            context=["I love coffee"],
        ),
        # feedback=Feedback(
        #     rating=1,
        #     expected_output="this is a mocha",
        #     explanation="The actual output is not the expected output",
        # ),
    )
    update_current_trace(
        name="ok",
        # metadata={"input": "input"},
        # thread_id="131324ljihfsadiuyip",
        # user_id="111",
        # feedback=Feedback(
        #     rating=5,te
        #     expected_output="Testing again",
        #     explanation="The actual output is not the expected output",
        # ),
    )

    # return LLMTestCase(input="..", actual_output=final_response)


from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics import DAGMetric, GEval

# geval_metric = GEval(
#     name="Persuasiveness",
#     criteria="Determine how persuasive the `actual output` is to getting a user booking in a call.",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
# )

# conciseness_node = BinaryJudgementNode(
#     criteria="Does the actual output contain less than or equal to 4 sentences?",
#     children=[
#         VerdictNode(verdict=False, score=0),
#         VerdictNode(verdict=True, child=geval_metric),
#     ],
# )

# # create the DAG
# dag = DeepAcyclicGraph(root_nodes=[conciseness_node])
# metric = DAGMetric(dag=dag, name="Persuasiveness")


###################################

import asyncio
import re


def mask_function(data):
    if type(data) is str:
        censored_data = re.sub("Elon", "[REDACTED NAME]", data)
        return censored_data


trace_manager.configure(mask=mask_function, environment="production")


# # Gather multiple traceable tasks
async def run_parallel_examples():
    tasks = [
        meta_agent("How tall is Mount Everest?"),
        meta_agent("What's the capital of Brazil?"),
        meta_agent("Who won the last World Cup?"),
        meta_agent("Explain quantum entanglement."),
        # meta_agent("What's the latest iPhone model?"),
        # meta_agent("How do I cook a perfect steak?"),
        # meta_agent("Tell me a joke about robots."),
        # meta_agent("What causes lightning?"),
        # meta_agent("Who painted the Mona Lisa?"),
        # meta_agent("What's the population of Japan?"),
        # meta_agent("How do vaccines work?"),
        # meta_agent("Recommend a good sci-fi movie."),
    ]
    await asyncio.gather(*tasks)


# Run it
asyncio.run(run_parallel_examples())


# @observe()
# async def run_customizations_agent(input_items, context=None):
#     @observe()
#     async def run_customizations_agent_again(input_items, context=None):
#         return LLMTestCase(input="input_items", actual_output="context")

#     result = await run_customizations_agent_again(input_items, context)
#     return AnswerRelevancyMetric()


# asyncio.run(
#     run_customizations_agent(input_items=["item1", "item2"], context="context")
# )
