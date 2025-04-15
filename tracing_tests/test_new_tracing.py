from deepeval.tracing import (
    observe,
    update_current_span_attributes,
    update_current_span_test_case_parameters,
    LlmAttributes,
    RetrieverAttributes,
    ToolAttributes,
    AgentAttributes,
)
import random
from time import sleep

#######################################################
## Example ############################################
#######################################################


# Example with explicit attribute setting
@observe(type="llm", model="gpt-4o")
def generate_text(prompt: str):
    generated_text = f"Generated text for: {prompt}"
    attributes = LlmAttributes(
        input=prompt,
        output=generated_text,
        input_token_count=len(prompt.split()),
        output_token_count=len(generated_text.split()),
    )
    update_current_span_attributes(attributes)
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return generated_text


@observe
def embed_query(text: str):
    embedding = [0.1, 0.2, 0.3]
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return embedding


# Example of a retrieval node with embedded embedder
@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str, top_k: int = 3):
    embedding = embed_query(query)
    documents = [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
        f"Document 3 about {query}",
    ]
    update_current_span_attributes(
        RetrieverAttributes(
            embedding_input=query,
            retrieval_context=documents,
        )
    )
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return documents


# Simple example of using the observe decorator
@observe(type="tool")
def get_weather(city: str):
    """Get the weather for a city."""
    # Simulate API call
    weather = f"Sunny in {city}"

    # Create attributes
    attributes = ToolAttributes(
        input_parameters={"asdfsaf": city}, output=weather
    )

    # Set attributes using the helper function
    update_current_span_attributes(attributes)

    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))

    # The function returns just the weather result
    return weather


# First get the location
@observe(type="tool")
def get_location(query: str):
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return "San Francisco"  # Simulated location lookup


# Example of an agent that uses both retrieval and LLM
@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
)
def random_research_agent(user_query: str, testing: bool = False):
    documents = retrieve_documents(user_query, top_k=3)
    analysis = generate_text(user_query)
    # set_current_span_attributes(
    #     AgentAttributes(
    #         input=user_query,
    #         output=analysis,
    #     )
    # )
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return analysis


# Example of a complex agent with multiple tool uses
@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
    name="weather_research_agent",
)
def weather_research_agent(user_query: str):
    location = get_location(user_query)
    weather = get_weather(location)
    documents = retrieve_documents(f"{weather} in {location}", top_k=2)
    response = f"In {location}, it's currently {weather}. Additional context: {documents[0]}"
    update_current_span_attributes(
        AgentAttributes(
            input=user_query,
            output=response,
        )
    )
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return response


@observe(
    type="agent",
    agent_handoffs=["random_research_agent", "weather_research_agent"],
)
def supervisor_agent(user_query: str):
    research = random_research_agent(user_query)
    weather_research = weather_research_agent(user_query)

    update_current_span_attributes(
        AgentAttributes(
            input=user_query,
            output=research + weather_research,
        )
    )

    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))

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
def custom_embed(text: str, model: str = "custom-model"):
    embedding = [0.1, 0.2, 0.3]
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return embedding


@observe("CustomRetriever", name="custom retriever")
def custom_retrieve(query: str, embedding_model: str = "custom-model"):
    embedding = custom_embed(query, embedding_model)
    documents = [
        f"Custom doc 1 about {query}",
        f"Custom doc 2 about {query}",
    ]
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    # return documents


@observe("CustomLLM")
def custom_generate(prompt: str, model: str = "custom-model"):
    response = f"Custom response for: {prompt}"
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
def custom_research_agent(query: str):
    if random.random() < 0.5:
        docs = custom_retrieve(query)
        analysis = custom_generate(str(docs))
        # Add sleep of 1-3 seconds
        sleep(random.uniform(1, 3))
        return analysis
    else:
        # Add sleep of 1-3 seconds
        sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(type="agent", available_tools=["get_weather", "get_location"])
def weather_agent(query: str):
    if random.random() < 0.5:
        location = get_location(query)
        if random.random() < 0.5:
            weather = get_weather(location)
            # Add sleep of 1-3 seconds
            sleep(random.uniform(1, 3))
            return f"Weather in {location}: {weather}"
        else:
            # Add sleep of 1-3 seconds
            sleep(random.uniform(1, 3))
            return f"Weather in {location}"
    else:
        # Add sleep of 1-3 seconds
        sleep(random.uniform(1, 3))
        return "Weather information unavailable"


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
def research_agent(query: str):
    if random.random() < 0.5:
        docs = retrieve_documents(query)
        analysis = generate_text(str(docs))
        # Add sleep of 1-3 seconds
        sleep(random.uniform(1, 3))
        return analysis
    else:
        # Add sleep of 1-3 seconds
        sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
)
def meta_agent(query: str):
    # 50% probability of executing the function
    if random.random() < 0.5:
        weather_info = weather_agent(query)
    else:
        weather_info = "Weather information unavailable"

    if random.random() < 0.5:
        research_info = research_agent(query)
    else:
        research_info = "Research information unavailable"

    if random.random() < 0.5:
        custom_info = custom_research_agent(query)
    else:
        custom_info = "Custom research information unavailable"

    final_response = f"""
    Weather: {weather_info}
    Research: {research_info}
    Custom Analysis: {custom_info}
    """

    return final_response


@observe(type="retriever", embedder="custom-model")
def meta_agent_2(query: str):
    update_current_span_attributes(
        RetrieverAttributes(
            embedding_input=query,
            retrieval_context=[
                "Custom doc 1 about {query}",
                "Custom doc 2 about {query}",
            ],
        )
    )
    return


@observe(
    type="agent", agent_handoffs=["meta_agent_2"], metrics=["Answer Relevancy"]
)
def meta_agent_3():
    update_current_span_test_case_parameters(
        input="What's the weather like in San Francisco?",
        actual_output="I don't know man",
    )
    return meta_agent_2("What's the weather like in San Francisco?")


# if __name__ == "__main__":
#     for i in range(10):
#         result = meta_agent_3()
#         print(f"Final result: {result}")

#     sleep(10)
#     # # print(f"Final result: {result}")
#     # traces = trace_manager.get_all_traces_dict()
#     # # print(f"Traces: {traces}")
#     # trace_api = trace_manager.post_trace(traces[0])
#     # print(f"Trace API: {trace_api}")
