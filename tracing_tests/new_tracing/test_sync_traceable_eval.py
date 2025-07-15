from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.evaluate import AsyncConfig, DisplayConfig, evaluate
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    update_current_span,
    observe,
    RetrieverAttributes,
    LlmAttributes,
    trace_manager,
)

import openai
from openai import OpenAI
import os
from time import sleep, perf_counter
import random
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

trace_manager._daemon = False

#######################################################
## Example ############################################
#######################################################


@observe(type="llm", model="gpt-4o")
def generate_text(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides informative and accurate responses.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        generated_text = response.choices[0].message.content

        attributes = LlmAttributes(
            input=prompt,
            output=generated_text,
            input_token_count=response.usage.prompt_tokens,
            output_token_count=response.usage.completion_tokens,
        )
        update_current_span(attributes=attributes)
        sleep(random.uniform(0.5, 1.5))
        return generated_text
    except Exception as e:
        fallback_text = f"Generated text for: {prompt} (API error: {str(e)})"
        attributes = LlmAttributes(
            input=prompt,
            output=fallback_text,
            input_token_count=len(prompt.split()),
            output_token_count=len(fallback_text.split()),
        )
        update_current_span(attributes=attributes)
        return fallback_text


# Example of a retrieval node with embedded embedder
@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str, top_k: int = 3):
    try:
        # Generate embedding for the query
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002", input=query
        )
        query_embedding = embedding_response.data[0].embedding

        # Simulate document retrieval with realistic documents
        sample_documents = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
            "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.",
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way.",
            "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world, such as images and videos. It enables machines to identify objects, faces, and scenes in visual data.",
        ]

        # Simple similarity-based retrieval (in a real system, you'd use vector similarity)
        relevant_docs = sample_documents[:top_k]

        update_current_span(
            attributes=RetrieverAttributes(
                embedding_input=query,
                retrieval_context=relevant_docs,
            )
        )
        sleep(random.uniform(0.5, 1.5))
        return relevant_docs
    except Exception as e:
        fallback_docs = [
            f"Document 1 about {query}",
            f"Document 2 about {query}",
            f"Document 3 about {query}",
        ]
        update_current_span(
            attributes=RetrieverAttributes(
                embedding_input=query,
                retrieval_context=fallback_docs,
            )
        )
        return fallback_docs


@observe("CustomEmbedder")
def custom_embed(text: str, model: str = "text-embedding-ada-002"):
    try:
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        sleep(random.uniform(0.5, 1.5))
        return embedding
    except Exception as e:
        # Fallback to a simple embedding
        embedding = [0.1, 0.2, 0.3] * 50  # Simulate a longer embedding vector
        sleep(random.uniform(0.5, 1.5))
        return embedding


@observe("CustomRetriever", name="custom retriever")
def custom_retrieve(
    query: str, embedding_model: str = "text-embedding-ada-002"
):
    try:
        embedding = custom_embed(query, embedding_model)

        # Simulate a custom knowledge base with more specific documents
        custom_documents = [
            f"Specialized research document on {query}: This document contains detailed analysis and insights about {query} based on recent studies and expert opinions.",
            f"Technical report about {query}: Comprehensive technical analysis covering various aspects of {query} including implementation details and best practices.",
            f"Industry analysis on {query}: Market trends, competitive landscape, and future projections related to {query}.",
            f"Academic paper on {query}: Peer-reviewed research findings and theoretical frameworks related to {query}.",
            f"Case study about {query}: Real-world examples and practical applications of {query} in different contexts.",
        ]

        # Return top 2 most relevant documents
        documents = custom_documents[:2]
        sleep(random.uniform(0.5, 1.5))
        return documents
    except Exception as e:
        documents = [
            f"Custom doc 1 about {query}",
            f"Custom doc 2 about {query}",
        ]
        sleep(random.uniform(0.5, 1.5))
        return documents


@observe("CustomLLM")
def custom_generate(prompt: str, model: str = "gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized AI assistant that provides detailed, accurate, and well-structured responses. Focus on being helpful and informative.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.5,
        )
        custom_response = response.choices[0].message.content
        sleep(random.uniform(0.5, 1.5))
        return custom_response
    except Exception as e:
        fallback_response = (
            f"Custom response for: {prompt} (API error: {str(e)})"
        )
        sleep(random.uniform(0.5, 1.5))
        return fallback_response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
def custom_research_agent(query: str):
    try:
        if (
            random.random() < 0.8
        ):  # Higher success rate for more realistic behavior
            docs = custom_retrieve(query)
            analysis_prompt = (
                f"Based on the following documents, provide a comprehensive analysis of '{query}':\n\nDocuments:\n"
                + "\n\n".join(docs)
                + "\n\nAnalysis:"
            )
            analysis = custom_generate(analysis_prompt)
            sleep(random.uniform(0.5, 1.5))
            return analysis
        else:
            sleep(random.uniform(0.5, 1.5))
            return "Research information unavailable due to insufficient data or processing constraints."
    except Exception as e:
        sleep(random.uniform(0.5, 1.5))
        return f"Research error: {str(e)}"


@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
    metrics=[AnswerRelevancyMetric(), BiasMetric()],
)
def weather_agent(query: str):
    try:
        # Generate a realistic weather response using OpenAI
        weather_prompt = f"""You are a weather information assistant. The user is asking: "{query}"

Please provide a realistic weather response. If the query is about a specific location, provide weather information for that location. If it's a general weather question, provide helpful information about weather patterns, forecasting, or weather-related topics.

Keep your response informative but concise (2-3 sentences)."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful weather information assistant.",
                },
                {"role": "user", "content": weather_prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )

        weather_response = response.choices[0].message.content

        update_current_span(
            test_case=LLMTestCase(input=query, actual_output=weather_response)
        )
        sleep(random.uniform(0.5, 1.5))
        return weather_response
    except Exception as e:
        fallback_response = (
            "Weather information unavailable due to service interruption."
        )
        update_current_span(
            test_case=LLMTestCase(input=query, actual_output=fallback_response)
        )
        sleep(random.uniform(0.5, 1.5))
        return fallback_response


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
def research_agent(query: str):
    try:
        docs = retrieve_documents(query)
        research_prompt = f"""Based on the following retrieved documents, provide a comprehensive research analysis of '{query}':

Documents:
{chr(10).join([f"{i+1}. {doc}" for i, doc in enumerate(docs)])}

Please provide a well-structured analysis that synthesizes the information from these documents and addresses the user's query directly."""

        analysis = generate_text(research_prompt)
        sleep(random.uniform(0.5, 1.5))
        return analysis
    except Exception as e:
        fallback_analysis = (
            f"Research analysis unavailable due to processing error: {str(e)}"
        )
        sleep(random.uniform(0.5, 1.5))
        return fallback_analysis


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
    metrics=[AnswerRelevancyMetric(), BiasMetric()],
)
def meta_agent(input: str):
    try:
        # Gather information from all specialized agents
        weather_info = weather_agent(input)
        research_info = research_agent(input)
        custom_info = custom_research_agent(input)

        # Use OpenAI to synthesize a coherent final response
        synthesis_prompt = f"""You are a meta-agent that synthesizes information from multiple specialized agents. 

User Query: "{input}"

Information from different agents:
- Weather Agent: {weather_info}
- Research Agent: {research_info}
- Custom Research Agent: {custom_info}

Please provide a well-structured, coherent response that integrates all this information to answer the user's query. Make sure the response flows naturally and doesn't just list the different sources separately."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a meta-agent that synthesizes information from multiple specialized agents into coherent, helpful responses.",
                },
                {"role": "user", "content": synthesis_prompt},
            ],
            max_tokens=600,
            temperature=0.4,
        )

        final_response = response.choices[0].message.content

        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=final_response),
            metadata={"user_id": "11111", "date": "1/1/11"},
        )
        return final_response
    except Exception as e:
        # Fallback to simple concatenation if synthesis fails
        weather_info = weather_agent(input)
        research_info = research_agent(input)
        custom_info = custom_research_agent(input)
        final_response = f"""
        Weather: {weather_info}
        Research: {research_info}
        Custom Analysis: {custom_info}
        """
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=final_response),
            metadata={"user_id": "11111", "date": "1/1/11"},
        )
        return final_response


###################################v

from deepeval.dataset import Golden

goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
    Golden(input="Summarize the latest research on quantum computing."),
    Golden(input="Who won the FIFA World Cup in 2018?"),
    Golden(input="What are the health benefits of meditation?"),
    Golden(input="Explain the theory of relativity in simple terms."),
    Golden(input="What's the capital of Australia?"),
    Golden(
        input="How does machine learning differ from traditional programming?"
    ),
    Golden(input="Give me a brief history of the internet."),
    Golden(input="What are the main causes of climate change?"),
]


from deepeval import test_run
from deepeval.dataset import Golden

goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
]

from deepeval import test_run
from deepeval.dataset import Golden

goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
]

# # Run Async
# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
#     async_config=AsyncConfig(run_async=True),
#     display_config=DisplayConfig(show_indicator=True),
# )

# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
#     async_config=AsyncConfig(run_async=True),
#     display_config=DisplayConfig(show_indicator=False),
# )
# # Run Sync
# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
#     async_config=AsyncConfig(run_async=False),
#     display_config=DisplayConfig(show_indicator=True),
# )
# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
#     async_config=AsyncConfig(run_async=True),
#     display_config=DisplayConfig(show_indicator=True),
# )


# # Assert Test
# def test_meta_agent_0():
#     golden = Golden(input="What's the weather like in SF?")
#     assert_test(golden=golden, observed_callback=meta_agent, run_async=False)


# def test_meta_agent_1():
#     golden = Golden(input="What's the weather like in SF?")
#     assert_test(golden=golden, observed_callback=meta_agent, run_async=False)
