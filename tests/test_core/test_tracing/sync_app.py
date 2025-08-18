from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    update_current_span,
    update_llm_span,
    update_retriever_span,
    observe,
)

from openai import OpenAI
import random
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
        update_llm_span(
            input_token_count=response.usage.prompt_tokens,
            output_token_count=response.usage.completion_tokens,
        )
        return generated_text
    except Exception as e:
        fallback_text = f"Generated text for: {prompt} (API error: {str(e)})"
        update_llm_span(
            input_token_count=len(prompt.split()),
            output_token_count=len(fallback_text.split()),
        )
        return fallback_text


# Example of a retrieval node with embedded embedder
@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str, top_k: int = 3):
    try:
        sample_documents = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
            "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.",
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way.",
            "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world, such as images and videos. It enables machines to identify objects, faces, and scenes in visual data.",
        ]
        relevant_docs = sample_documents[:top_k]
        update_retriever_span(
            top_k=top_k,
            chunk_size=5,
        )
        return relevant_docs
    except Exception:
        fallback_docs = [
            f"Document 1 about {query}",
            f"Document 2 about {query}",
            f"Document 3 about {query}",
        ]
        update_retriever_span(
            top_k=top_k,
            chunk_size=5,
        )
        return fallback_docs


@observe("CustomEmbedder")
def custom_embed(text: str, model: str = "text-embedding-ada-002"):
    try:
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        return embedding
    except Exception:
        embedding = [0.1, 0.2, 0.3] * 50
        return embedding


@observe("CustomRetriever", name="custom retriever")
def custom_retrieve(query: str):
    try:
        custom_documents = [
            f"Specialized research document on {query}: This document contains detailed analysis and insights about {query} based on recent studies and expert opinions.",
            f"Technical report about {query}: Comprehensive technical analysis covering various aspects of {query} including implementation details and best practices.",
            f"Industry analysis on {query}: Market trends, competitive landscape, and future projections related to {query}.",
            f"Academic paper on {query}: Peer-reviewed research findings and theoretical frameworks related to {query}.",
            f"Case study about {query}: Real-world examples and practical applications of {query} in different contexts.",
        ]
        documents = custom_documents[:2]
        return documents
    except Exception:
        documents = [
            f"Custom doc 1 about {query}",
            f"Custom doc 2 about {query}",
        ]
        update_retriever_span(
            top_k=2,
            chunk_size=5,
        )
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
        update_llm_span(
            input_token_count=response.usage.prompt_tokens,
            output_token_count=response.usage.completion_tokens,
        )
        return custom_response
    except Exception as e:
        fallback_response = (
            f"Custom response for: {prompt} (API error: {str(e)})"
        )
        update_llm_span(
            input_token_count=len(prompt.split()),
            output_token_count=len(fallback_response.split()),
        )
        return fallback_response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
def custom_research_agent(query: str):
    try:
        if random.random() < 0.8:
            docs = custom_retrieve(query)
            analysis_prompt = (
                f"Based on the following documents, provide a comprehensive analysis of '{query}':\n\nDocuments:\n"
                + "\n\n".join(docs)
                + "\n\nAnalysis:"
            )
            analysis = custom_generate(analysis_prompt)
            return analysis
        else:
            return "Research information unavailable due to insufficient data or processing constraints."
    except Exception as e:
        return f"Research error: {str(e)}"


@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
    metrics=[BiasMetric()],
)
def weather_agent(query: str):
    try:
        weather_prompt = f"""You are a weather information assistant. The user is asking: "{query}"
            Please provide a realistic weather response. If the query is about a specific location, 
            provide weather information for that location. If it's a general weather question, 
            provide helpful information about weather patterns, forecasting, or weather-related topics.
            Keep your response informative but concise (2-3 sentences).
        """

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
            input=query,
            output=weather_response,
        )
        return weather_response
    except:
        fallback_response = (
            "Weather information unavailable due to service interruption."
        )
        update_current_span(
            input=query,
            output=fallback_response,
        )
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
        update_current_span(
            input=query,
            output=analysis,
        )
        return analysis
    except Exception as e:
        fallback_analysis = (
            f"Research analysis unavailable due to processing error: {str(e)}"
        )
        update_current_span(
            input=query,
            output=fallback_analysis,
        )
        return fallback_analysis


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
    metrics=[AnswerRelevancyMetric()],
)
def meta_agent(input: str):
    try:
        weather_info = weather_agent(input)
        research_info = research_agent(input)
        custom_info = custom_research_agent(input)
        synthesis_prompt = f"""You are a meta-agent that synthesizes information from multiple specialized agents. 
            User Query: "{input}"
            Information from different agents:
            - Weather Agent: {weather_info}
            - Research Agent: {research_info}
            - Custom Research Agent: {custom_info}
            Please provide a well-structured, coherent response that integrates all this information to answer the user's query. Make sure the response flows naturally and doesn't just list the different sources separately.
        """
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
            input=input,
            output=final_response,
            metadata={"user_id": "11111", "date": "1/1/11"},
        )
        return final_response
    except Exception:
        weather_info = weather_agent(input)
        research_info = research_agent(input)
        custom_info = custom_research_agent(input)
        final_response = f"""
            Weather: {weather_info}
            Research: {research_info}
            Custom Analysis: {custom_info}
        """
        update_current_span(
            input=input,
            output=final_response,
            metadata={"user_id": "11111", "date": "1/1/11"},
        )
        return final_response
