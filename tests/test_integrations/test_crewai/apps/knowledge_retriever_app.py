"""
tests/test_integrations/test_crewai/apps/knowledge_retriever_app.py
A crew configured with knowledge sources to test retrieval spans.
"""

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.string_knowledge_source import (
    StringKnowledgeSource,
)


def get_knowledge_app():
    content = "The secret launch code is ALPHA-ZULU-99."
    string_source = StringKnowledgeSource(content=content)

    llm = LLM(model="gpt-4o-mini", temperature=0)

    agent = Agent(
        role="Security Analyst",
        goal="Retrieve secret information",
        backstory="You are an authorized analyst. You do not have the codes memorized; you must always look them up in the knowledge base.",
        llm=llm,
        verbose=True,
    )

    task = Task(
        description="What is the launch code? You MUST search the knowledge base for 'launch code' to find the answer.",
        expected_output="The exact launch code found in the knowledge base.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        knowledge_sources=[string_source],
        verbose=True,
    )

    return crew
