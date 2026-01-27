import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import (
    StringKnowledgeSource,
)

from deepeval.integrations.crewai import instrument_crewai
from tests.test_integrations.utils import assert_trace_json, generate_trace_json

# instrument_crewai()

# Create a knowledge source
content = "Users name is John. He is 30 years old and lives in San Francisco."
string_source = StringKnowledgeSource(content=content)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gpt-4o-mini", temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="You are a master at understanding people and their preferences.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[
        string_source
    ],  # Enable knowledge by adding the sources here
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "knowledge_retriever.json")


@assert_trace_json(json_path)
def test_knowledge_retriever():
    crew.kickoff(
        inputs={"question": "What city does John live in and how old is he?"}
    )
