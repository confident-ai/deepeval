from crewai import Task, Crew
from deepeval.integrations.crewai.agent import Agent
import os
from deepeval.integrations.crewai import instrumentator
from crewai.tools.base_tool import tool
import time
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "<your-api-key>"
instrumentator(api_key="<your-api-key>")

# Define a simple tool
@tool("greet")
def greet_tool(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

# Define another tool
@tool("add")
def add_tool(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

# Create two agents, each with memory and tools
my_knowledge = StringKnowledgeSource(content="Alice is a VIP customer. Always greet her warmly!")

agent1 = Agent(
    role="Greeter",
    goal="Greet users in a friendly way",
    backstory="A friendly AI who loves to make people feel welcome.",
    tools=[greet_tool],
    memory=True,  # Enable memory for this agent
    knowledge_sources=[my_knowledge],
    metric_collection="test_collection_1",
)

agent2 = Agent(
    role="Adder",
    goal="Add numbers provided by the user",
    backstory="A helpful assistant who enjoys solving math problems.",
    tools=[add_tool],
    memory=True,  # Enable memory for this agent
    metric_collection="test_collection_1",
)

# Define tasks for each agent
task1 = Task(
    description="Greet Alice using the greet tool.",
    expected_output="A friendly greeting to Alice.",
    agent=agent1,
)

task2 = Task(
    description="Add 3 and 5 using the add tool.",
    expected_output="The sum of 3 and 5.",
    agent=agent2,
)

# Create the crew with both agents and tasks
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Enable shared memory for the crew
)

# Run the crew
if __name__ == "__main__":
    results = crew.kickoff()
    time.sleep(7) # Wait for traces to be posted to observatory
    # print(results)
