import os
import time
import asyncio
from crewai import Task, Crew, Agent

from deepeval.integrations.crewai import instrument_crewai

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

instrument_crewai(api_key=os.getenv("CONFIDENT_API_KEY"))


async def create_and_run_crew(crew_id, topic):
    """Create and run a single crew with the given topic"""
    # Define your agents with roles and goals
    consultant = Agent(
        role="Consultant",
        goal="Write clear, concise explanation.",
        backstory="An expert consultant with a keen eye for software trends.",
    )

    # Create tasks for your agents
    task = Task(
        description=f"Explain the latest trends in {topic}.",
        expected_output="A clear and concise explanation.",
        agent=consultant,
    )

    # Instantiate your crew
    crew = Crew(
        agents=[consultant],
        tasks=[task],
    )

    # Kickoff your crew
    result = await crew.kickoff_async()
    print(f"Crew {crew_id} ({topic}) result: {result}")
    return result


async def main():
    # Define topics for each crew
    topics = ["AI", "Machine Learning", "Data Science"]

    # Create tasks for all crews
    tasks = [
        create_and_run_crew(i + 1, topic) for i, topic in enumerate(topics)
    ]

    # Run all crews concurrently
    results = await asyncio.gather(*tasks)

    print(f"\nAll crews completed! Total results: {len(results)}")
    return results


# Run the async main function
if __name__ == "__main__":
    results = asyncio.run(main())
    time.sleep(7)  # Wait for traces to be posted to observatory
