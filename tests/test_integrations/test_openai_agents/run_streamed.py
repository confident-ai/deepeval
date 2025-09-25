from weather_agent import weather_agent
from deepeval.openai_agents import Runner

async def run_streamed():
    run_streamed = Runner.run_streamed(
        weather_agent,
        "What's the weather in London?",
        metric_collection="test_collection_1",
        name="test_name_1",
        user_id="test_user_id_1",
        thread_id="test_thread_id_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_metadata_1"},
    )

    async for chunk in run_streamed.stream_events():
        continue