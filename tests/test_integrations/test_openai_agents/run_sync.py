from weather_agent import weather_agent
from agents import Runner, add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor

add_trace_processor(DeepEvalTracingProcessor())


def run_sync():
    Runner.run_sync(
        weather_agent,
        "What's the weather in London?",
    )


# if __name__ == "__main__":
#     run_sync()
