# from pydantic_ai import RunContext
# import asyncio

# from deepeval.integrations.pydantic_ai import Agent

# joke_selection_agent = Agent(
#     'openai:gpt-4o',
#     system_prompt=(
#         'Use the `joke_factory` to generate some jokes, then choose the best. '
#         'You must return just a single joke.'
#     ),
#     trace_name="joke_selection_agent",
# )
# joke_generation_agent = Agent(
#     'openai:gpt-4o', output_type=list[str],
# )


# @joke_selection_agent.tool
# async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
#     r = await joke_generation_agent.run(
#         f'Please generate {count} jokes.',
#         usage=ctx.usage,
#     )
#     return r.output

# async def execute_agent():
#     result = await joke_selection_agent.run('Tell me a joke.', name="joke_selection_agent_2")
#     print(result.output)

# asyncio.run(execute_agent())
# result = joke_selection_agent.run_sync('Tell me a joke.')
# print(result.output)
# > Did you hear about the toothpaste scandal? They called it Colgate.


########################################################

# from dataclasses import dataclass
# import asyncio
# import httpx

# from pydantic_ai import RunContext
# from deepeval.integrations.pydantic_ai import Agent


# @dataclass
# class ClientAndKey:
#     http_client: httpx.AsyncClient
#     api_key: str


# joke_selection_agent = Agent(
#     'openai:gpt-4o',
#     deps_type=ClientAndKey,
#     system_prompt=(
#         'Use the `joke_factory` tool to generate some jokes on the given subject, '
#         'then choose the best. You must return just a single joke.'
#     ),
# )
# joke_generation_agent = Agent(
#     'openai:gpt-4o',
#     deps_type=ClientAndKey,
#     output_type=list[str],
#     system_prompt=(
#         'Use the "get_jokes" tool to get some jokes on the given subject, '
#         'then extract each joke into a list.'
#     ),
# )


# @joke_selection_agent.tool
# async def joke_factory(ctx: RunContext[ClientAndKey], count: int) -> list[str]:
#     r = await joke_generation_agent.run(
#         f'Please generate {count} jokes.',
#         deps=ctx.deps,
#         usage=ctx.usage,
#     )
#     return r.output


# @joke_generation_agent.tool
# async def get_jokes(ctx: RunContext[ClientAndKey], count: int) -> str:
#     response = await ctx.deps.http_client.get(
#         'https://example.com',
#         params={'count': count},
#         headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
#     )
#     response.raise_for_status()
#     return response.text


# async def main():
#     async with httpx.AsyncClient() as client:
#         deps = ClientAndKey(client, 'foobar')
#         result = await joke_selection_agent.run('Tell me a joke.', deps=deps)
#         print(result.output)
#         #> Did you hear about the toothpaste scandal? They called it Colgate.
#         # print(result.usage())
#         #> RunUsage(input_tokens=309, output_tokens=32, requests=4, tool_calls=2)

# asyncio.run(main())


from typing import Literal

from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import RunContext
from deepeval.integrations.pydantic_ai import Agent, instrument_pydantic_ai
from pydantic_ai.messages import ModelMessage

instrument_pydantic_ai()


class FlightDetails(BaseModel):
    flight_number: str


class Failed(BaseModel):
    """Unable to find a satisfactory choice."""


flight_search_agent = Agent[None, FlightDetails | Failed](
    "openai:gpt-4o",
    name="flight_search_agent",
    output_type=FlightDetails | Failed,  # type: ignore
    system_prompt=(
        'Use the "flight_search" tool to find a flight '
        "from the given origin to the given destination."
    ),
)


@flight_search_agent.tool
async def flight_search(
    ctx: RunContext[None], origin: str, destination: str
) -> FlightDetails | None:
    # in reality, this would call a flight search API or
    # use a browser to scrape a flight search website
    return FlightDetails(flight_number="AK456")


async def find_flight() -> FlightDetails | None:
    message_history: list[ModelMessage] | None = None
    for _ in range(3):
        prompt = Prompt.ask(
            "Where would you like to fly from and to?",
        )
        result = await flight_search_agent.run(
            prompt,
            message_history=message_history,
        )
        if isinstance(result.output, FlightDetails):
            return result.output
        else:
            message_history = result.all_messages(
                output_tool_return_content="Please try again."
            )


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal["A", "B", "C", "D", "E", "F"]


# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, SeatPreference | Failed](
    "openai:gpt-4o",
    name="seat_preference_agent",
    output_type=SeatPreference | Failed,  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        "Seats A and F are window seats. "
        "Row 1 is the front row and has extra leg room. "
        "Rows 14, and 20 also have extra leg room. "
    ),
)


async def find_seat() -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask("What seat would you like?")

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            print("Could not understand seat preference. Please try again.")
            message_history = result.all_messages()


async def main():

    opt_flight_details = await find_flight()
    if opt_flight_details is not None:
        print(f"Flight found: {opt_flight_details.flight_number}")
        # > Flight found: AK456
        seat_preference = await find_seat()
        print(f"Seat preference: {seat_preference}")
        # > Seat preference: row=1 seat='A'


# import asyncio
# asyncio.run(main())
