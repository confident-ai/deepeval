"""
Streaming OpenAI Agent
Complexity: MEDIUM - Tests streaming execution with tool calls
"""

from agents import Agent, function_tool, ModelSettings


@function_tool
def get_company_info(symbol: str) -> str:
    """Get company information for a ticker symbol."""
    info = {
        "AAPL": "Apple Inc. - Technology company",
        "GOOGL": "Alphabet Inc. - Technology company",
        "MSFT": "Microsoft Corporation - Technology company",
    }
    return info.get(symbol.upper(), f"Company info not available for {symbol}")


agent = Agent(
    name="StreamingAgent",
    instructions="""You are a helpful assistant. 
    If asked for company info, use the tool. 
    If asked a general question, write a short poem about it to generate many tokens.""",
    model="gpt-4o",
    tools=[get_company_info],
    model_settings=ModelSettings(temperature=0.0),
)
