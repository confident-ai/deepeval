from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import tool
import asyncio

#############################################################
### Setup LLM ###############################################
#############################################################

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


get_word_length.invoke("abc")
tools = [get_word_length]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


async def chatbot(input):
    res = agent_executor.invoke(input)
    return res


#############################################################
### test chatbot event tracking
#############################################################

user_inputs = [
    {"input": "How many letters are in the word 'eudca'?"},
    {
        "input": "What is the length of the word 'supercalifragilisticexpialidocious'?"
    },
    {
        "input": "Can you tell me the number of characters in 'antidisestablishmentarianism'?"
    },
    {
        "input": "How many characters does the word 'floccinaucinihilipilification' have?"
    },
    {
        "input": "What is the length of the word 'pneumonoultramicroscopicsilicovolcanoconiosis'?"
    },
]


async def query_and_print(query: str):
    await chatbot(query)
    print("end of " + str(query))


async def main():
    tasks = [query_and_print(query) for query in user_inputs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
