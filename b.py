from deepeval.prompt import Prompt

prompt = Prompt(alias="ApiPromptTest")
prompt.pull()

print(prompt.tools)

openai_tools = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": tool.mode == "STRICT",
        },
    }
    for tool in prompt.tools
]

print(openai_tools)



# from deepeval.prompt import Prompt
# from deepeval.prompt.api import (
#     PromptMessage,
#     ModelSettings,
#     ModelProvider,
#     OutputType,
#     ReasoningEffort,
#     Verbosity,
# )
# from pydantic import BaseModel

# class Obj(BaseModel):
#     name: str

# class ResponseSchema(BaseModel):
#     answer: str
#     confidence: float
#     obj: Obj

# prompt = Prompt(alias="NestedObject")

# prompt.push(
#     messages=[
#         PromptMessage(role="system", content="You are a helpful assistant."),
#     ],
#     model_settings=ModelSettings(
#         provider=ModelProvider.OPEN_AI,
#         name="gpt-4o",
#         temperature=0.7,
#         max_tokens=1000,
#         top_p=0.9,
#         frequency_penalty=0.1,
#         presence_penalty=0.1,
#         stop_sequence=["END"],
#         reasoning_effort=ReasoningEffort.MINIMAL,
#         verbosity=Verbosity.LOW,
#     ),
#     output_type=OutputType.SCHEMA,
#     output_schema=ResponseSchema,
# )
