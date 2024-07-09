import asyncio
from typing import List
from pydantic import BaseModel
from deepeval.models import GPTModel

model = GPTModel()

# test generate
##################################
# print(model.generate("Hi, what's your name?"))

# test a_generate
##################################
# async def test_a_generate():
#     print(await model.a_generate("Hi, what's your name?"))
#     print(await model.a_generate("Hi, what's your name?"))
# asyncio.run(test_a_generate())

# test generate_raw_response
##################################
# print(model.generate_raw_response("Hi, what's your name?"))

# test a_generate_raw_response
##################################
# async def test_a_generate_raw_response():
#     print(await model.a_generate_raw_response("Hi, what's your name?"))
#     print(await model.a_generate_raw_response("Hi, what's your name?"))
# asyncio.run(test_a_generate_raw_response())

# test generate JSON MODE
##################################
class RedTeamInput(BaseModel):
    input: str
    
class RedTeanModel(BaseModel):
    red_team_inputs: List[RedTeamInput]

response = model.generate("Give me a list of LLM red-teaming inputs", RedTeanModel)
print(response)

# test a_generate JSON MODE
##################################
class Text2SQL(BaseModel):
    input: str
    
class Text2SQLModel(BaseModel):
    text2sql_inputs: List[Text2SQL]

async def test_a_generate():
    print(await model.a_generate("Give me a list of LLM red-teaming inputs", Text2SQL))
    print(await model.a_generate("Give me a list of Text2SQL inputs", Text2SQLModel))
asyncio.run(test_a_generate())
