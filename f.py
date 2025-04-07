from deepeval.prompt import Prompt

prompt = Prompt(alias="asdfdasf")
prompt.pull(version="00.00.02")
prompt_to_llm = prompt.interpolate()
print(prompt_to_llm)
