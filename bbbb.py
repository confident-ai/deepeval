from deepeval.prompt import Prompt

prompt = Prompt(alias="First Prompt")
prompt.pull(version="00.00.03")
prompt_to_llm = prompt.interpolate(asdf="...")
print(prompt_to_llm)
