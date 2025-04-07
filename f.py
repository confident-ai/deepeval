from deepeval.prompt import Prompt

prompt = Prompt(alias="asdfsafdasfdasfasd")
prompt.pull(version="00.00.01")
prompt_to_llm = prompt.interpolate(name="...")
print(prompt_to_llm)
