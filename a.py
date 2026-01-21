# """
# Example script demonstrating how to use DeepEval's PromptOptimizer.
# """

# from openai import OpenAI
# from deepeval.optimizer import PromptOptimizer
# from deepeval.prompt import Prompt
# from deepeval.dataset import Golden
# from deepeval.metrics import AnswerRelevancyMetric

# # Initialize OpenAI client
# client = OpenAI()


# def model_callback(prompt: Prompt, golden: Golden) -> str:
#     """
#     Callback function that runs your LLM with the optimized prompt.
#     This is called during scoring to evaluate how well the prompt performs.
#     """
#     # Interpolate the prompt template with the golden's input
#     final_prompt = prompt.interpolate(query=golden.input)

#     # Call your LLM
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": final_prompt}],
#     )

#     return response.choices[0].message.content


# # Define your initial prompt template (intentionally bad for testing optimization)
# prompt = Prompt(
#     text_template="""idk maybe try to respond to this thing if u want lol

# {query}

# whatever:"""
# )

# # Define your evaluation dataset (goldens)
# goldens = [
#     Golden(
#         input="What is the capital of France?",
#         expected_output="Paris",
#     ),
#     Golden(
#         input="Who wrote Romeo and Juliet?",
#         expected_output="William Shakespeare",
#     ),
#     Golden(
#         input="What is the chemical symbol for gold?",
#         expected_output="Au",
#     ),
#     Golden(
#         input="In what year did World War II end?",
#         expected_output="1945",
#     ),
# ]

# # Define metrics to optimize for
# metrics = [AnswerRelevancyMetric(threshold=0.7)]

# from deepeval.optimizer.configs import DisplayConfig
# from deepeval.optimizer.algorithms import GEPA

# # Create the optimizer
# optimizer = PromptOptimizer(
#     model_callback=model_callback,
#     metrics=metrics,
#     optimizer_model="gpt-4o",  # Model used for rewriting prompts
#     display_config=DisplayConfig(announce_ties=True),
#     algorithm=GEPA(iterations=1),
# )

# # Run optimization
# optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)

# # Print results
# print("\n" + "=" * 60)
# print("OPTIMIZATION COMPLETE")
# print("=" * 60)
# print(f"\nOriginal prompt:\n{prompt.text_template}")
# print(f"\nOptimized prompt:\n{optimized_prompt.text_template}")


from deepeval.metrics import GEval

metric = GEval()
