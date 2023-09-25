# Define your completion protocol
import openai
from deepeval.dataset import EvaluationDataset
from deepeval.metrics.factual_consistency import FactualConsistencyMetric

ds = EvaluationDataset.from_csv(
    "review-test.csv",
    query_column="query",
    expected_output_column="expected_output",
)
print(ds.sample())


def generate_chatgpt_output(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "assistant",
                "content": "The customer success phone line is 1200-231-231 and the customer success state is in Austin.",
            },
            {"role": "user", "content": query},
        ],
    )
    expected_output = response.choices[0].message.content
    return expected_output


factual_consistency_metric = FactualConsistencyMetric()

ds.run_evaluation(
    completion_fn=generate_chatgpt_output, metrics=[factual_consistency_metric]
)
