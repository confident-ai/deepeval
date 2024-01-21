from deepeval.test_case import LLMTestCase
from deepeval.dataset.utils import convert_goldens_to_test_cases
from typing import List


def get_column_order(scores: dict):
    order = ["epoch", "step", "loss", "learning_rate"]
    order.extend([key for key in scores.keys() if key not in order])

    return order


def generate_test_cases(
    model, tokenizer, tokenizer_args, evaluation_dataset
) -> List[LLMTestCase]:
    goldens = evaluation_dataset.goldens
    for golden in goldens:
        prompt = f"""{'CONTEXT: ' + str("; ".join(golden.context)) if golden.context else ''}
                QUESTION: {golden.input}
                ANSWER:"""

        tokenized_output = tokenizer(prompt, **tokenizer_args)
        input_ids = tokenized_output.input_ids
        outputs = model.generate(input_ids)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        golden.actual_output = decoded_output

    test_cases = convert_goldens_to_test_cases(evaluation_dataset.goldens)
    return test_cases
