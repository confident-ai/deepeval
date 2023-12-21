from typing import Union, List

from deepeval.metrics import BaseMetric
from deepeval.dataset import EvaluationDataset
from transformers.trainer_callback import TrainerCallback


class DeepEvalCallback(TrainerCallback):
    """
    A [transformers.TrainerCallback] that logs various LLM evaluation metrics to DeepEval
    """
    
    def __init__(self, metrics: Union[BaseMetric, List[BaseMetric]], evaluation_dataset: EvaluationDataset):
        super().__init__()
        self.metrics = metrics
        self.evaluation_dataset = evaluation_dataset
        
    
        
    def on_epoch_end(self, args, state, control, model, tokenizer, **kwargs):
        # if self.eval_steps is not None and state.global_step % self.eval_steps == 0:
        #     input_text = "What if these shoes don't fit?"
        #     context = ["All customers are eligible for a 30 day full refund at no extra costs."]
        #     actual_output = "We offer a 30-day full refund at no extra costs."

        #     # Replace with actual logic for metric calculation
        #     hallucination_metric = HallucinationMetric(minimum_score=0.7)
        #     test_case = LLMTestCase(input=input_text, actual_output=actual_output, context=context)
        #     assert_test(test_case, [hallucination_metric])

        #     # Log or save the metric values as needed
        print("---------ONE EPOCH ENDED---------")
        print(model)

    def on_train_end(self, args, state, control, model, tokenizer, **kwargs):
        print("---------TRAIN ENDED---------")
        print(model)
        