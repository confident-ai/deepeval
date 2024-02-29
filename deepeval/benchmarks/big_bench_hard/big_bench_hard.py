from typing import List
from datasets import load_dataset
# TODO: to be deleted
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
from deepeval.scorer import Scorer



class BigBenchHard(DeepEvalBaseBenchmark):
    def __init__(
        self, tasks: List[BigBenchHardTask] = None
    ):
        super().__init__()
        self.tasks: List[BigBenchHardTask] = tasks
        self.scorer = Scorer()

    def evaluate(self, model: DeepEvalBaseLLM):
        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            total_predictions = len(goldens)
            correct_predictions = 0
            for golden in goldens:
                if self.predict(model, golden):
                    correct_predictions += 1
            
            print(f"Result for Big Bench Hard (task={task.value}: {correct_predictions/total_predictions}")

    def predict(
        self, model: DeepEvalBaseLLM, task: BigBenchHardTask, golden: Golden
    ):
        ##### Example of using the BOOLEAN_EXPRESSIONS task #####

        ##### 1. Define prompt template IF NECESSARY to confine output format ######
        ##### (Only define your own if not found in the original papers) #####
        prompt: dict = BigBenchHardTemplate.generate_output(
            input=input,
            task=task
        )
        prediction = model(prompt)

        ##### 2. Define metrics IF NECESSARY to evaluate prediction #####
        ##### (Only define metrics if not found in the origianl papers) #######
        return self.scorer.exact_match_score(golden.expected_output, prediction)
        

    def load_benchmark_dataset(self, task: BigBenchHardTask) -> List[Golden]:
        # load from hugging face
        dataset = load_dataset("lukaemon/bbh", task.value)
        goldens: List[Golden] = []
        for data in dataset["test"]:
            golden = Golden(input=data["input"], expectedOutput=data["target"])
            goldens.append(golden)

        return goldens

###################################
### Example Usage, delete later ###
###################################
class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def load_model(self):
        return self.model

    def _call(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    def get_model_name(self):
        return "Mistral 7B"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")



#############################
######## Example Usage ######
#############################

######## 1. Create Custom Model to be evaluated #################
mistral_7b = Mistral7B(model=model)

######## 2. Define benchmark with tasks #############
benchmark = BigBenchHard(tasks=[BigBenchHardTask.BOOLEAN_EXPRESSIONS])

######## 3. Provide custom model to be evaluated ##########
benchmark.evaluate(model=mistral_7b)

### Done! ###