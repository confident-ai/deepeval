from deepeval.dataset import EvaluationDataset
from deepeval import evaluate

dataset = EvaluationDataset()
dataset.pull(alias="QA Dataset")

print(len(dataset.goldens))
