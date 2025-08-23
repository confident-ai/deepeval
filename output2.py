from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset, Golden
load_dotenv()

goldens = [
    Golden(input="One"),
    Golden(input="Two")
]

dataset = EvaluationDataset(goldens)
dataset.push(alias="Trial2")

dataset2 = EvaluationDataset()
dataset2.pull(alias="Hello")

print(dataset2)
