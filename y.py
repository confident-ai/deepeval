from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="Legal Documents Dataset")

print(dataset)
