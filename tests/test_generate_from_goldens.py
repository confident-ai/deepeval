from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset


def test_expand_dataset_from_contexts():
    dataset = EvaluationDataset()
    dataset.pull("DataWiz QA Dataset")
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_goldens(dataset.goldens)
    new_dataset = EvaluationDataset(goldens=goldens)
    new_dataset.push("Expanded DataWiz QA Dataset")


def test_expand_dataset_from_inputs():
    dataset = EvaluationDataset()
    dataset.pull("QA Dataset")
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_goldens(dataset.goldens)
    new_dataset = EvaluationDataset(goldens=goldens)
    new_dataset.push("Expanded QA Dataset")


if __name__ == "__main__":
    # test_expand_dataset_from_contexts()
    test_expand_dataset_from_inputs()
