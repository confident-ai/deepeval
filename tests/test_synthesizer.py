import os
import pytest
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset


@pytest.mark.skip(reason="openai is expensive")
def test_synthesizer():
    module_b_dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(
        module_b_dir, "synthesizer_data", "pdf_example.pdf"
    )
    synthesizer = Synthesizer()
    synthesizer.generate_goldens_from_docs(
        document_paths=[file_path],
        include_expected_output=True,
        max_goldens_per_document=2,
    )
    synthesizer.save_as(file_type="json", directory="./results")


# module_b_dir = os.path.dirname(os.path.realpath(__file__))

# file_path = os.path.join(
#     module_b_dir, "synthesizer_data", "pdf_example.pdf"
# )
# synthesizer = Synthesizer(model="gpt-3.5-turbo")
# synthesizer.generate_goldens_from_docs(
#     document_paths=[file_path],
#     max_goldens_per_document=2,
# )
# synthesizer.save_as(file_type="json", directory="./results")

# dataset = EvaluationDataset()
# dataset.generate_goldens(
#     contexts=[["as"]]
# )
# dataset.save_as(file_type="json", directory="./results")
