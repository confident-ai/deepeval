import os
import pytest
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.models import OpenAIEmbeddingModel


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

# file_path1 = os.path.join(module_b_dir, "synthesizer_data", "pdf_example.pdf")
# file_path2 = os.path.join(module_b_dir, "synthesizer_data", "docx_example.docx")
# file_path3 = os.path.join(module_b_dir, "synthesizer_data", "txt_example.txt")
# synthesizer = Synthesizer(embedder=OpenAIEmbeddingModel(model="text-embedding-3-large"))
# synthesizer.generate_goldens_from_docs(
#     document_paths=[file_path1, file_path2, file_path3],
#     max_goldens_per_document=2,
# )
# synthesizer.generate_goldens(
#     contexts=[["Hey I love the weather"]],
#     max_goldens_per_context=2,
# )
# synthesizer.save_as(file_type="json", directory="./results")

# dataset = EvaluationDataset()
# dataset.generate_goldens_from_docs(
#     synthesizer=synthesizer,
#     document_paths=[file_path1, file_path2, file_path3],
#     max_goldens_per_document=2,
# )
# dataset.save_as(file_type="json", directory="./results")
