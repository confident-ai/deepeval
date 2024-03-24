import os
from deepeval.synthesizer import Synthesizer


module_b_dir = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(module_b_dir, "synthesizer_data", "pdf_example.pdf")
synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    docuemnt_paths=[file_path],
    max_goldens_per_document=10,
)
synthesizer.save_as(file_type="json", directory="./results")
