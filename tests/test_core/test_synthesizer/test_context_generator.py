import pytest
import os
from itertools import chain

from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.models.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def context_generator_fixture():
    generator = ContextGenerator(
        document_paths=[
            os.path.join(MODULE_DIR, "synthesizer_data", "pdf_example.pdf")
        ],
        embedder=OpenAIEmbeddingModel(),
    )
    yield generator


@pytest.fixture
def ensure_synthesizer_data():
    data_dir = os.path.join(MODULE_DIR, "synthesizer_data")
    pdf_path = os.path.join(data_dir, "pdf_example.pdf")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF file not found: {pdf_path}")


def test_generate_contexts(
    context_generator_fixture,
    ensure_synthesizer_data,
):
    context_generator: ContextGenerator = context_generator_fixture
    contexts, source_files, context_scores = (
        context_generator.generate_contexts(
            max_contexts_per_source_file=2,
            min_contexts_per_source_file=1,
        )
    )
    unique_chunks = len(set(chain.from_iterable(contexts)))
    assert contexts is not None, "Contexts should not be None"
    assert source_files is not None, "Source files should not be None"
    assert context_scores is not None, "Context scores should not be None"
    assert len(contexts) > 0, "No contexts were generated"
    assert unique_chunks > 0, "No unique chunks were utilized"
    assert (
        unique_chunks <= context_generator.total_chunks
    ), "More chunks utilized than available"


def test_multiple_context_generations(
    context_generator_fixture,
    ensure_synthesizer_data,
):
    context_generator: ContextGenerator = context_generator_fixture
    contexts1, _, _ = context_generator.generate_contexts(
        max_contexts_per_source_file=2,
        min_contexts_per_source_file=1,
    )
    contexts2, _, _ = context_generator.generate_contexts(
        max_contexts_per_source_file=2,
        min_contexts_per_source_file=1,
    )
    unique_chunks1 = len(set(chain.from_iterable(contexts1)))
    unique_chunks2 = len(set(chain.from_iterable(contexts2)))
    assert (
        contexts1 is not None and contexts2 is not None
    ), "Both context generations should succeed"
    assert (
        len(contexts1) > 0 and len(contexts2) > 0
    ), "Both generations should produce contexts"
    assert (
        unique_chunks1 > 0 and unique_chunks2 > 0
    ), "Both generations should produce unique chunks"
    assert (
        unique_chunks1 <= context_generator.total_chunks
        and unique_chunks2 <= context_generator.total_chunks
    ), "More chunks utilized than available"
