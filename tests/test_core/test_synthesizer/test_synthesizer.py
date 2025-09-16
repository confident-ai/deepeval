from typing import List
import pytest
import os

from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
    Evolution,
)
from deepeval.dataset import Golden


TABLES = {
    "students": [
        """CREATE TABLE Students (
        StudentID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Email VARCHAR(100) UNIQUE,
        DateOfBirth DATE,
        Gender CHAR(1),
        Address VARCHAR(200),
        PhoneNumber VARCHAR(15)
    );"""
    ],
}
TEST_SCENARIOS = [
    {
        "scenario": "Food blogger researching international cuisines.",
        "task": "Recipe assistant for suggesting regional dishes.",
        "input_format": "3 sentences long string.",
    },
    {
        "scenario": "New developer learning Python syntax.",
        "task": "Coding copilot for writing simple Python scripts.",
        "input_format": "1-2 lines of code.",
    },
    {
        "scenario": "Entrepreneur seeking advice on launching a startup.",
        "task": "Business coach providing startup tips.",
        "input_format": "2 action items for starting a business.",
    },
]
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_FILES = {
    "pdf": os.path.join(MODULE_DIR, "synthesizer_data", "pdf_example.pdf"),
    "docx": os.path.join(MODULE_DIR, "synthesizer_data", "docx_example.docx"),
    "txt": os.path.join(MODULE_DIR, "synthesizer_data", "txt_example.txt"),
}
SQL_CONTEXTS = list(TABLES.values())
SQL_SOURCES = list(TABLES.keys())


@pytest.fixture
def evolution_config():
    return EvolutionConfig(
        num_evolutions=1,
        evolutions={Evolution.COMPARATIVE: 0.3, Evolution.HYPOTHETICAL: 0.7},
    )


@pytest.fixture
def styling_config():
    scenario = TEST_SCENARIOS[0]
    return StylingConfig(
        scenario=scenario["scenario"],
        task=scenario["task"],
        input_format=scenario["input_format"],
        expected_output_format="3-5 sentences response",
    )


@pytest.fixture
def context_config():
    return ContextConstructionConfig(
        max_contexts_per_document=2,
        min_contexts_per_document=1,
        chunk_size=100,
        max_context_length=4,
        min_context_length=2,
    )


@pytest.fixture
def sync_synthesizer(evolution_config, styling_config):
    return Synthesizer(
        async_mode=False,
        evolution_config=evolution_config,
        styling_config=styling_config,
        max_concurrent=3,
    )


@pytest.fixture
def async_synthesizer(evolution_config, styling_config):
    return Synthesizer(
        async_mode=True,
        evolution_config=evolution_config,
        styling_config=styling_config,
        max_concurrent=3,
    )


def test_generate_goldens_from_contexts(sync_synthesizer: Synthesizer):
    goldens: List[Golden] = sync_synthesizer.generate_goldens_from_contexts(
        contexts=SQL_CONTEXTS,
        source_files=SQL_SOURCES,
        max_goldens_per_context=2,
        _send_data=False,
    )

    assert goldens is not None
    assert isinstance(goldens, list)
    assert len(goldens) > 0
    assert all(isinstance(g, Golden) for g in goldens)

    for golden in goldens:
        assert golden.input is not None
        assert isinstance(golden.input, str)
        assert len(golden.input) > 0
        if hasattr(golden, "expected_output") and golden.expected_output:
            assert isinstance(golden.expected_output, str)


def test_generate_goldens_from_docs(
    sync_synthesizer: Synthesizer, context_config
):
    goldens = sync_synthesizer.generate_goldens_from_docs(
        max_goldens_per_context=1,
        document_paths=[TEST_FILES["txt"]],
        context_construction_config=context_config,
        include_expected_output=True,
        _send_data=False,
    )

    assert goldens is not None
    assert isinstance(goldens, list)
    assert len(goldens) > 0
    assert all(isinstance(g, Golden) for g in goldens)
    for golden in goldens:
        assert golden.source_file is not None
        assert isinstance(golden.source_file, str)


def test_generate_goldens_from_scratch(sync_synthesizer: Synthesizer):
    num_goldens = 2
    goldens = sync_synthesizer.generate_goldens_from_scratch(
        num_goldens=num_goldens,
        _send_data=False,
    )
    assert goldens is not None
    assert isinstance(goldens, list)
    assert len(goldens) > 0
    assert len(goldens) >= 1
    assert all(isinstance(g, Golden) for g in goldens)


@pytest.mark.asyncio
async def test_async_generate_goldens_from_contexts(
    async_synthesizer: Synthesizer,
):
    goldens: List[Golden] = (
        await async_synthesizer.a_generate_goldens_from_contexts(
            contexts=SQL_CONTEXTS, include_expected_output=True
        )
    )
    assert goldens is not None
    assert isinstance(goldens, list)
    assert len(goldens) > 0
    assert all(isinstance(g, Golden) for g in goldens)


@pytest.mark.asyncio
async def test_async_generate_goldens_from_docs(
    async_synthesizer: Synthesizer, context_config
):
    goldens = await async_synthesizer.a_generate_goldens_from_docs(
        max_goldens_per_context=1,
        document_paths=[TEST_FILES["txt"]],
        context_construction_config=context_config,
        include_expected_output=True,
    )
    assert goldens is not None
    assert isinstance(goldens, list)
    assert len(goldens) > 0
    assert all(isinstance(g, Golden) for g in goldens)


@pytest.mark.asyncio
async def test_async_generate_goldens_from_scratch(
    async_synthesizer: Synthesizer,
):
    num_goldens = 2
    goldens = await async_synthesizer.a_generate_goldens_from_scratch(
        num_goldens=num_goldens
    )
    assert goldens is not None
    assert isinstance(goldens, list)
    assert len(goldens) > 0
    assert len(goldens) >= 1
    assert all(isinstance(g, Golden) for g in goldens)
