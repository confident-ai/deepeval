from typing import Callable
import asyncio
import pytest
import time
import os

from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.models import OpenAIEmbeddingModel
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import *
from deepeval.synthesizer import (
    Evolution,
    PromptEvolution,
)

#########################################################
### Context #############################################
#########################################################

table1 = """CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100) UNIQUE,
    DateOfBirth DATE,
    Gender CHAR(1),
    Address VARCHAR(200),
    PhoneNumber VARCHAR(15)
);"""
table2 = """CREATE TABLE Courses (
    CourseID INT PRIMARY KEY,
    CourseName VARCHAR(100),
    TeacherID INT,
    Credits INT,
    DepartmentID INT,
    FOREIGN KEY (TeacherID) REFERENCES Teachers(TeacherID),
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);"""
table3 = """CREATE TABLE Enrollments (
    EnrollmentID INT PRIMARY KEY,
    StudentID INT,
    CourseID INT,
    EnrollmentDate DATE,
    Grade CHAR(2),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);"""
table4 = """CREATE TABLE Teachers (
    TeacherID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100) UNIQUE,
    DepartmentID INT,
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);"""

context_1 = """About MadeUpCompany
MadeUpCompany is a pioneering technology firm founded in 2010, specializing in cloud computing, data analytics, and machine learning. Our headquarters is based in San Francisco, California, with satellite offices spread across New York, London, and Tokyo. We are committed to offering state-of-the-art solutions that help businesses and individuals achieve their full potential. With a diverse team of experts from various industries, we strive to redefine the boundaries of innovation and efficiency."""
context_2 = """Products and Services
We offer a suite of services ranging from cloud storage solutions, data analytics platforms, to custom machine learning models tailored for specific business needs. Our most popular product is CloudMate, a cloud storage solution designed for businesses of all sizes. It offers seamless data migration, top-tier security protocols, and an easy-to-use interface. Our data analytics service, DataWiz, helps companies turn raw data into actionable insights using advanced algorithms."""
context_3 = """Pricing
We have a variety of pricing options tailored to different needs. Our basic cloud storage package starts at $9.99 per month, with premium plans offering more storage and functionalities. We also provide enterprise solutions on a case-by-case basis, so it’s best to consult with our sales team for customized pricing."""
context_4 = """Technical Support
Our customer support team is available 24/7 to assist with any technical issues. We offer multiple channels for support including live chat, email, and a toll-free number. Most issues are typically resolved within 24 hours. We also have an extensive FAQ section on our website and a community forum for peer support."""
context_5 = """Security and Compliance
MadeUpCompany places the utmost importance on security and compliance. All our products are GDPR compliant and adhere to the highest security standards, including end-to-end encryption and multi-factor authentication."""

module_b_dir = os.path.dirname(os.path.realpath(__file__))
file_path1 = os.path.join(module_b_dir, "synthesizer_data", "pdf_example.pdf")
file_path2 = os.path.join(module_b_dir, "synthesizer_data", "docx_example.docx")
file_path3 = os.path.join(module_b_dir, "synthesizer_data", "txt_example.txt")
file_path4 = os.path.join(module_b_dir, "synthesizer_data", "pdf_example_2.pdf")
file_path5 = os.path.join(module_b_dir, "synthesizer_data", "pdf_example_3.pdf")

contexts = [
    [context_1, context_2, context_3, context_4],
    [context_1, context_2, context_3, context_4],
]
sql_context = [
    [table1, table2, table3, table4],
    [table1, table2, table3, table4],
]
document_paths = [file_path1, file_path2, file_path3]


#########################################################
### Scenario, Task, Input Format, Output Format #########
#########################################################

scenarios = [
    {
        "scenario": "Food blogger researching international cuisines.",
        "task": "Recipe assistant for suggesting regional dishes.",
        "input_format": "3 sentences long string.",
    },
    {
        "scenario": "Autistic kid asking about household chores.",
        "task": "Personal assistant for helping with daily tasks.",
        "input_format": "1 sentence step-by-step instruction.",
    },
    {
        "scenario": "New developer learning Python syntax.",
        "task": "Coding copilot for writing simple Python scripts.",
        "input_format": "1-2 lines of code.",
    },
    {
        "scenario": "Senior citizen asking about medications.",
        "task": "Health assistant providing medication instructions.",
        "input_format": "2 sentence easy-to-understand advice.",
    },
    {
        "scenario": "Teenager asking about personal finance.",
        "task": "Financial assistant offering budgeting tips.",
        "input_format": "1 financial tip with a short explanation.",
    },
    {
        "scenario": "Entrepreneur seeking advice on launching a startup.",
        "task": "Business coach providing startup tips.",
        "input_format": "2 action items for starting a business.",
    },
    {
        "scenario": "Middle school student asking about scientific concepts.",
        "task": "Educational tutor explaining science topics.",
        "input_format": "1 explanation of a concept in simple terms.",
    },
    {
        "scenario": "Fitness enthusiast asking about workout routines.",
        "task": "Fitness coach providing exercise suggestions.",
        "input_format": "1 workout suggestion with a brief explanation.",
    },
    {
        "scenario": "Artist asking for inspiration for a new project.",
        "task": "Creative assistant providing artistic ideas.",
        "input_format": "1-2 ideas for art projects.",
    },
    {
        "scenario": "Traveler asking about the best tourist spots.",
        "task": "Travel assistant recommending destinations.",
        "input_format": "1 destination suggestion with a brief reason.",
    },
]

# #########################################################
# ### Test Input Generator ################################
# #########################################################

# synthesizer_sync = Synthesizer(async_mode=False)
# inputs, scores = synthesizer_sync.generate_and_filter_synthetic_inputs(
#     context = [context_1, context_2, context_3, context_4],
#     max_goldens_per_context = 3,
#     max_retries = 3,
#     threshold = 0.5,
# )
# print(inputs)
# print(scores)
# inputs, scores = asyncio.run(
#     synthesizer_sync.a_generate_and_filter_synthetic_inputs(
#         context = [context_1, context_2, context_3, context_4],
#         max_goldens_per_context = 3,
#         max_retries = 3,
#         threshold = 0.5,
#     )
# )
# print(inputs)
# print(scores)

# #########################################################
# ### Test Context Generator ##############################
# #########################################################


async def test_context_generator(
    load_docs_function: Callable, is_async: bool = False
):
    start_time = time.time()
    if is_async:
        loaded_docs = await load_docs_function()
    else:
        loaded_docs = load_docs_function()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration} seconds")
    return loaded_docs


def test_context_generator_generate_contexts():
    embedder = "text-embedding-3-large"
    context_generator = ContextGenerator(
        document_paths=[file_path3, file_path2, file_path1],
        embedder=embedder,
        chunk_size=66,
    )
    context_generator._load_docs()
    contexts, _ = context_generator.generate_contexts(5)
    assert len(contexts) == 15


# embedder = initialize_embedding_model("text-embedding-3-large")
# context_generator = ContextGenerator(
#     document_paths=document_paths, embedder=embedder
# )
# loaded_docs_sync = asyncio.run(
#     test_context_generator(context_generator._load_docs)
# )
# loaded_docs_async = asyncio.run(
#     test_context_generator(context_generator._a_load_docs, is_async=True)
# )
# assert loaded_docs_async == loaded_docs_async
# test_context_generator_generate_contexts()


evolution_config = EvolutionConfig(
    num_evolutions=1,
    evolutions={Evolution.COMPARATIVE: 0.2, Evolution.HYPOTHETICAL: 0.8},
)

filtration_config = FiltrationConfig()
seed = 3
styling_config = StylingConfig(
    scenario=scenarios[seed]["scenario"],
    task=scenarios[seed]["task"],
    input_format=scenarios[seed]["input_format"],
    expected_output_format="3 word answer",
)

synthesizer = Synthesizer(
    async_mode=False,
    evolution_config=evolution_config,
    styling_config=styling_config,
)


#########################################################
### Generate Goldens ####################################
#########################################################


def test_generate_goldens_from_contexts(synthesizer: Synthesizer):
    start_time = time.time()
    goldens = synthesizer.generate_goldens_from_contexts(
        max_goldens_per_context=2,
        contexts=sql_context,
        _send_data=False,
    )
    end_time = time.time()
    duration = end_time - start_time
    print("Generated goldens:", len(goldens))
    print(f"Time taken: {duration} seconds")
    print(synthesizer.to_pandas())


# synthesizer_sync = Synthesizer(async_mode=False)
# synthesizer_async = Synthesizer(async_mode=True)
# test_generate_goldens_from_contexts(synthesizer_sync)
# test_generate_goldens_from_contexts(synthesizer_async)

#########################################################
### Generate Goldens From Docs ##########################
#########################################################


def test_generate_goldens_from_docs(synthesizer: Synthesizer):
    start_time = time.time()

    goldens = synthesizer.generate_goldens_from_docs(
        max_goldens_per_context=1,
        document_paths=document_paths,
        context_construction_config=ContextConstructionConfig(
            chunk_size=100, max_context_length=10
        ),
        _send_data=False,
    )
    end_time = time.time()
    duration = end_time - start_time
    print("Generated goldens from docs:", goldens)
    print(f"Time taken: {duration} seconds")
    print(synthesizer.to_pandas())
    synthesizer.push("Test Dataset 3")


synthesizer_sync = Synthesizer(
    async_mode=False,
)
synthesizer_async = Synthesizer(async_mode=True, max_concurrent=3)

# test_generate_goldens_from_docs(synthesizer_sync)
test_generate_goldens_from_docs(synthesizer_async)

#########################################################
### Generate Goldens From Scratch #######################
#########################################################


def test_generate_generate_goldens_from_scratch(synthesizer: Synthesizer):
    start_time = time.time()
    goldens = synthesizer.generate_goldens_from_scratch(
        num_goldens=5,
    )
    end_time = time.time()
    duration = end_time - start_time
    print("Generated goldens from scratch:", len(goldens))
    print(f"Time taken: {duration} seconds")
    print(synthesizer.to_pandas())


# synthesizer_sync = Synthesizer(async_mode=False)
# synthesizer_async = Synthesizer(async_mode=True)
# test_generate_generate_goldens_from_scratch(synthesizer_sync)
# test_generate_generate_goldens_from_scratch(synthesizer_async)

# #########################################################
# ### Generate From Datasets ##############################
# #########################################################

# seed = 0

# evolution_config_again = EvolutionConfig(    evolutions={
#         Evolution.COMPARATIVE: 0.2,
#         Evolution.HYPOTHETICAL: 0.8,
#     },
#         num_evolutions=1,)
# styling_config_again = StylingConfig(    scenario="red-teaming prompts",
#     task="elicit discriminatory responses from LLMs",
#     input_format="string less than 15 words",)
# synthesizer_again = Synthesizer(evolution_config=evolution_config_again)

# dataset = EvaluationDataset()
# dataset.generate_goldens_from_docs(
#     max_contexts_per_document=1,
#     synthesizer=synthesizer
# )
# print(dataset.goldens)

# dataset = EvaluationDataset()
# dataset.generate_goldens_from_contexts(
#     max_goldens_per_context=2,
#     contexts=contexts,
# )
# print(dataset.goldens)

# dataset = EvaluationDataset()
# dataset.generate_goldens_from_scratch(
#     num_goldens=5,
#     synthesizer=synthesizer_again
# )
# print(dataset.goldens)

#########################################################
### Save to JSON/CSV ####################################
#########################################################


def test_save_goldens(synthesizer: Synthesizer, file_type: str):
    goldens = synthesizer.generate_goldens_from_docs(
        max_goldens_per_context=3,
        document_paths=document_paths,
        context_construction_config=ContextConstructionConfig(chunk_size=100),
        _send_data=False,
    )
    if file_type == "csv":
        synthesizer.save_as("csv", "./goldens")
    elif file_type == "json":
        synthesizer.save_as("json", "./goldens")


def test_load_goldens(file_name: str):
    _, extension = os.path.splitext(file_name)
    dataset = EvaluationDataset()
    print(extension)
    if extension == ".csv":
        dataset.add_goldens_from_csv_file(
            file_name,
            input_col_name="input",
            actual_output_col_name="actual_output",
            expected_output_col_name="expected_output",
            context_col_name="context",
            context_col_delimiter="|",
            source_file_col_name="source_file",
        )
        print(dataset.goldens)
    elif extension == ".json":
        dataset.add_goldens_from_json_file(
            file_name,
            input_key_name="input",
            actual_output_key_name="actual_output",
            expected_output_key_name="expected_output",
            context_key_name="context",
            source_file_key_name="source_file",
        )
        print(dataset.goldens)


synthesizer = Synthesizer(async_mode=True)
# test_save_goldens(synthesizer, "json")
# test_load_goldens("./goldens/20241122_153727.csv")
# test_load_goldens("./goldens/20241122_154545.json")

#########################################################
### Test Costs ##########################################
#########################################################


def test_synthesis_costs(synthesizer: Synthesizer):
    synthesizer.generate_goldens_from_docs(
        max_goldens_per_context=3,
        document_paths=document_paths,
        context_construction_config=ContextConstructionConfig(chunk_size=100),
        _send_data=False,
    )
    # Would be great to test with non-local model
    if synthesizer.using_native_model:
        assert "synthesis_cost" in dir(synthesizer)


#########################################################
### Test Everything #####################################
#########################################################

# test_generate_goldens_from_contexts(synthesizer)
# test_generate_goldens_from_docs(synthesizer)
# test_generate_generate_goldens_from_scratch(synthesizer)
