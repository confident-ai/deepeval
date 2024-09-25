import os
import pytest
import time
from typing import Callable
import asyncio
from deepeval.synthesizer.utils import initialize_embedding_model
from deepeval.synthesizer import Synthesizer, UseCase
from deepeval.dataset import EvaluationDataset
from deepeval.models import OpenAIEmbeddingModel
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.synthesizer import (
    Evolution,
    PromptEvolution,
    RTAdversarialAttack,
    RTVulnerability,
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

# async def test_context_generator(
#     load_docs_function: Callable, is_async: bool = False
# ):
#     start_time = time.time()
#     if is_async:
#         loaded_docs = await load_docs_function()
#     else:
#         loaded_docs = load_docs_function()
#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"Time taken: {duration} seconds")
#     return loaded_docs

# def test_context_generator_generate_contexts():
#     embedder = initialize_embedding_model("text-embedding-3-large")
#     context_generator = ContextGenerator(
#         document_paths=[file_path3, file_path2, file_path1],
#         embedder=embedder,
#         chunk_size=66,
#     )
#     context_generator._load_docs()
#     contexts, _ = context_generator.generate_contexts(5)
#     assert len(contexts) == 15


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

#########################################################
### Generate Goldens ####################################
#########################################################

# def test_generate_goldens(
#     synthesizer: Synthesizer, usecase: UseCase = UseCase.QA
# ):
#     start_time = time.time()
#     goldens = synthesizer.generate_goldens(
#         max_goldens_per_context=2,
#         num_evolutions=2,
#         evolutions={Evolution.COMPARATIVE: 0.2, Evolution.HYPOTHETICAL: 0.8},
#         #contexts=contexts,
#         contexts=sql_context,
#         use_case=usecase,
#         _send_data=False,
#     )
#     end_time = time.time()
#     duration = end_time - start_time
#     print("Generated goldens:", len(goldens))
#     print(f"Time taken: {duration} seconds")
#     print(synthesizer.to_pandas())

# synthesizer_sync = Synthesizer(async_mode=False)
# synthesizer_async = Synthesizer(async_mode=True)
# test_generate_goldens(synthesizer_sync)
# test_generate_goldens(synthesizer_async)

# synthesizer_sync = Synthesizer(async_mode=False)
# synthesizer_async = Synthesizer(async_mode=True)
# test_generate_goldens(synthesizer_sync, UseCase.TEXT2SQL)
# test_generate_goldens(synthesizer_async, UseCase.TEXT2SQL)

#########################################################
### Generate Goldens From Docs ##########################
#########################################################


def test_generate_goldens_from_docs(
    synthesizer: Synthesizer, usecase: UseCase = UseCase.QA
):
    start_time = time.time()
    goldens = synthesizer.generate_goldens_from_docs(
        max_goldens_per_context=2,
        max_contexts_per_document=3,
        num_evolutions=1,
        evolutions={Evolution.COMPARATIVE: 0.2, Evolution.HYPOTHETICAL: 0.8},
        document_paths=document_paths,
        use_case=usecase,
        chunk_size=65,
        _send_data=False,
    )
    end_time = time.time()
    duration = end_time - start_time
    print("Generated goldens from docs:", goldens)
    print(f"Time taken: {duration} seconds")
    print(synthesizer.to_pandas())


synthesizer_sync = Synthesizer(async_mode=False, model=SchematicGPTModel())
synthesizer_async = Synthesizer(async_mode=True, model=SchematicGPTModel())
test_generate_goldens_from_docs(synthesizer_sync)
test_generate_goldens_from_docs(synthesizer_async)

synthesizer_sync = Synthesizer(async_mode=False, model=SchematicGPTModel())
synthesizer_async = Synthesizer(async_mode=True, model=SchematicGPTModel())
test_generate_goldens_from_docs(synthesizer_sync, UseCase.TEXT2SQL)
test_generate_goldens_from_docs(synthesizer_async, UseCase.TEXT2SQL)

#########################################################
### Generate Goldens From Scratch #######################
#########################################################


def test_generate_generate_goldens_from_scratch(synthesizer: Synthesizer):
    start_time = time.time()
    goldens = synthesizer.generate_goldens_from_scratch(
        subject="red-teaming prompts",
        task="elicit discriminatory responses from LLMs",
        output_format="string less than 15 words",
        evolutions={
            PromptEvolution.COMPARATIVE: 0.2,
            PromptEvolution.HYPOTHETICAL: 0.8,
        },
        num_initial_goldens=5,
        num_evolutions=1,
    )
    end_time = time.time()
    duration = end_time - start_time
    print("Generated goldens from scratch:", len(goldens))
    print(f"Time taken: {duration} seconds")
    print(synthesizer.to_pandas())


synthesizer_sync = Synthesizer(async_mode=False)
synthesizer_async = Synthesizer(async_mode=True)
test_generate_generate_goldens_from_scratch(synthesizer_sync)
test_generate_generate_goldens_from_scratch(synthesizer_async)
