import os
import pytest
from deepeval.synthesizer import Synthesizer, UseCase
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

synthesizer = Synthesizer(
    embedder=OpenAIEmbeddingModel(model="text-embedding-3-large")
)
results = synthesizer.generate_goldens_from_scratch(
    subject="red-teaming prompts",
    task="elicit discriminatory responses from LLMs",
    output_format="string less than 15 words",
    num_initial_goldens=5,
)
print(results)

# goldens = synthesizer.generate_goldens_from_docs(
#     document_paths=[file_path1, file_path2, file_path3],
#     max_goldens_per_document=1,
# )
# goldens = synthesizer.generate_goldens(contexts=[["Wow what a nice day"]])
# goldens = synthesizer.generate_red_teaming_goldens(
#     contexts=[["Yesterday my boss gave me a salary bump."]],
#     include_expected_output=True,
# )

# synthesizer.save_as(file_type="json", directory="./results")

# module_b_dir = os.path.dirname(os.path.realpath(__file__))

# dataset = EvaluationDataset()

# dataset.generate_red_teaming_goldens(max_goldens=2, synthesizer=synthesizer)

# dataset.save_as(file_type="json", directory="./results")

# dataset.save_as(file_type="json", directory="./results")


# table1 = """CREATE TABLE Students (
#     StudentID INT PRIMARY KEY,
#     FirstName VARCHAR(50),
#     LastName VARCHAR(50),
#     Email VARCHAR(100) UNIQUE,
#     DateOfBirth DATE,
#     Gender CHAR(1),
#     Address VARCHAR(200),
#     PhoneNumber VARCHAR(15)
# );"""

# table2 = """CREATE TABLE Courses (
#     CourseID INT PRIMARY KEY,
#     CourseName VARCHAR(100),
#     TeacherID INT,
#     Credits INT,
#     DepartmentID INT,
#     FOREIGN KEY (TeacherID) REFERENCES Teachers(TeacherID),
#     FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
# );"""

# table3 = """CREATE TABLE Enrollments (
#     EnrollmentID INT PRIMARY KEY,
#     StudentID INT,
#     CourseID INT,
#     EnrollmentDate DATE,
#     Grade CHAR(2),
#     FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
#     FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
# );"""

# table4 = """CREATE TABLE Teachers (
#     TeacherID INT PRIMARY KEY,
#     FirstName VARCHAR(50),
#     LastName VARCHAR(50),
#     Email VARCHAR(100) UNIQUE,
#     DepartmentID INT,
#     FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
# );"""

# contexts = [[table1, table2, table3, table4]]
# synthesizer = Synthesizer()
# text_to_sql_goldens = synthesizer.generate_goldens(
#     max_goldens_per_context=15, contexts=contexts, use_case=UseCase.TEXT2SQL
# )
# for golden in text_to_sql_goldens:
#     print("Input             : " + str(golden.input))
#     print("Expected Output   : " + str(golden.expected_output))
