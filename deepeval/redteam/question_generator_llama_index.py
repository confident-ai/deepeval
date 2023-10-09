import os
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.evaluation import DatasetGenerator
from llama_index.llms import OpenAI
from typing import List, Optional
import openai


class QuestionGenerator:
    """
    An automated question generator leveraging the llama_index.

    This class is designed to generate questions from a given document.
    It utilizes the llama_index to produce questions and also allows the inclusion 
    of custom 'bad' questions.

    Attribute
        llm: The language model from llama_index.
        service_context: Service context for the language model.
    """

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0, openai_api_key: Optional[str] = None):
        """
        Initializes the QuestionGenerator with the specified model and temperature.

        Args:
            model_name (str): The name of the model to be used. Default is "gpt-4".
            temperature (float): The temperature setting for the model. Default is 0.
            open_api_key (str): The OpenAI api key
        """
        openai.api_key = openai_api_key
        self.llm = OpenAI(temperature=temperature, model=model_name)
        self.service_context = ServiceContext.from_defaults(llm=self.llm)

    def generate_questions(self, num_questions: int, directory_path: str, bad_questions: Optional[List[str]] = None) -> List[str]:
        """
        Generates questions based on the content of the document at the specified directory path.

        Args:
            num_questions (int): The number of questions to be generated.
            directory_path (str): The path to the directory containing the document.
            bad_questions (list, optional): A list of custom 'bad' questions to be appended to the generated questions.

        Returns:
            list: A list of generated questions combined with the 'bad' questions if provided.
        """
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        data_generator = DatasetGenerator.from_documents(documents)

        eval_questions = data_generator.generate_questions_from_nodes(num=num_questions)

        if bad_questions:
            eval_questions += bad_questions

        return eval_questions


