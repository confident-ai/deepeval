import uuid
from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import StructuredTool, tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from deepeval.test_case import ConversationalTestCase, Turn

## Comment in once you have qdrant installed
# from qdrant_client import models, QdrantClient
# from sentence_transformers import SentenceTransformer


class Appointment(BaseModel):
    id: str
    name: str
    email: str
    date: date
    symptoms: Optional[List[str]] = Field(default=None)
    diagnosis: Optional[str] = Field(default=None)


# Simple in-memory store for chat histories
chat_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]


class MedicalChatbot:
    def __init__(self, model: str, system_prompt: str):
        self.model = ChatOpenAI(model=model)
        self.system_prompt = system_prompt
        # For RAG engine
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        # For managing appointments
        self.appointments: List[Appointment] = []

        # Setup agent with memory
        self.setup_agent()

    def setup_agent(self):
        """Setup the agent with tools and memory"""

        # Create prompt messages
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt), 
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")
        ])

        # Create agent
        tools = [
            StructuredTool.from_function(func=self.retrieve_knowledge), 
            StructuredTool.from_function(func=self.create_appointment)
        ]
        agent = create_tool_calling_agent(self.model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        self.agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    @tool
    def retrieve_knowledge(self, query: str) -> str:
        """A tool to retrieve data on various diagnosis methods from gale encyclopedia"""
        hits = self.client.query_points(
            collection_name="gale_encyclopedia", 
            query=self.encoder.encode(query).tolist(), 
            limit=3
        ).points

        contexts = [hit.payload['content'] for hit in hits]
        return "\n".join(contexts)

    @tool
    def create_appointment(self, name: str, email: str, date: str) -> str:
        """Create a new appointment with the given ID, name, email, and date"""
        try:
            appointment = Appointment(
                id=str(uuid.uuid4()),
                name=name,
                email=email,
                date=date.fromisoformat(date)
            )
            self.appointments.append(appointment)
            return f"Created new appointment with ID: {appointment.id} for {name} on {date}."
        except ValueError:
            return f"Invalid date format. Please use YYYY-MM-DD format."

    def index_knowledge(self, document_path: str):
        """Index medical knowledge from a text file into Qdrant"""
        with open(document_path) as file:
            documents = file.readlines()

        # Create namespace in qdrant
        self.client.create_collection(
            collection_name="gale_encyclopedia",
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(), 
                distance=models.Distance.COSINE
            ),
        )

        # Vectorize and index into qdrant
        self.client.upload_points(
            collection_name="gale_encyclopedia",
            points=[
                models.PointStruct(
                    id=idx, 
                    vector=self.encoder.encode(doc).tolist(), 
                    payload={"content": doc}
                ) for idx, doc in enumerate(documents)
            ],
        )


# Initialize test case list
test_cases = []

def start_session(chatbot: MedicalChatbot, session_id: Optional[str] = None):
    """Start an interactive session with the chatbot"""
    print("Hello! I am Baymax, your personal healthcare companion.")
    print("How are you feeling today? (type 'exit' to quit)")

    # Initialize turns list
    turns = []

    while True:
        if session_id is None:
            session_id = str(uuid.uuid4())

        user_input = input("Your query: ")
        if user_input.lower() == 'exit':
            break

        response = chatbot.agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        # Add turns to list
        turns.append(Turn(role="user", content=user_input))
        turns.append(Turn(role="assistant", content=response["output"]))

        print("Baymax:", response["output"])



# These parameters will be evaluated later
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are Baymax, a friendly and knowledgeable medical assistant. 
Your role is to help patients by:
1. Listening to their symptoms and concerns
2. Using medical knowledge to provide informed responses
3. Helping them schedule appointments when needed
4. Being empathetic and professional in all interactions

Always prioritize patient safety and recommend professional medical consultation for serious symptoms."""

# Initialize chatbot and start session
chatbot = MedicalChatbot(model=MODEL, system_prompt=SYSTEM_PROMPT)

# Note: You'll need to provide the path to your medical encyclopedia text file
# chatbot.index_knowledge("path-to-your-encyclopedia.txt")

start_session(chatbot) 

# Print test cases
print(test_cases)