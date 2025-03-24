from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import chromadb
import deepeval
import time

from deepeval.tracing import Tracer, TraceType, AgentAttributes


class MedicalAppointment(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    date: Optional[str] = None
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = None


class MedicalAppointmentSystem:
    def __init__(self, data_directory, db_path):
        self.appointments = {}
        self.load_data(data_directory)
        self.store_data(db_path)
        self.setup_tools()
        self.setup_agent()

    def load_data(self, data_directory):
        self.documents = SimpleDirectoryReader(data_directory).load_data()

    def store_data(self, db_path):
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_or_create_collection("medical_knowledge")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        self.index = VectorStoreIndex.from_documents(
            self.documents, storage_context=storage_context
        )

    def setup_tools(self):
        query_engine = self.index.as_query_engine()
        self.medical_diagnosis_tool = QueryEngineTool.from_defaults(
            query_engine,
            name="medical_diagnosis",
            description="A RAG engine for retrieving medical information.",
        )
        self.get_appointment_state_tool = FunctionTool.from_defaults(
            fn=self.get_appointment_state
        )
        self.update_appointment_tool = FunctionTool.from_defaults(
            fn=self.update_appointment
        )
        self.create_appointment_tool = FunctionTool.from_defaults(
            fn=self.create_appointment
        )
        self.record_diagnosis_tool = FunctionTool.from_defaults(
            fn=self.record_diagnosis
        )
        self.confirm_appointment_tool = FunctionTool.from_defaults(
            fn=self.confirm_appointment, return_direct=True
        )

    def get_appointment_state(self, appointment_id: str) -> str:
        try:
            return str(self.appointments[appointment_id].dict())
        except KeyError:
            return f"Appointment ID {appointment_id} not found"

    def update_appointment(
        self, appointment_id: str, property: str, value: str
    ) -> str:
        appointment = self.appointments.get(appointment_id)
        if appointment:
            setattr(appointment, property, value)
            return f"Appointment ID {appointment_id} updated with {property} = {value}"
        return "Appointment not found"

    def create_appointment(self, appointment_id: str) -> str:
        self.appointments[appointment_id] = MedicalAppointment()
        return "Appointment created."

    def record_diagnosis(self, appointment_id: str, diagnosis: str) -> str:
        appointment = self.appointments.get(appointment_id)
        if appointment and appointment.symptoms:
            appointment.diagnosis = diagnosis
            return f"Diagnosis recorded for Appointment ID {appointment_id}. Diagnosis: {diagnosis}"
        return "Diagnosis cannot be recorded. Please tell me more about your symptoms."

    def confirm_appointment(self, appointment_id: str):
        appointment = self.appointments.get(appointment_id)
        if appointment:
            details = (
                f"Name: {appointment.name}, Email: {appointment.email}, "
                f"Date: {appointment.date}, Symptoms: {appointment.symptoms}, "
                f"Diagnosis: {appointment.diagnosis}"
            )
            confirmation_prompt = f"Details automatically confirmed: {details}."

            # Automatically confirm and return a message
            return "Details confirmed and saved automatically."

        return "Appointment not found."

    def setup_agent(self):
        gpt = OpenAI(model="gpt-4o", temperature=0.1)
        self.agent = FunctionCallingAgent.from_tools(
            tools=[
                self.get_appointment_state_tool,
                self.update_appointment_tool,
                self.create_appointment_tool,
                self.record_diagnosis_tool,
                self.medical_diagnosis_tool,
                self.confirm_appointment_tool,
            ],
            llm=gpt,
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are an expert in medical diagnosis and are now connected to the patient booking system. "
                        "The first step is to create the appointment and record the symptoms. Ask for specific symptoms! "
                        "Then after the symptoms have been created, make a diagnosis. Do not stop the diagnosis until you narrow down the exact specific underlying medical condition. "
                        "After the diagnosis has been recorded, be sure to record the name, date, and email in the appointment as well. Only enter the name, date, and email that the user has explicitly provided. "
                        "Update the symptoms and diagnosis in the appointment."
                        "Once all information has been recorded, confirm the appointment."
                        "!!!!!!!!!!!!!!!!!!!!!!YOU SHOULD CONDUCT EVERYTHING IN THAI AND NOT ENGLISH!!!!!!!!!!!!!!!!!!!!!!"
                    ),
                )
            ],
            max_function_calls=10,
            allow_parallel_tool_calls=False,
        )

    async def a_chat(self, input):
        response: AgentChatResponse = await self.agent.achat(input)
        return response.response

    def interactive_session(self):
        deepeval.trace_llama_index()

        print("Welcome to the Medical Diagnosis and Booking System!")
        print("Please enter your symptoms or ask about appointment details.")

        user_id = 5
        conversation_id = 5

        while True:
            with Tracer(trace_type=TraceType.AGENT) as agent_trace:

                user_input = input("Your query: ")
                if user_input.lower() == "exit":
                    break

                start_time = time.time()
                response: AgentChatResponse = self.agent.chat(user_input)
                end_time = time.time()

                # Logic for restarting conversation
                if response.response.strip().lower() == "restart":
                    user_id += 1
                    conversation_id += 1
                    print(
                        "\nRestarting conversation with new user and conversation IDs..."
                    )
                    print(f"New User ID: user{user_id:03}")
                    print(
                        f"New Conversation ID: conversation{conversation_id:03}"
                    )
                    continue

                agent_trace.set_attributes(
                    AgentAttributes(
                        input=user_input,
                        output=response.response,
                        name="Outer Trace",
                        description="Medical Chatbot Outer Trace",
                    )
                )

                agent_trace.monitor(
                    event_name="Medical Chatbot",
                    model="gpt-4o",
                    input=user_input,
                    response=response.response,
                    retrieval_context=[
                        node.get_text() for node in response.source_nodes
                    ],
                    completion_time=end_time - start_time,
                    distinct_id=f"user{user_id:03}",
                    conversation_id=f"conversation{conversation_id:03}",
                )

                print("Agent Response:", response.response)


target_model = MedicalAppointmentSystem(
    data_directory="experiment/conversation_simulator/test/medical_data",
    db_path="./chroma_db",
)
