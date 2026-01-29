"""
Session OpenAI Agent
Complexity: HIGH - Conversation history
"""
from agents import Agent, ModelSettings, SQLiteSession

def get_agent():
    return Agent(
        name="SessionAgent",
        instructions="Remember the user's name.",
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0.0),
    )

def get_session(session_id: str):
    # In-memory DB for test isolation
    return SQLiteSession(session_id=session_id, db_path=":memory:")