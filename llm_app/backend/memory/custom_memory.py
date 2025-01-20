from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class CustomMemory:
    def __init__(self):
        self.memory = {"user_messages": [], "ai_messages": []}

    def add_user_message(self, message: str) -> None:
        """Add a user message to memory"""
        self.memory["user_messages"].append(message)

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to memory"""
        self.memory["ai_messages"].append(message)

    def get_conversation_history(self) -> str:
        """Get the conversation history as LangChain messages"""

        user_messages = " ".join(self.memory["user_messages"])
        ai_messages = " ".join(self.memory["ai_messages"])

        return ai_messages + user_messages

    def clear(self) -> None:
        """Clear the memory"""
        self.memory = {"user_messages": [], "ai_messages": []}
