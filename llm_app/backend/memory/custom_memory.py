from typing import List
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

    def get_conversation_history(self) -> List[BaseMessage]:
        """
        Get the conversation history as a list of LangChain BaseMessage objects.
        This format is required for the LangChain chat model's stream/invoke methods.
        """
        messages = []
        for user_msg, ai_msg in zip(
            self.memory["user_messages"], self.memory["ai_messages"]
        ):
            messages.extend(
                [
                    HumanMessage(content=user_msg),
                    AIMessage(content=ai_msg),
                ]
            )

        # Add final user message if present
        if len(self.memory["user_messages"]) > len(self.memory["ai_messages"]):
            messages.append(HumanMessage(content=self.memory["user_messages"][-1]))

        return messages

    def clear(self) -> None:
        """Clear the memory"""
        self.memory = {"user_messages": [], "ai_messages": []}
