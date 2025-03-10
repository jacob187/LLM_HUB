from typing import Iterator
from langchain_core.messages import HumanMessage
from ..llms.models.base_llm import BaseLLM
from ..memory.custom_memory import CustomMemory


class ChatManager:
    def __init__(
        self,
        llm: BaseLLM,
        memory: bool | CustomMemory = False,
    ):

        self.__memory = (
            CustomMemory()
            if memory is True
            else memory if isinstance(memory, CustomMemory) else None
        )

        self.__model = llm.get_language_model()

    def generate_streamed_response(self, prompt: str) -> Iterator[str]:
        """Generate a response from the language model"""
        if self.__memory:
            # Add the new user message to memory
            self.__memory.add_user_message(prompt)

            # Get conversation history as List[BaseMessage]
            messages = self.__memory.get_conversation_history()

            ai_message = ""
            # Pass the FULL conversation history to the model
            for chunk in self.__model.stream(messages):
                if chunk.content:
                    ai_message += chunk.content
                    yield chunk.content

            # Store AI response in memory
            self.__memory.add_ai_message(ai_message)
        else:
            # If no memory, just stream the current message
            for chunk in self.__model.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield chunk.content

    def generate_response(self, prompt: str) -> str:
        """Generate a non-streamed response"""
        if self.__memory:
            self.__memory.add_user_message(prompt)
            messages = self.__memory.get_conversation_history()
            response = self.__model.invoke(messages)
            self.__memory.add_ai_message(response.content)
            return response.content
        else:
            response = self.__model.invoke([HumanMessage(content=prompt)])
            return response.content
