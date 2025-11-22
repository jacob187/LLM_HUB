from typing import Iterator, Optional
from langchain_core.messages import HumanMessage
from ..llms.models.base_llm import BaseLLM
from ..memory.custom_memory import CustomMemory


class ChatManager:
    def __init__(
        self,
        llm: BaseLLM,
        memory: bool | CustomMemory = False,
    ):

        self._memory: Optional[CustomMemory] = (
            CustomMemory()
            if memory is True
            else memory if isinstance(memory, CustomMemory) else None
        )

        self._model = llm.get_language_model()

    @property
    def has_memory(self) -> bool:
        return self._memory is not None

    @property
    def memory(self) -> Optional[CustomMemory]:
        return self._memory

    def generate_streamed_response(self, prompt: str) -> Iterator[str]:
        if self._memory:
            self._memory.add_user_message(prompt)
            messages = self._memory.get_conversation_history()

            ai_message = ""
            for chunk in self._model.stream(messages):
                if chunk.content:
                    ai_message += chunk.content
                    yield chunk.content

            self._memory.add_ai_message(ai_message)
        else:
            for chunk in self._model.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield chunk.content

    def generate_response(self, prompt: str) -> str:
        if self._memory:
            self._memory.add_user_message(prompt)
            messages = self._memory.get_conversation_history()
            response = self._model.invoke(messages)
            self._memory.add_ai_message(response.content)
            return response.content
        else:
            response = self._model.invoke([HumanMessage(content=prompt)])
            return response.content

    def clear_memory(self) -> None:
        if self._memory:
            self._memory.clear()
