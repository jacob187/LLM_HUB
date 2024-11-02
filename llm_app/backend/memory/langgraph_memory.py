from typing import Any, List
from .base_memory import BaseMemoryWrapper
from langchain_core.messages import BaseMessage


class LangGraphMemory(BaseMemoryWrapper):
    def __init__(self, provider: str):
        super().__init__(provider)
        self.messages: List[BaseMessage] = []

    def _create_memory(self) -> Any:
        return self.messages

    def add_user_message(self, message: str) -> None:
        pass

    def get_memory(self):
        pass
