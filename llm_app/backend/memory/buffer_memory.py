from .base_memory import BaseMemoryWrapper

from typing import Any, Optional
from langchain.memory import ConversationBufferMemory


class BufferMemory(BaseMemoryWrapper):
    def __init__(self, buffer_memory: ConversationBufferMemory):
        """
        Initializes the buffer memory.

        Args:
            buffer_memory: The buffer memory object.
        """
        super().__init__(provider="buffer")
        self._buffer_memory = buffer_memory

    def _create_memory(self) -> Any:
        """Create memory instance"""
        return ConversationBufferMemory()

    def add_user_message(self, message: str) -> None:
        """Add user message to memory"""
        self.memory.add_user_message(message)

    def get_memory(self) -> Optional[ConversationBufferMemory]:
        """Return memory"""
        return self.memory
