from abc import ABC, abstractmethod
from typing import Any, Optional


# TODO: consider implantation; ConversationBufferMemory, Chains, Vector-store, langgraph
# ConversationTokenBufferMemory
class BaseMemoryWrapper(ABC):

    def __init__(self):
        self.memory = self._create_memory()

    @abstractmethod
    def _create_memory(self) -> Any:
        """Create memory instance"""
        raise NotImplementedError

    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """Add user message to memory"""
        raise NotImplementedError

    @abstractmethod
    def get_memory(self) -> Optional[Any]:
        """Return memory"""
        return self.memory
