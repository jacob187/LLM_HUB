from typing import Optional
from .langgraph_memory import LangGraphMemory
from .base_memory import BaseMemoryWrapper


class MemoryFactory:
    @staticmethod
    def create_memory(
        memory_type: str = "langgraph", provider: str = "default"
    ) -> BaseMemoryWrapper:
        if memory_type == "langgraph":
            return LangGraphMemory(provider=provider)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
