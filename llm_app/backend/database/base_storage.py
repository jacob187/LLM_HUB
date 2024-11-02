from abc import ABC, abstractmethod
from typing import List, Dict
from uuid import UUID


class BaseStorage(ABC):
    """Base interface for persistence operations"""

    @abstractmethod
    def save_conversation(self, conversation_id: UUID, messages: List[Dict]) -> None:
        pass

    @abstractmethod
    def load_conversation(self, conversation_id: UUID) -> List[Dict]:
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: UUID) -> None:
        pass
