from abc import ABC, abstractmethod


# TODO: consider implantation; ConversationBufferMemory, Chains, Vector-store, langgraph
# ConversationTokenBufferMemory
class BaseMemoryWrapper(ABC):

    @abstractmethod
    def _create_memory(self):
        pass
