from abc import ABC, abstractmethod

from langchain.base_language import BaseLanguageModel


class BaseLLM(
    ABC,
):
    """
    Base class for all LLMs.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        max_tokens: int,
        api_key: str,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

    @abstractmethod
    def create_llm(self) -> BaseLanguageModel:
        """Create and return a new LLM object."""
        raise NotImplementedError

    @abstractmethod
    def normalize_temperature(self, temperature: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError
