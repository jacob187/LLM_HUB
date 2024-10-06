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
        api_model: str,
        provider: str,
        user_model: str,
        max_tokens: int,
        api_key: str,
        temperature: float = 0.7,
    ):
        self.__provider = provider
        self.__api_model = api_model
        self.__user_model = user_model
        self.__temperature = temperature
        self.__max_tokens = max_tokens
        self.__api_key = api_key

    @property
    def get_api_model(self) -> str:
        return self.__api_model

    @property
    def get_user_model(self) -> str:
        return self.__user_model

    @property
    def get_provider(self) -> str:
        return self.__provider

    @property
    def get_temperature(self) -> float:
        return self.__temperature

    @property
    def get_max_tokens(self) -> int:
        return self.__max_tokens

    @abstractmethod
    def _create_llm(self) -> BaseLanguageModel:
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
