from abc import ABC, abstractmethod
from typing import Any

from langchain.base_language import BaseLanguageModel


class BaseLLM(
    ABC,
):
    """
    An abstract base class for all LLMs.

    Attributes
    ----------

    api_model: str
        The API model name.
    provider: str
        The LLM provider.
    user_model: str
        The English name for the model.
    max_tokens: int
        The maximum number of tokens to generate.
    api_key: str
        The API key for the LLM.
    temperature: float
        The temperature for the model.

    """

    def __init__(
        self,
        api_model: str,
        provider: str,
        user_model: str,
        api_key: str,
        max_tokens: int | None = None,  # Make optional
        temperature: float = 0.7,
    ):
        self.__provider = provider
        self.__api_model = api_model
        self.__user_model = user_model
        self.__temperature = temperature
        self.__max_tokens = max_tokens
        self.__api_key = api_key
        self._llm = self._create_llm()

    @property
    def get_api_model(self) -> str:
        """
        Returns API code for the model.
        """
        return self.__api_model

    @property
    def get_user_model(self) -> str:
        """
        Returns the English name for the model.
        """
        return self.__user_model

    @property
    def get_provider(self) -> str:
        """
        Returns the LLM provider.
        """
        return self.__provider

    @property
    def get_temperature(self) -> float:
        """
        Returns the set temperature for the model.
        """
        return self.__temperature

    @property
    def get_max_tokens(self) -> int | None:
        """
        Retrieves the max tokens for the model.
        """
        return self.__max_tokens

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> int:
        """
        Sets the max tokens for the model.

        Args:
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The maximum number of tokens to generate.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_llm(self) -> BaseLanguageModel[Any]:
        """Create and return a new LLM object."""
        raise NotImplementedError

    def set_temperature(self, temperature: float) -> float:
        """
        Sets the temperature for the model, and updates the Langchain base model.
        """
        self.__temperature = self.normalize_temperature(temperature)
        self._create_llm()
        return self.__temperature

    def normalize_temperature(self, temperature: float) -> float:
        """
        Ensures the temperature is within the valid range of 0.0 to 1.0.
        """
        return max(0.0, min(1.0, temperature))

    def get_language_model(self) -> BaseLanguageModel[Any]:
        """
        Returns the language model object.
        """
        return self._llm
