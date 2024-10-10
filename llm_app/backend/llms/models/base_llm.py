from abc import ABC, abstractmethod
from typing import Iterator

from langchain.base_language import BaseLanguageModel
from langchain_core.messages import HumanMessage


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
        """
        Retruns API code for the model.
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
    def get_max_tokens(self) -> int:
        """
        Retrives the max tokens for the model.
        """
        return self.__max_tokens

    @abstractmethod
    def _create_llm(self) -> BaseLanguageModel:
        """Create and return a new LLM object."""
        raise NotImplementedError

    def normalize_temperature(self, temperature: float) -> float:
        """
        Ensures the temperature is within the valid range of 0.0 to 1.0.

        A
        """
        return max(0.0, min(1.0, temperature))

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

    def generate_steamed_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000
    ) -> Iterator[str]:
        """
         Generates a streamed response from a HumanMessage.

        Args:
            prompt: The prompt to generate a response for.
            temperature: The temperature for the model.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            An iterator of strings, each representing a chunk of the response.
        """
        user_temperature = (
            self.normalize_temperature(float(temperature)) if temperature else 0.7
        )
        user_max_tokens = self.set_max_tokens(int(max_tokens)) if max_tokens else 1000

        self._llm.temperature = user_temperature
        self._llm.max_tokens = user_max_tokens

        print(f"{self._llm.temperature} {self._llm.max_tokens}")

        for chunk in self._llm.stream([HumanMessage(content=prompt)]):
            if chunk.content is not None:
                yield chunk.content

    # TODO: Setup response method so I have different ways to handle the response, i.e.
    # streaming, not streaming, json, etc.
    # TODO: Implement Top P, Stop Sequences, Frequency Penalty, and Presence Penalty.
