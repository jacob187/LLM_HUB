from abc import ABC, abstractmethod
from typing import Iterator

from langchain.base_language import BaseLanguageModel
from langchain_core.messages import HumanMessage


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

    def normalize_temperature(self, temperature: float) -> float:
        return max(0.0, min(1.0, temperature))

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> int:
        raise NotImplementedError

    def generate_steamed_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000
    ) -> Iterator[str]:
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
