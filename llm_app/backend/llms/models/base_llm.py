from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain.base_language import BaseLanguageModel


@dataclass
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
    _llm: BaseLanguageModel[Any]
        The language model object.

    """

    api_model: str = field(init=False)
    provider: str = field(init=False)
    user_model: str
    api_key: str = field(init=False, repr=False)
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    _llm: BaseLanguageModel[Any] = field(init=False, repr=False)

    def __post_init__(self):
        self._llm = self._create_llm()

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
        self.temperature = self.normalize_temperature(temperature)
        self._llm = self._create_llm()
        return self.temperature

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
