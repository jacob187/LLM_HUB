from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain.base_language import BaseLanguageModel


@dataclass
class BaseLLM(ABC):
    """
    An abstract base class for all LLMs.

    Attributes
    ----------
    user_model: str
        The English name for the model.
    api_key: str
        The API key for the LLM.
    temperature: float
        The temperature for the model.
    max_tokens: int, optional
        The maximum number of tokens to generate, by default None
    """

    user_model: str
    api_key: Optional[str] = field(default=None, repr=False)
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # To be defined in subclasses
    api_model: str = field(init=False)
    provider: str = field(init=False)

    _llm: Optional[BaseLanguageModel[Any]] = field(init=False, repr=False, default=None)

    @property
    def llm(self) -> BaseLanguageModel[Any]:
        """Lazily initializes and returns the language model object."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    @abstractmethod
    def _create_llm(self) -> BaseLanguageModel[Any]:
        """Create and return a new LLM object."""
        raise NotImplementedError

    def set_temperature(self, temperature: float) -> float:
        """
        Sets the temperature for the model, and updates the Langchain base model.
        """
        self.temperature = self.normalize_temperature(temperature)
        self.llm.temperature = self.temperature
        return self.temperature

    def set_max_tokens(self, max_tokens: int) -> int:
        """
        Sets the max tokens for the model.
        """
        self.max_tokens = max_tokens
        self.llm.max_tokens = self.max_tokens
        return self.max_tokens

    def normalize_temperature(self, temperature: float) -> float:
        """
        Ensures the temperature is within the valid range of 0.0 to 1.0.
        """
        return max(0.0, min(1.0, temperature))

    def get_language_model(self) -> BaseLanguageModel[Any]:
        """
        Returns the language model object.
        """
        return self.llm
