from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseLLM(
    ABC,
):
    """
    Base class for all LLMs.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        available_models: List[str],
        max_tokens: int,
        _api_key: str,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.model = model
        self.available_models = available_models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = _api_key

    @abstractmethod
    def normalize_temperature(self, temperature_range: Tuple(int, int)) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError
