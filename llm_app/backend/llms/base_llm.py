from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field, PrivateAttr


class BaseLLM(ABC, BaseModel):
    """
    Base class for all LLMs.
    """

    provider: str = Field(description="The provider of the LLM.")
    model: str = Field(description="The model to use.")
    available_models: List[str] = Field(
        frozen=True, description="The available models."
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature, degree of randomness or creativity, of the model.",
    )
    # TODO: Validate max_tokens based on the model
    max_tokens: int = Field(
        default=1024, description="The maximum number of tokens to generate."
    )

    _api_key: str = PrivateAttr()

    def set_api_key(self, value: str) -> None:
        self._api_key = value

    @abstractmethod
    def get_available_models(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def initialize_available_models(self) -> None:
        pass
