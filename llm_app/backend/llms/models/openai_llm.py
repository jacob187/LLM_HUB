from .base_llm import BaseLLM
from ...utils import available_models

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field

load_dotenv()


@dataclass
class OpenAILLM(BaseLLM):
    provider: str = field(default="openai", init=False)

    def __post_init__(self):
        try:
            model_info = available_models.OPENAIMODELS[self.user_model]
            self.api_model = model_info["api"]
            # If max_tokens is not provided, use the default from the model info
            if self.max_tokens is None:
                self.max_tokens = model_info["max_output"]
        except KeyError as e:
            raise ValueError(
                f"Model {self.user_model} not found in available OpenAI models."
            ) from e

        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found, please add it to a .env file at the root of this project."
                )

    def _create_llm(self) -> ChatOpenAI:
        """
        Creates the OpenAI LLM model given the API key, model, max tokens, and temperature.
        """
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.api_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
