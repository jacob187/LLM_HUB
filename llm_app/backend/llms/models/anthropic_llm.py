from .base_llm import BaseLLM
from ...utils import available_models

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field

load_dotenv()


@dataclass
class AnthropicLLM(BaseLLM):
    provider: str = field(default="anthropic", init=False)

    def __post_init__(self):
        try:
            model_info = available_models.ANTHROPICMODELS[self.user_model]
            self.api_model = model_info["api"]
            if self.max_tokens is None:
                self.max_tokens = model_info["max_output"]
        except KeyError as e:
            raise ValueError(
                f"Model {self.user_model} not found in available Anthropic models."
            ) from e

        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not found, please add it to a .env file at the root of this project."
                )

    def _create_llm(self) -> ChatAnthropic:
        """
        Creates the Anthropic LLM model given the API key, model, max tokens, and temperature.
        """
        return ChatAnthropic(
            api_key=self.api_key,
            model=self.api_model,
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature,
        )

    def set_max_tokens(self, max_tokens: int) -> int:
        """
        Validates and sets the max tokens for the model.
        Returns the validated max_token value.
        """
        MAX_TOKENS = available_models.ANTHROPICMODELS[self.user_model]["max_output"]
        if max_tokens > MAX_TOKENS:
            raise ValueError(
                f"Max tokens {max_tokens} is greater than the maximum allowed {MAX_TOKENS}"
            )
        self.max_tokens = max_tokens
        self.llm.max_tokens = self.max_tokens
        return self.max_tokens
