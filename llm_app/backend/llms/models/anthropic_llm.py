from .base_llm import BaseLLM
from ...utils import available_models

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field

load_dotenv()


@dataclass
class AnthropicLLM(BaseLLM):
    api_model: str = field(init=False)
    max_tokens: int = field(init=False)
    api_key: str = field(init=False)
    _llm: ChatAnthropic = field(init=False)

    def __post_init__(self):
        self.provider = "anthropic"
        # Keeps track of the technical model name and max tokens
        self.api_model = available_models.ANTHROPICMODELS[self.user_model]["api"]
        self.max_tokens = available_models.ANTHROPICMODELS[self.user_model][
            "max_output"
        ]

        # Initialize the API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found, please add it to a .env file at the root of this project."
            )
        super().__post_init__()
        self._llm = self._create_llm()

    def _create_llm(self) -> ChatAnthropic:
        """
        Creates the OpenAI LLM model given the API key, model, max tokens, and temperature.
        """
        return ChatAnthropic(
            api_key=self.api_key,
            model=self.api_model,
            max_tokens=self.max_tokens,
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
        else:
            self.max_tokens = max_tokens  # Update parent class attribute.
            self._llm = self._create_llm()
            return self.max_tokens
