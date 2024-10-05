from .base_llm import BaseLLM
from ...utils import available_models

import langchain_anthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str):
        # Keeps track of the non-technical model name
        self._user_model = model

        # Keeps track of the technical model name and max tokens
        self._api_model = available_models.ANTHROPICMODELS[model]["api"]
        max_tokens = available_models.ANTHROPICMODELS[model]["max_output"]

        # Initialize the API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found")

        super().__init__(
            provider="anthropic",
            model=self._api_model,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        self.llm = self.create_llm()

    def create_llm(self) -> langchain_anthropic.ChatAnthropic:
        return langchain_anthropic.ChatAnthropic(
            api_key=self.api_key,
            model=self._api_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    @property
    def user_model(self) -> str:
        return self._user_model

    @property
    def api_model(self) -> str:
        return self._api_model

    def normalize_temperature(self, temperature: float) -> float:
        return max(0.0, min(1.0, temperature))

    def set_max_tokens(self, max_tokens: int) -> int:
        MAX_TOKENS = available_models.ANTHROPICMODELS[self.user_model]["max_output"]
        if max_tokens > MAX_TOKENS:
            raise ValueError(
                f"Max tokens {max_tokens} is greater than the maximum allowed {MAX_TOKENS}"
            )
        else:
            return max_tokens

    def generate_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000
    ) -> str:
        user_temperature = (
            self.normalize_temperature(float(temperature)) if temperature else 0.7
        )
        user_max_tokens = self.set_max_tokens(int(max_tokens)) if max_tokens else 1000

        self.llm.temperature = user_temperature
        self.llm.max_tokens = user_max_tokens

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
