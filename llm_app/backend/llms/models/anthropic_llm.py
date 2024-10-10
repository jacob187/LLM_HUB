from .base_llm import BaseLLM
from ...utils import available_models

import langchain_anthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


class AnthropicLLM(BaseLLM):
    def __init__(self, user_model: str):
        """
        Initializes the OpenAI LLM model.

        Args:
            user_model: The English name for the model.

        Raises:
            ValueError: If the OpenAI API key is not found.
        """

        # Keeps track of the technical model name and max tokens
        self.__api_model = available_models.ANTHROPICMODELS[user_model]["api"]
        self.__max_tokens = available_models.ANTHROPICMODELS[user_model]["max_output"]

        # Initialize the API key
        self.__api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.__api_key:
            raise ValueError(
                "Anthropic API key not found, please add it to a .env file at the root of this project."
            )

        super().__init__(
            provider="anthropic",
            user_model=user_model,
            api_model=self.__api_model,
            api_key=self.__api_key,
            max_tokens=self.__max_tokens,
        )
        self._llm = self._create_llm()

    def _create_llm(self) -> langchain_anthropic.ChatAnthropic:
        return langchain_anthropic.ChatAnthropic(
            api_key=self.__api_key,
            model=self.get_api_model,
            max_tokens=self.get_max_tokens,
            temperature=self.get_temperature,
        )

    def set_max_tokens(self, max_tokens: int) -> int:
        MAX_TOKENS = available_models.ANTHROPICMODELS[self.get_user_model]["max_output"]
        if max_tokens > MAX_TOKENS:
            raise ValueError(
                f"Max tokens {max_tokens} is greater than the maximum allowed {MAX_TOKENS}"
            )
        else:
            return max_tokens
