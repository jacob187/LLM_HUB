from .base_llm import BaseLLM
from ...utils import available_models

import langchain_openai
from dotenv import load_dotenv
import os

load_dotenv()


class OpenAILLM(BaseLLM):
    def __init__(self, user_model: str):

        # Keeps track of the technical model name and max tokens
        self.__api_model = available_models.OPENAIMODELS[user_model]["api"]
        self.__max_tokens = available_models.OPENAIMODELS[user_model]["max_output"]

        # Initialize the API key
        self.__api_key = os.getenv("OPENAI_API_KEY")
        if not self.__api_key:
            raise ValueError(
                "OpenAI API key not found, please add it to a .env file at the root of this project."
            )

        super().__init__(
            provider="openai",
            user_model=user_model,
            api_model=self.__api_model,
            api_key=self.__api_key,
            max_tokens=self.__max_tokens,
        )
        self._llm = self._create_llm()

    def _create_llm(self) -> langchain_openai.ChatOpenAI:
        return langchain_openai.ChatOpenAI(
            api_key=self.__api_key,
            model=self.get_api_model,
            max_tokens=self.get_max_tokens,
            temperature=self.get_temperature,
        )

    def set_max_tokens(self, max_tokens: int) -> int:
        MAX_TOKENS = available_models.OPENAIMODELS[self.get_user_model]["max_output"]
        if max_tokens > MAX_TOKENS:
            raise ValueError(
                f"Max tokens {max_tokens} is greater than the maximum allowed {MAX_TOKENS}"
            )
        else:
            return max_tokens
