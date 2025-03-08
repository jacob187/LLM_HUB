from .base_llm import BaseLLM
from ...utils import available_models

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class OpenAILLM(BaseLLM):
    def __init__(self, user_model: str):
        """
        Initializes the OpenAI LLM model.

        Args:
            user_model: The English name for the model.

        Raises:
            ValueError: If the OpenAI API key is not found.
        """

        # Keeps track of the technical model name and max tokens
        try:
            self.__api_model = available_models.OPENAIMODELS[user_model]["api"]
            self.__max_tokens = available_models.OPENAIMODELS[user_model]["max_output"]
        except Exception as e:
            raise ValueError(
                f"Model {user_model} not found in available OpenAI models."
            )

        # Initialize the API key if it is present.
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

    def _create_llm(self) -> ChatOpenAI:
        """
        Creates the OpenAI LLM model given the API key, model, max tokens, and temperature.
        """

        if self.__api_model == "o1-mini" or self.__api_model == "o3-mini":
            return ChatOpenAI(
                api_key=self.__api_key,
                model=self.__api_model,
                max_tokens=self.max_tokens,
                temperature=1,  # o1-mini and o3-mini models do not support temperature
            )

        return ChatOpenAI(
            api_key=self.__api_key,
            model=self.__api_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def set_max_tokens(self, max_tokens: int) -> int:
        """
        Validates and sets the max tokens for the model.
        Returns the validated max_token value.
        """
        MAX_TOKENS = available_models.OPENAIMODELS[self.user_model]["max_output"]
        if max_tokens > MAX_TOKENS:
            raise ValueError(
                f"Max tokens {max_tokens} is greater than the maximum allowed {MAX_TOKENS}"
            )
        self._BaseLLM__max_tokens = max_tokens  # Update parent class variable
        self._llm = self._create_llm()
        return self._BaseLLM__max_tokens
