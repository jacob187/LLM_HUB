from .base_llm import BaseLLM
from ...utils import available_models

import langchain_anthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


class OpenAILLM(BaseLLM):
    def __init__(self, user_model: str):

        # Keeps track of the technical model name and max tokens
        self.__api_model = available_models.OPENAIMODELS[user_model]["api"]
        self.__max_tokens = available_models.OPENAIMODELS[user_model]["max_output"]

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


# TODO: Implement the methods, determine class specific methods
