from typing import Iterator, Optional
from langchain_core.messages import HumanMessage
from ..memory.base_memory import BaseMemoryWrapper
from ..llms.models.base_llm import BaseLLM


# TODO fix chat_manager so it just creates response and does not handle settings logic.
class ChatManager:
    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemoryWrapper] = None,
    ):
        self.__llm = llm

        # Todo
        self.__memory = memory

        self.__model = llm.get_language_model()

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the language model based on the given prompt and parameters.

        Args:
            prompt (str): The input text prompt to generate a response for
            temperature (float): Controls randomness in the response generation.
            max_tokens (int): The maximum number of tokens to generate in the response

        Returns:
            str: The generated text response from the language model

        """

        response = self.__model.invoke([HumanMessage(content=prompt)])
        return response.content

    def generate_streamed_response(self, prompt: str) -> Iterator[str]:
        """
        Generate a streamed response from the language model based on the given prompt and parameters.

        Args:
            prompt (str): The input text prompt to generate a response for
            temperature (float): Controls randomness in the response generation.
            max_tokens (int): The maximum number of tokens to generate in the response

        Returns:
            str: The generated text response from the language model

        """

        for chunk in self.__model.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content
