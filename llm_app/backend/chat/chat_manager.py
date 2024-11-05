from typing import Iterator, Optional
from langchain_core.messages import HumanMessage
from ..memory.base_memory import BaseMemoryWrapper
from ..llms.models.base_llm import BaseLLM


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

    def generate_response(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        validated_temp = self.__llm.normalize_temperature(temperature)
        validated_tokens = self.__llm.set_max_tokens(max_tokens)

        self.__model.temperature = validated_temp
        self.__model.max_tokens = validated_tokens

        response = self.__model.invoke([HumanMessage(content=prompt)])
        return response.content

    def generate_streamed_response(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Iterator[str]:
        validated_temp = self.__llm.normalize_temperature(temperature)
        validated_tokens = self.__llm.set_max_tokens(max_tokens)

        self.__model.temperature = validated_temp
        self.__model.max_tokens = validated_tokens

        for chunk in self.__model.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content
