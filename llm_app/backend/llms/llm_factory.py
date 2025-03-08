from llm_app.backend.llms.models.anthropic_llm import AnthropicLLM
from llm_app.backend.llms.models.openai_llm import OpenAILLM
from llm_app.backend.llms.models.base_llm import BaseLLM
from llm_app.backend.utils.available_models import OPENAIMODELS, ANTHROPICMODELS


class LLMFactory:
    """
    A factory class that creates LLM instances.
    """

    @staticmethod
    def create_llm(user_model: str) -> BaseLLM:
        """
        Creates an LLM instance based on the user's model.

        Args:
            user_model: The user's model, in English.

        Returns:
            A BaseLLM instance.
        """
        if user_model in OPENAIMODELS:
            return OpenAILLM(user_model)
        elif user_model in ANTHROPICMODELS:
            return AnthropicLLM(user_model)
        else:
            raise ValueError(f"Unsupported model: {user_model}")

    @staticmethod
    def merge_models() -> dict[str, str]:
        """
        Creates a dictionary of all available models from the json files.

        Returns:
            A dictionary of all available models.
        """

        return {**OPENAIMODELS, **ANTHROPICMODELS}
