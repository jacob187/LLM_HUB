from llm_app.backend.llms.models.anthropic_llm import AnthropicLLM
from llm_app.backend.llms.models.openai_llm import OpenAILLM
from llm_app.backend.llms.models.base_llm import BaseLLM
from llm_app.backend.utils.available_models import OPENAIMODELS, ANTHROPICMODELS


class LLMFactory:
    @staticmethod
    def create_llm(user_model: str) -> BaseLLM:
        all_models = LLMFactory.merge_models()
        if user_model in OPENAIMODELS:
            return OpenAILLM(user_model)
        elif user_model in ANTHROPICMODELS:
            return AnthropicLLM(user_model)
        else:
            raise ValueError(f"Unsupported model: {user_model}")

    @staticmethod
    def merge_models() -> dict[str, str]:
        return {**OPENAIMODELS, **ANTHROPICMODELS}
