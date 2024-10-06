import unittest
import sys
import os
from llm_app.backend.llms.models.openai_llm import OpenAILLM
from llm_app.backend.utils import available_models


def main():
    # Initialize the AnthropicLLM with a specific model
    print("Pick a model: ")

    i = 1
    models = list(available_models.OPENAIMODELS.items())
    for key, value in models:
        print(f"{i}. {key}")
        i += 1

    model_number = int(input("Enter the model number: "))
    model = models[model_number - 1][0]
    try:
        llm = OpenAILLM(model)
        # print(llm.__api_key)
    except ValueError as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    prompt = input("Enter your prompt: ")

    try:
        temperature = input("Enter the temperature 0 - 1: ")
        max_tokens = input("Enter the max tokens: ")
        response = llm.generate_response(prompt, temperature, max_tokens)

        print(f"\nResponse from {model}:\n{response}")
    except Exception as e:
        print(f"Error generating response: {e}")


if __name__ == "__main__":
    # unittest.main()
    main()
