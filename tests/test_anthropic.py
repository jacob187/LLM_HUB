import unittest
import sys
import os
from llm_app.backend.llms.models.anthropic_llm import AnthropicLLM
from llm_app.backend.utils import available_models


class TestAnthropicLLM(unittest.TestCase):
    def setUp(self):

        self.llm = AnthropicLLM(user_model="Claude 3 Haiku")

    def test_normalize_temperature(self):
        self.assertEqual(self.llm.normalize_temperature(0.5), 0.5)
        self.assertEqual(self.llm.normalize_temperature(-1), 0.0)
        self.assertEqual(self.llm.normalize_temperature(2), 1.0)

    def test_set_max_tokens(self):
        self.assertEqual(self.llm.set_max_tokens(1000), 1000)
        with self.assertRaises(ValueError):
            self.llm.set_max_tokens(10000)

    def test_generate_response(self):
        response = self.llm.generate_response("Hello, how are you?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotIn(os.getenv("ANTHROPIC_API_KEY"), response)

    def test_getters(self):
        self.assertEqual(self.llm.get_api_model, "claude-3-haiku-20240307")
        self.assertEqual(self.llm.get_user_model, "Claude 3 Haiku")
        self.assertEqual(self.llm.get_provider, "anthropic")
        self.assertEqual(self.llm.get_temperature, 0.7)
        self.assertEqual(self.llm.get_max_tokens, 4096)


def main():
    # Initialize the AnthropicLLM with a specific model
    print("Pick a model: ")

    i = 1
    models = list(available_models.ANTHROPICMODELS.items())
    for key, value in models:
        print(f"{i}. {key}")
        i += 1

    model_number = int(input("Enter the model number: "))
    model = models[model_number - 1][0]
    try:
        llm = AnthropicLLM(model)
        # print(llm.__api_key)
    except ValueError as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    prompt = input("Enter your prompt: ")

    try:
        temperature = input("Enter the temperature 0 - 1: ")
        max_tokens = input("Enter the max tokens: ")
        print(f"\nStreaming response from {model}:\n")
        for chunk in llm.generate_steamed_response(prompt, temperature, max_tokens):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error generating response: {e}")


if __name__ == "__main__":
    # unittest.main()
    main()
