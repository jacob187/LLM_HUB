import unittest
import sys
from llm_app.backend.llms.anthropic_llm import AnthropicLLM


class TestAnthropicLLM(unittest.TestCase):
    def setUp(self):
        self.llm = AnthropicLLM(model="Claude 3 Haiku")

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


def main():
    # Initialize the AnthropicLLM with a specific model
    model = "Claude 3.5 Sonnet"
    try:
        llm = AnthropicLLM(model)
    except ValueError as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    prompt = input("Enter your prompt: ")

    try:
        temperature = float(input("Enter the temperature 0 - 1: "))
        max_tokens = int(input("Enter the max tokens: "))
        response = llm.generate_response(prompt, temperature, max_tokens)

        print(f"\nResponse from {model}:\n{response}")
    except Exception as e:
        print(f"Error generating response: {e}")


if __name__ == "__main__":
    # Running unit tests will use tokens
    # unittest.main()
    main()
