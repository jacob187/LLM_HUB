import unittest
import sys
import os
from llm_app.backend.llms.models.openai_llm import OpenAILLM
from llm_app.backend.chat.chat_manager import ChatManager
from llm_app.backend.utils import available_models


class TestOpenAILLM(unittest.TestCase):
    def setUp(self):

        self.llm = OpenAILLM(user_model="GPT-4o Mini")
        self.chat_manager = ChatManager(self.llm)

    def test_normalize_temperature(self):
        self.assertEqual(self.llm.normalize_temperature(0.5), 0.5)
        self.assertEqual(self.llm.normalize_temperature(-1), 0.0)
        self.assertEqual(self.llm.normalize_temperature(2), 1.0)

    def test_set_temperature(self):
        self.llm.set_temperature(0.5)
        self.assertEqual(self.llm.get_temperature, 0.5)

    def test_generate_response(self):
        response = self.chat_manager.generate_response(prompt="Hello, how are you?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotIn(os.getenv("OPENAI_API_KEY"), response)

    def test_set_max_tokens(self):
        self.llm.set_max_tokens(1)
        self.chat_manager = ChatManager(llm=self.llm)
        response = self.chat_manager.generate_response(
            prompt="Write a poem that is 1000 words"
        )

        approximate_tokens = len(response) / 4

        self.assertLess(approximate_tokens, 5)

        self.assertEquals(self.llm.get_max_tokens, 1)


def main():
    try:
        llm = OpenAILLM("GPT-4o Mini")
    except ValueError as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    prompt = input("Enter your prompt: ")

    try:
        temp_input = input("Enter the temperature 0 - 1: ").strip()
        max_tokens_input = input("Enter the max tokens: ").strip()

        if temp_input:
            temperature = float(temp_input)
            llm.set_temperature(temperature)
            print(f"Temperature set to: {llm.get_temperature}")

        if max_tokens_input:
            max_tokens = int(max_tokens_input)
            llm.set_max_tokens(max_tokens)
            print(f"Max tokens set to: {llm.get_max_tokens}")

        chat_manager = ChatManager(llm)
        for chunk in chat_manager.generate_streamed_response(prompt=prompt):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error generating response: {e}")


if __name__ == "__main__":

    # main()
    unittest.main()
