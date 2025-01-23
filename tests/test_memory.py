from llm_app.backend.llms.llm_factory import LLMFactory
from llm_app.backend.chat.chat_manager import ChatManager


def chat():
    llm = LLMFactory.create_llm("GPT-4o Mini")
    chat_manager = ChatManager(llm=llm, memory=True)

    print("\nChat started (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        # Iterate over the streamed response chunks
        for chunk in chat_manager.generate_streamed_response(user_input):
            print(chunk, end="", flush=True)

        print()


if __name__ == "__main__":
    chat()
