from llm_app.backend.llms.llm_factory import LLMFactory
from llm_app.backend.chat.chat_manager import ChatManager


def chat():
    llm = LLMFactory.create_llm("GPT-4o Mini")
    chat_manager = ChatManager(llm=llm, memory=True)

    print("\nChat started (type 'quit' to exit)")
    print("Type 'debug' to see current memory state")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        if user_input.lower() == "debug":
            print("\nCurrent Memory State:")
            memory = chat_manager._ChatManager__memory.memory
            for i in range(len(memory["user_messages"])):
                print(f"\nTurn {i+1}:")
                print(f"User: {memory['user_messages'][i]}")
                if i < len(memory["ai_messages"]):
                    print(f"AI: {memory['ai_messages'][i]}")
            continue

        print("\nAI: ", end="", flush=True)
        # Iterate over the streamed response chunks
        for chunk in chat_manager.generate_streamed_response(user_input):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    chat()
