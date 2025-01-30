import streamlit as st
from llm_app.backend.llms.llm_factory import LLMFactory
from llm_app.backend.chat.chat_manager import ChatManager
from llm_app.frontend.components.chat_box import chat_box
from llm_app.frontend.components.settings_sidebar import settings_sidebar

st.set_page_config(
    page_title="LLM Hub",
    page_icon="🤖",
    layout="wide",
)


def main():
    st.title("LLM HUB")

    # Initialize session state
    if "user_model" not in st.session_state:
        st.session_state.user_model = None
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 3000
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = False

    # Model selection and settings through sidebar
    merged_models = LLMFactory.merge_models()
    user_model_max_tokens = (
        merged_models[st.session_state.user_model]["max_output"]
        if st.session_state.user_model
        else 3000
    )

    new_model, new_temp, new_tokens, memory = settings_sidebar(user_model_max_tokens)

    # Update chat manager if model or memory setting changes
    should_update_manager = (
        new_model != st.session_state.user_model
        or memory != st.session_state.memory_enabled
        or new_temp != st.session_state.temperature
        or new_tokens != st.session_state.max_tokens
    )

    # Clear messages if model changes
    if new_model != st.session_state.user_model:
        st.session_state.messages = []  # Clear UI messages
        if (
            st.session_state.chat_manager
            and st.session_state.chat_manager._ChatManager__memory
        ):
            st.session_state.chat_manager._ChatManager__memory.clear()  # Clear memory

    if should_update_manager:
        st.session_state.user_model = new_model
        st.session_state.memory_enabled = memory
        st.session_state.temperature = new_temp
        st.session_state.max_tokens = new_tokens

        # Create new LLM with updated parameters
        llm = LLMFactory.create_llm(st.session_state.user_model)
        llm.set_temperature(new_temp)
        llm.set_max_tokens(new_tokens)

        # If enabling memory, transfer existing messages
        if memory and st.session_state.messages:
            chat_manager = ChatManager(llm=llm, memory=True)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_manager._ChatManager__memory.add_user_message(msg["content"])
                else:
                    chat_manager._ChatManager__memory.add_ai_message(msg["content"])
            st.session_state.chat_manager = chat_manager
        else:
            # If disabling memory or no messages exist
            st.session_state.chat_manager = ChatManager(llm=llm, memory=memory)

    # Display memory status
    if st.session_state.memory_enabled:
        st.sidebar.success("Memory: Enabled")
    else:
        st.sidebar.warning("Memory: Disabled")

    # Display chat box
    if st.session_state.chat_manager:
        chat_box(st.session_state.chat_manager)


if __name__ == "__main__":
    main()
