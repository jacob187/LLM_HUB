import streamlit as st
from llm_app.backend.llms.llm_factory import LLMFactory
from llm_app.backend.chat.chat_manager import ChatManager
from llm_app.frontend.components.select_model import select_model
from llm_app.frontend.components.select_settings import select_settings
from llm_app.frontend.components.chat_box import chat_box

st.set_page_config(
    page_title="LLM Hub",
    page_icon="🤖",
    layout="centered",
)


def main():
    st.title("LLM HUB")

    # Initialize session state (references `app.py`, startLine: 17, endLine: 27)
    if "user_model" not in st.session_state:
        st.session_state.user_model = None
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1000
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Model selection
    new_model = select_model()
    if new_model != st.session_state.user_model:
        st.session_state.user_model = new_model
        llm = LLMFactory.create_llm(st.session_state.user_model)
        st.session_state.chat_manager = ChatManager(llm=llm)
        st.session_state.messages = []

    # Settings (references `select_settings.py`, startLine: 5, endLine: 32)
    merged_models = LLMFactory.merge_models()
    user_model_max_tokens = merged_models[st.session_state.user_model]["max_output"]
    st.session_state.temperature, st.session_state.max_tokens = select_settings(
        user_model_max_tokens
    )

    # Display chat box
    chat_box(
        st.session_state.chat_manager,
        st.session_state.temperature,
        st.session_state.max_tokens,
    )

    # Button to start a new chat
    if st.button("New Chat"):
        st.session_state.messages = []  # Clear messages
        st.rerun()  # Force a rerun to update the UI


if __name__ == "__main__":
    main()
