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
        st.session_state.max_tokens = 300
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Model selection and settings through sidebar
    merged_models = LLMFactory.merge_models()
    user_model_max_tokens = (
        merged_models[st.session_state.user_model]["max_output"]
        if st.session_state.user_model
        else 3000
    )

    new_model, new_temp, new_tokens = settings_sidebar(user_model_max_tokens)

    # Update model if changed
    if new_model != st.session_state.user_model:
        st.session_state.user_model = new_model
        llm = LLMFactory.create_llm(st.session_state.user_model)
        st.session_state.chat_manager = ChatManager(llm=llm)
        st.session_state.messages = []

    st.session_state.temperature = new_temp
    st.session_state.max_tokens = new_tokens

    # Display chat box
    if st.session_state.chat_manager:
        chat_box(
            st.session_state.chat_manager,
            st.session_state.temperature,
            st.session_state.max_tokens,
        )


if __name__ == "__main__":
    main()
