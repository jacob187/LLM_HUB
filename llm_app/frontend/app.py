import streamlit as st

from llm_app.backend.llms.llm_factory import LLMFactory
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

    # Initialize session state for chat status
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = False

    if not st.session_state.chat_active:
        user_model = (
            select_model()
        )  # Model selection is only available when chat is inactive
        merged_models = LLMFactory.merge_models()
        user_model_max_tokens = merged_models[user_model]["max_output"]
        temperature, max_tokens = select_settings(user_model_max_tokens)

        # Creates an LLM instance based on the user's model.
        user_llm = LLMFactory.create_llm(user_model=user_model)

        # Set chat active state
        st.session_state.chat_active = True

        chat_box(user_llm, temperature, max_tokens)

        # Button to start a new chat
        if st.button("New Chat"):
            st.session_state.chat_active = False
    else:
        st.write(
            "You are currently in a chat. Click 'New Chat' to select a different model."
        )
        if st.button("New Chat"):
            st.session_state.chat_active = False


if __name__ == "__main__":
    main()
