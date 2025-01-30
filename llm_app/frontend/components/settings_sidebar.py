import streamlit as st
from llm_app.frontend.components.select_model import select_model
from llm_app.frontend.components.select_settings import select_settings


def settings_sidebar(user_model_max_tokens: int):
    """
    Creates a collapsible sidebar with model and parameter controls
    """
    # Use Streamlit's native sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        selected_model = select_model()
        temperature, max_tokens = select_settings(user_model_max_tokens)
        memory = st.checkbox("Memory")

    return selected_model, temperature, max_tokens, memory
