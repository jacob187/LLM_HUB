import streamlit as st

from llm_app.backend.llms.llm_factory import LLMFactory


def select_model() -> str:
    """
    Selects a model using selectbox from the available models.

    Returns:
        str: The model name.
    """

    # Create the selectbox
    selected_model = st.selectbox(
        label="Choose an AI model",
        options=(llm_name for llm_name in LLMFactory.merge_models().keys()),
    )
    return selected_model
