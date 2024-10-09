import streamlit as st
from llm_app.backend.utils.available_models import OPENAIMODELS, ANTHROPICMODELS


def select_model():
    # Combine both model dictionaries
    all_models = {**OPENAIMODELS, **ANTHROPICMODELS}

    # Create a list of tuples (display_name, api_name)
    model_options = [(name, info["api"]) for name, info in all_models.items()]

    # Create the selectbox
    selected_model = st.selectbox(
        "Select a model",
        options=[name for name, _ in model_options],
        format_func=lambda x: x,
    )

    # Get the corresponding API name
    api_model = next(api for name, api in model_options if name == selected_model)

    return selected_model, api_model
