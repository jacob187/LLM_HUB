import streamlit as st

from llm_app.backend.utils.available_models import available_models


def select_model():
    options = st.selectbox("Select a model", available_models.get_consumer_model_name())
