import streamlit as st

from llm_app.backend.llms.models.base_llm import BaseLLM


def chat_box(llm: BaseLLM, temperature: float, max_tokens: int) -> None:
    st.write("This is a chat box")
