import streamlit as st

from llm_app.backend.llms.models.anthropic_llm import AnthropicLLM
from llm_app.frontend.components.select_model import select_model


def main():
    st.title("LLM HUB")
    selected_model, api_model = select_model()
    st.write(f"Selected model: {selected_model}")
    st.write(f"API model: {api_model}")


# Component for custome settings (temperature, macx_tokens)


# Component for selecting the model


if __name__ == "__main__":
    main()
