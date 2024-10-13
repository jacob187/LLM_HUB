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
    user_model = select_model()

    merged_models = LLMFactory.merge_models()
    user_model_max_tokens = merged_models[user_model]["max_output"]
    temperature, max_tokens = select_settings(user_model_max_tokens)

    # Creates an LLM instance based on the user's model.
    user_llm = LLMFactory.create_llm(user_model=user_model)

    # st.write(user_llm)
    # st.write(user_llm.get_api_model)
    chat_box(user_llm, temperature, max_tokens)


if __name__ == "__main__":
    main()
