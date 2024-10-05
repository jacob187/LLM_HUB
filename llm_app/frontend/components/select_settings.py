import streamlit as st


# TODO: implement logic to have slider dynamically change max tokens given the model
def select_settings() -> Tuple[float, int]:
    temperature = 0
    max_tokens = 1000

    return temperature, max_tokens
