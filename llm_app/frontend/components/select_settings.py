from typing import Tuple
import streamlit as st


# TODO: implement logic to have slider dynamically change max tokens given the model
def select_settings(user_model_max_tokens: int) -> Tuple[float, int]:
    """
    Selects the temperature and max tokens for the model.

    Args:
        user_model_max_tokens (int): The maximum number of tokens for the model.

    Returns:
        Tuple[float, int]: The temperature and max tokens.
    """

    temperature = st.slider(
        "Temperature control",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
    )
    max_tokens = st.slider(
        "Max Token control",
        min_value=1,
        # Changes max token choice dynamically based on the model.
        max_value=user_model_max_tokens,
        value=3000,
        step=1,
    )
    return temperature, max_tokens
