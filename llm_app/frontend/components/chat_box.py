# llm_app/frontend/components/chat_box.py
import streamlit as st
from llm_app.backend.chat.chat_manager import ChatManager


def chat_box(chat_manager: ChatManager, temperature: float, max_tokens: int) -> None:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"].capitalize()
        content = message["content"]

        # Shorten the label to the first 50 characters for better UI
        display_text = (content[:47] + "...") if len(content) > 50 else content

        with st.expander(f"{role}: {display_text}", expanded=True):
            st.markdown(content)

    # Accept user input
    if prompt := st.chat_input("Your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.expander(
            "User: " + (prompt[:47] + "..." if len(prompt) > 50 else prompt),
            expanded=True,
        ):
            st.markdown(prompt)

        # Generate assistant response
        with st.expander("Assistant:", expanded=True) as expander:
            message_placeholder = st.empty()
            full_response = ""
            for chunk in chat_manager.generate_streamed_response(
                prompt, temperature, max_tokens
            ):
                full_response += chunk
                message_placeholder.markdown(full_response)
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
