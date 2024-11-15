# llm_app/frontend/components/chat_box.py
import streamlit as st
from llm_app.backend.chat.chat_manager import ChatManager


def chat_box(chat_manager: ChatManager, temperature: float, max_tokens: int) -> None:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        # Create preview text
        preview = (content[:47] + "...") if len(content) > 50 else content

        # Display message with avatar and expandable content
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            with st.expander(preview, expanded=True):
                st.markdown(content)

    # Accept user input
    if prompt := st.chat_input("Your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message with avatar and expandable content
        with st.chat_message("user", avatar="🧑‍💻"):
            preview = (prompt[:47] + "...") if len(prompt) > 50 else prompt
            with st.expander(preview, expanded=True):
                st.markdown(prompt)

        # Generate and display assistant response with avatar
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Show "Thinking..." while generating
            with st.expander("Generating...", expanded=True) as response_expander:
                for chunk in chat_manager.generate_streamed_response(
                    prompt, temperature, max_tokens
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

                # Update expander header after response is complete
                preview = (
                    (full_response[:47] + "...")
                    if len(full_response) > 50
                    else full_response
                )
                response_expander.header(preview)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
