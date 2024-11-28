# llm_app/frontend/components/chat_box.py
import streamlit as st
from llm_app.backend.chat.chat_manager import ChatManager


def chat_box(chat_manager: ChatManager) -> None:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add custom CSS for better text wrapping and formatting
    st.markdown(
        """
        <style>
        .stChatMessage {
            max-width: 100%;
        }
        .stChatMessage .stMarkdown {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }
        .stMarkdown p {
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            white-space: pre-wrap !important;
            margin-right: 1rem !important;
        }
        .stMarkdown pre {
            white-space: pre-wrap !important;
            overflow-x: auto !important;
            max-width: calc(100% - 2rem) !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Rest of the chat_box code remains the same
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

            # Show response in expander
            with st.expander("Assistant is typing...", expanded=True):
                for chunk in chat_manager.generate_streamed_response(prompt=prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        # Force a rerun to display the final message properly
        st.rerun()
