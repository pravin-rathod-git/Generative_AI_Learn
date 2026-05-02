import streamlit as st
from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 🔹 Load env
load_dotenv()

# 🔹 Initialize model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.8
)

# 🔹 Page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Mistral AI Chatbot")

# 🔹 Mode selection (sidebar)
mode_option = st.sidebar.selectbox(
    "Choose AI Mode",
    ["Angry 😡", "Funny 😂", "Sad 😢"]
)

def get_mode_prompt(mode_option):
    if "Angry" in mode_option:
        return "You are an angry AI. Respond aggressively but stay helpful."
    elif "Funny" in mode_option:
        return "You are a funny AI. Use humor and jokes."
    elif "Sad" in mode_option:
        return "You are a sad AI. Respond emotionally."

# 🔹 Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=get_mode_prompt(mode_option))
    ]

# 🔹 Reset button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = [
        SystemMessage(content=get_mode_prompt(mode_option))
    ]

# 🔹 Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 🔹 User input
user_input = st.chat_input("Type your message...")

# 🔹 Memory limit
MAX_HISTORY = 10

def trim_messages(messages):
    return messages[-MAX_HISTORY:]

# 🔹 Handle input
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages = trim_messages(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = model.invoke(st.session_state.messages)

        st.session_state.messages.append(
            AIMessage(content=response.content)
        )

        with st.chat_message("assistant"):
            st.markdown(response.content)

    except Exception as e:
        st.error(f"Error: {e}")