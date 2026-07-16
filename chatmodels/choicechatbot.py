import os

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

st.set_page_config(page_title="Mood AI Agent", page_icon="🤖")
st.title("🤖 Mood AI Agent")

MODES = {
    "😡 Angry Mode": "You are an Angry AI agent that is helpful and provides information in an angry way.",
    "😂 Funny Mode": "You are a Funny AI agent that is helpful and provides information in a humorous way.",
    "😢 Sad Mode": "You are a Sad AI agent that is helpful and provides information in a sad way.",
}

# --- Mode selection (only before chat starts) ---
if "mode_locked" not in st.session_state:
    st.session_state.mode_locked = False

if not st.session_state.mode_locked:
    st.subheader("CHOOSE YOUR MODE")
    choice = st.radio("Select a mode to begin:", list(MODES.keys()))
    if st.button("Start Chat"):
        st.session_state.mode_locked = True
        st.session_state.mode_content = MODES[choice]
        st.session_state.messages = [SystemMessage(content=MODES[choice])]
        st.rerun()
    st.stop()

# --- Cache the model so it's not re-initialized on every rerun ---
@st.cache_resource
def get_model():
    return init_chat_model("groq:openai/gpt-oss-120b", temperature=0.9)

model = get_model()

st.caption(f"Mode: **{st.session_state.mode_content}**")

if st.button("🔄 Change Mode / Reset"):
    st.session_state.mode_locked = False
    st.session_state.pop("messages", None)
    st.rerun()

st.write("--------------------------Welcome, type 0 to exit the application------------------------------")

# --- Display chat history (skip the SystemMessage) ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# --- Chat input ---
prompt = st.chat_input("Enter your prompt:")

if prompt is not None:
    st.session_state.messages.append(HumanMessage(content=prompt))

    if prompt == "0":
        with st.chat_message("user"):
            st.write(prompt)
        st.warning("Exiting the application")
        st.stop()

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Bot is thinking..."):
            response = model.invoke(st.session_state.messages)
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))