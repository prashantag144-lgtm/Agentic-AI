import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

st.set_page_config(page_title="Funny AI Agent", page_icon="🤖")
st.title("🤖 Funny AI Agent")

# ---- Initialize model (same as original script) ----
@st.cache_resource
def get_model():
    return init_chat_model("groq:openai/gpt-oss-120b", temperature=0.9)

model = get_model()

# ---- Initialize chat history in session state ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a Funny AI agent that is helpful and provides information in a humorous way.")
    ]

# ---- Render existing conversation (skip the SystemMessage) ----
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---- Chat input ----
prompt = st.chat_input("Enter your prompt:")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    response = model.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.markdown(response.content)