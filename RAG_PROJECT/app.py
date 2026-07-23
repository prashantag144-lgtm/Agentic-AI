import streamlit as st
from project import ask_question

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 RAG Document Assistant")

# ==========================
# Upload Section
# ==========================

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:

    with open(f"uploaded_files/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File Uploaded Successfully!")

# ==========================
# Question Section
# ==========================

question = st.text_input("Ask a Question")

if st.button("Search"):

    if question:
        with st.spinner("Searching..."):
            answer = ask_question(question)

        st.success("Answer")
        st.write(answer)

    else:
        st.warning("Please enter a question.")