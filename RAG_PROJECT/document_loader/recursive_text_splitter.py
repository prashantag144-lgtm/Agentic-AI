import os

from langchain_community.document_loaders import TextLoader
from langchain_groq import data
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = TextLoader("/workspaces/Agentic-AI/RAG_PROJECT/document_loader/notes.txt")

docs=data.load()


text_splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size=10,
    chunk_overlap=1,

)


chunks=text_splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)  