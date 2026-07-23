import os

from langchain_community.document_loaders import TextLoader
from langchain_groq import data
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("/workspaces/Agentic-AI/RAG_PROJECT/document_loader/notes.txt")

docs=data.load()


text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=8,
    chunk_overlap=1,

)


chunks=text_splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)  #consitst of meta data and page content

# Docs contains the list of documents where it consists of meta data and page content
#Page content is the actual data

""" print(docs[0]) """  #consitst of meta data and page content

# To view only page content

""" print(docs[0].page_content) """

# this page content can be use in main file also

# Load the text loader in main.py