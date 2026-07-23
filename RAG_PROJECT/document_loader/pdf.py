from langchain_community.document_loaders import PyPDFLoader

data= PyPDFLoader("/workspaces/Agentic-AI/RAG_PROJECT/document_loader/PLANNING AREA.pdf")

docs= data.load()

print(docs[0].page_content)  #consitst of meta data and page content