# load pdf 
# split into chunks
#  the embedding 
# stores into chroma


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data=PyPDFLoader("/workspaces/Agentic-AI/RAG_PROJECT/document_loader/PLANNING AREA.pdf")
docs =data.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size=10,
    chunk_overlap=1,

)

chunks=text_splitter.split_documents(docs)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"

)

