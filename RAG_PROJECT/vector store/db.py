from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()



docs=[
    Document(page_content="Python is widely used in Artificial Intelligence",metadata={"source":"AI_book"}),
    Document(page_content="Pandas is widely used for data analysis", metadata={"source":"DataScience_book"}),
    Document(page_content="Neural netweork is used in deep learning ",metadata={"source":"DL_book"}),
]



embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore= Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="Chroma-db"
)

result=vectorstore.similarity_search("what is used foro data analysis ?",k=2)
 
for r in result:
    print(r.page_content)

retriver=vectorstore.as_retriever()

docs=retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)