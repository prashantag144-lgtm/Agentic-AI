from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Using chat_promptTemplate , we can give the roles to llm

load_dotenv() 

#Data is loaded

""" data = TextLoader("/workspaces/Agentic-AI/RAG_PROJECT/Document_loaders/notes.txt")
docs= data.load()  """

data=PyPDFLoader("/workspaces/Agentic-AI/RAG_PROJECT/Document_loaders/PLANNING AREA.pdf")
docs =data.laod()
template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarises the text"),
     ("human", "{data}")]
)

model = init_chat_model("groq:openai/gpt-oss-120b", temperature=0.9)

messages = template.format_messages(data=docs[0].page_content)
response = model.invoke(messages)
print(response.content)