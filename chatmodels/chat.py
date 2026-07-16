""" import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file.")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key,
)

response = model.invoke("What is the meaning of Prashanth ")

print(response.content)  """


import os

from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

model=init_chat_model("groq:openai/gpt-oss-120b",temperature=0.9)
response = model.invoke("Give me a brief history of origin of AI")

print(response.content)