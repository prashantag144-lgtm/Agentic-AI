
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimension=64
)

vector=embeddings.embed_query("you are going to learn gen ai ")

print(vector)