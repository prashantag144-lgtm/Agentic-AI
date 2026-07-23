import os
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

url="https://www.apple.com/in/macbook-pro/"

data= WebBaseLoader(url)

docs= data.load()

print(docs[0].page_content)  #consitst of meta data and page content