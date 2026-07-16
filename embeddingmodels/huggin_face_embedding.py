from langchain_huggingface import HuggingFaceEmbeddings

embedding= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text=[
    "Hello this is Prashanth A",
    "Hello your are my gen ai instructor",
    "I want to learn gen ai from you"
]

vector=embedding.embed_documents(text)

print(vector)
