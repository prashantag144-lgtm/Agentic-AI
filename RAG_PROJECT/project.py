from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="CHROMA_DB",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = init_chat_model(
    "groq:openai/gpt-oss-120b",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are a helpful AI assistant.

Use ONLY the provided context.

If answer is not available say:

'I could not find the answer in the document.'
"""
),
(
"human",
"""
Context:
{context}

Question:
{question}
"""
)
]
)


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context":context,
            "question":question
        }
    )

    response = llm.invoke(final_prompt)

    return response.content