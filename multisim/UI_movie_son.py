import streamlit as st
from dotenv import load_dotenv
from typing import Optional

from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# Initialize model
model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0.9
)

# Pydantic Schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: list[str]
    director: Optional[str]
    cast: list[str]
    rating: Optional[float]
    summary: str

# Output Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the paragraph.

{format_instructions}
"""
    ),
    ("human", "{paragraph}")
])

# ---------------- Streamlit UI ---------------- #

st.title("Movie Information Extractor")

paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=250
)

if st.button("Extract Information"):

    if paragraph.strip():

        final_prompt = prompt.invoke(
            {
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions()
            }
        )

        response = model.invoke(final_prompt)

        st.subheader("Output")
        st.write(response.content)

    else:
        st.warning("Please enter a paragraph.")