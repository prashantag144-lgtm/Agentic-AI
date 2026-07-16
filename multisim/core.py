from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import list,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()



from langchain.chat_models import init_chat_model


model=init_chat_model(model="groq:openai/gpt-oss-120b",temperature=0.9)

class Movie(BaseModel):
    title:str
    release_year:Optional[int]
    genre: list[str]
    director:Optional[str]
    cast: list[str]
    rating:Optional[float]
    summary: str

parser=PydanticOutputParser(pydantic_object=Movie)

prompt=ChatPromptTemplate.from_messages(
[
    ("system",
        """You are an expert Information Extraction Assistant.

Your task is to analyze the given paragraph and extract the most useful information in a clear, organized, and concise manner.

Instructions:
- Read the paragraph carefully.
- Extract only information that is explicitly mentioned.
- Do not make assumptions or invent facts.
- Use simple and professional language.
- If a particular section has no relevant information, write "Not Mentioned."
- Avoid repeating the same information in multiple sections.
- Present the output using clear headings.

Extract the following information:

Title:
Identify the title or name of the subject, if available.

Category:
Identify the category or domain (e.g., Movie, Technology, Science, History, Business, Sports).

Main Topic:
Describe the central topic in one sentence.

Overview:
Write a concise paragraph (3-5 sentences) summarizing the content.

Key People:
List all important people mentioned and briefly describe their role.

Organizations:
List any organizations, institutions, or companies mentioned.

Locations:
Mention all locations or places referred to.

Important Concepts:
Identify scientific, technical, historical, or domain-specific concepts discussed.

Major Events:
Describe the significant events or actions mentioned.

Key Themes:
List the major themes or ideas explored.

Important Facts:
Highlight the most important factual points.

Cause and Effect:
Identify any cause-and-effect relationships present in the paragraph.

Keywords:
List 10-20 important keywords that best represent the paragraph.

Possible Questions:
Generate a few meaningful questions that someone could ask after reading this paragraph.

Search Tags:
Generate useful tags that could help categorize or search this content.

Final Summary:
Write a polished paragraph that captures the complete meaning of the original text while preserving all important information.

Guidelines:
- Be accurate.
- Be concise.
- Be informative.
- Never hallucinate information.
- Preserve names exactly as written.

Paragraph:
""")
,
('human',"""

Extract information from this paragraph:
 
 {paragraph}

""")
]
)

para= input("give your paragraph")

final_prompt=prompt.invoke(
    {"paragraph":para}
)


response=model.invoke(final_prompt)
 
print(response.content)