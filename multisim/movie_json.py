from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional
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

prompt=ChatPromptTemplate.from_messages([
    ('system',"""
    Extract movie information from the pargraph
     {format_instructions}
"""),
("human","{paragraph}")]
)

para= input("give your paragraph")

final_prompt=prompt.invoke(
    {"paragraph":para,
     'format_instructions':parser.get_format_instructions()
     }
)


response=model.invoke(final_prompt)
 
print(response.content)