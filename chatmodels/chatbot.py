import os

from dotenv import load_dotenv
load_dotenv()


from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage    

model=init_chat_model("groq:openai/gpt-oss-120b",temperature=0.9)

print("CHOOSE YOUR MODEL")
print("PRESS 1 FOR ANGRY MODE")
print("PRESS 2 FOR FUNNY MODE")
print("PRESS 3 FOR SAD MODE")

choice=input("Enter your choice: ")

if choice == "1":
    mode = SystemMessage(content="You are an Angry AI agent that is helpful and provides information in an angry way.")
elif choice == "2":
    mode = SystemMessage(content="You are a Funny AI agent that is helpful and provides information in a humorous way.")
elif choice == "3":
    mode = SystemMessage(content="You are a Sad AI agent that is helpful and provides information in a sad way.")
else:
    print("Invalid choice. Exiting.")
    exit()

messages=[
    SystemMessage(content=mode.content)
]

print("--------------------------Welcome type 0 to exit the application------------------------------")
while(True):
    prompt=input("Enter your prompt: ")
    messages.append(HumanMessage(content=prompt))
    if(prompt=="0"):
        print("Exiting the application")
        break
    
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot:",response.content)

