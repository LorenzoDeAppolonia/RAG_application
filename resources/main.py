import os

from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_openai import ChatOpenAI

from playground_secret_key import SECRET_KEY

os.environ['OPEN_API_KEY'] = SECRET_KEY
chat = ChatOpenAI(
    openai_api_key = os.environ['OPEN_API_KEY'],
    model = 'gpt-3.5-turbo'
)

messages = [
    SystemMessage(content='You are a tutor that helps highschool students.'),
    HumanMessage(content='Hi tutor, how are you today?'),
    AIMessage(content='I am great, thank you, how can I help you today?.'),
    HumanMessage(content='I would like you to explain to me second order derivatives')
]

# TODO : to have chat history you append both the AI response and the new prompt to the messages list

res = chat.invoke(messages)
messages.append(res)
prompt = HumanMessage(content='How does is this used in finding maxima and minima of a function')
messages.append(prompt)
res = chat.invoke(messages)
print(res.content)
