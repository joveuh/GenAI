import json, names
from transformers import pipeline

names = names.names

messages = [
    {"role": "system", "content": "I will give you a few full names, and you will tell me how does each name sounds by describing it in a few words (less than 5). Then you will tell me what emotion or feeling does it invoke."}
]
for name in names:
    message = {"role": "user", "content": f"name"}
    messages.append(message)

chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
chatbot(messages)

