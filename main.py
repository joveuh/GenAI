import names, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

names = names.names
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer(f"Following is a list of a person's given names.{str(names)}. Describe each name in a few words (less than 5). What feeling or emotions does it invoke?"
, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)


