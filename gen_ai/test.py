import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# Load env variables
load_dotenv()

# Initialize Mistral model
llm = ChatMistralAI(
    model="mistral-small",   # fast & cheap
    temperature=0.9,
    max_tokens=50
)

# Invoke model
response = llm.invoke("Explain RAG in simple words")

# Print output
print(response.content)

# 7111452a-4573-4739-9f90-76f80889882b