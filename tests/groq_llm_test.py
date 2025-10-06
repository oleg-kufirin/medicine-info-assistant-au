from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env file

api_key = os.getenv("GROQ_API_KEY")
print("API Key:", api_key)

model = os.getenv("SAFETY_MODEL", "llama-3.1-8b-instant") # or llama-3.3-70b-versatile, gemma-7b-it, etc.
print("Model:", model)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name=model,
    temperature=0,
)

# Run a single query
resp = llm.invoke("What is the capital of France?")
print(resp.content)
