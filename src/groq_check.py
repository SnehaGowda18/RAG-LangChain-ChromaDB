import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
)

response = llm.invoke(
    "What is Retrieval-Augmented Generation? Explain in one sentence."
)

print("Groq connection successful!")
print("Response:")
print(response.content)