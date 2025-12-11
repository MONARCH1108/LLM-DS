import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def query_groq_llm(
    user_input: str, model: str = "llama-3.3-70b-versatile", system_prompt: str = "", temperature: float = 0.2, ) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GROQ_API_KEY environment variable.")
    client = Groq(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


