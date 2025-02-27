from openai import OpenAI
import os
import os
from dotenv import load_dotenv  #  Import dotenv
#from lang-chain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
#  Load API Keys from .env File
load_dotenv()

print("DeepSeek API Key:", os.getenv("DEEP_SEEK_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("Hugging Face Token:", os.getenv("HF_TOKEN"))
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key= os.getenv("DEEP_SEEK_API_KEY"), base_url="https://api.deepseek.com")
models=["deepseek-reasoner", "deepseek-chat",  "gpt-3.5-turbo", "gpt-4o","anthropic:claude-3-5-sonnet-20241022", "ollama:llama3.1:8b", "groq:llama3-70b-8192", "groq:llama-3.2-3b-preview", "huggingface:deepseek-ai/DeepSeek-V3", "huggingface:MaziyarPanahi/calme-3.2-instruct-78b", "huggingface:mistralai/Mistral-7B-Instruct-v0.3"]
try:
    response = client.chat.completions.create(

        model=models[1],
        messages=[{"role": "system", "content": "Hello, are you working?"}]
    )
    print("✅ API Key is working!")
    print(response.choices[0].message.content)
except Exception as e:
    print("❌ ERROR: Invalid API Key or Authentication Issue!")
    print(str(e))