from openai import OpenAI
import os
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Retrieve the secret keys
deepseek_api_key = os.getenv("DEEP_SEEK_API_KEY")
hf_token = os.getenv("HF_TOKEN")



class CodeGenerationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("DEEP_SEEK_API_KEY"), base_url="https://api.deepseek.com")

    def generate_code(self, prompt, model="deepseek-reasoner"):
        """Queries DeepSeek LLM to generate Python code"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
