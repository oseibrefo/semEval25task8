import os
import json
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import os
import os
from dotenv import load_dotenv  # Import dotenv
#from lang-chain.chat_models import ChatOpenAI

#  Load API Keys from .env File
load_dotenv()

print("DeepSeek API Key:", os.getenv("DEEP_SEEK_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("Hugging Face Token:", os.getenv("HF_TOKEN"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#client = OpenAI(api_key=os.getenv("DEEP_SEEK_API_KEY", base_url="https://api.deepseek.com")
models=["deepseek-reasoner", "deepseek-chat",  "gpt-3.5-turbo", "gpt-4o","anthropic:claude-3-5-sonnet-20241022", "ollama:llama3.1:8b", "groq:llama3-70b-81"]
class InferenceAgent:
    def __init__(self, model_name=models[3]):
        """
        Initializes the Inference Agent with a preferred model.
        """
        #self.api_key = os.getenv("DEEP_SEEK_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(" ERROR: Missing API Key. Set DEEP_SEEK_API_KEY in environment variables.")

        self.model_name = model_name
        self.llm = ChatOpenAI(
            temperature=0.0,
            model_name=self.model_name,
            api_key=self.api_key
            #base_url="https://api.deepseek.com"
        )

    def construct_prompt(self, question, df, row_id):
        """
        Constructs a structured prompt including schema and sample rows.
        """
        prompt_text = """
        You are a Python expert analyzing tabular data step by step.

        Your final answer MUST be in Python code:
        - Put your code in triple backticks: ``` ... ```
        - The code must define exactly ONE function named answer(data).
        - The function returns a single line: return ...
        - The return type is exactly ONE of:
            * boolean: True/False or "Yes"/"No"
            * a single string or number
            * a list of strings or numbers e.g. ["cat","dog"] or [1,2]
        - DO NOT include ANY extra lines outside the triple-quoted block.
        - DO NOT include ANY additional text outside the triple-quoted code block.
        -Output an answer for each question irrespective of the outcome, making the number of questions exactly equal to the number of answers. If any error is encountered while doing that replace it with a placeholder, " Cant generate an answer"

        Use only these sample rows and columns. Then produce your code.

        Question ID: {row_id}
        Question: {question}
        Relevant Columns: {relevant_columns}
        Sample Rows: {sample_rows}

        Generate only valid Python code in triple backticks:
        """

        relevant_columns = ", ".join(df.columns)
        sample_rows = json.dumps(df.head(5).to_dict(orient="records"), indent=4)  # JSON-safe formatting

        prompt_template = PromptTemplate(
            template=prompt_text,
            input_variables=["question", "row_id", "relevant_columns", "sample_rows"]
        )

        return prompt_template.format(
            question=question,
            row_id=row_id,
            relevant_columns=relevant_columns,
            sample_rows=sample_rows
        )

    def generate_code(self, question, df, row_id):
        """
        Queries the LLM with the structured prompt to generate Python code.
        """
        prompt = self.construct_prompt(question, df, row_id)

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip() if response else "__INFERENCE_ERROR__: Empty response"
        except Exception as e:
            return f"__INFERENCE_ERROR__: {e}"
