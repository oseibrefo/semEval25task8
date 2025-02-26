from langchain.prompts import PromptTemplate

class PromptEngineeringAgent:
    def construct_prompt(self, question, schema_info):
        """Constructs a structured prompt including schema and sample rows."""
        if schema_info is None:
            print(" Error: Schema info is missing! Returning fallback prompt.")
            return f"Cannot process question: {question} due to missing schema."

        prompt = f"""
        You are a Python expert analyzing tabular data step by step. Use the following schema:
        
        - Columns: {", ".join(schema_info["columns"])}
        - Sample Rows: {schema_info["sample_rows"]}

        Question: {question}

        Generate valid Python code inside triple backticks, **without any explanation**.
        Your code should be:

        Your final answer MUST be in Python code:
        - Put your code in triple backticks: ``` ... ```
        - The code must define exactly ONE function named answer(data).
        - The function returns a single line: return ...
        - The return type is exactly ONE of:
            * boolean: True/False or "Yes"/"No"
            * a single string or number
            * a list of strings or numbers e.g. ["cat","dog"] or [1,2]
        - DO NOT include ANY extra lines outside the triple-quoted block.
        - DO NOT include ANY additional text outside the triple-quoted code block

        Example:
        ```
        def answer(data):
            return len(data)  # Example answer
        ```
        """
        return prompt

