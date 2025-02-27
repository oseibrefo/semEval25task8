import csv
import json
import tqdm
import pandas as pd
from pathlib import Path
from datasets import Dataset
import zipfile
import os

# Importing Agents
from agents.dataset_extraction import DatasetExtractionAgent
from agents.schema_agent import SchemaAgent
from agents.prompt_engineer_agent import PromptEngineeringAgent
from agents.code_generation_agent import CodeGenerationAgent
from agents.code_execution_agent import ExecutionAgent
from agents.predictions_agent import PredictionAgent
from agents.inference_agent import InferenceAgent

TOTAL_QUESTIONS = 522  # Ensure exactly 522 answers are stored


def normalize_path(path):
    """
    Normalize file paths to always use forward slashes (/),
    ensuring consistency across Windows & Unix systems.
    """
    return str(Path(path).absolute()).replace("\\", "/")  # Converts ALL backslashes to forward slashes


def main():
    # Initialize agents
    dataset_agent = DatasetExtractionAgent()
    schema_agent = SchemaAgent()
    prompt_agent = PromptEngineeringAgent()
    code_agent = CodeGenerationAgent()
    execution_agent = ExecutionAgent()
    prediction_agent = PredictionAgent()
    inference_agent = InferenceAgent()

    # Step 1: Extract dataset
    extracted_folder = dataset_agent.extract_data("competition.zip", "./competition")
    extracted_folder = normalize_path(extracted_folder)  # Fix path format

    # Step 2: Load test QA file
    test_qa_path = normalize_path(f"{extracted_folder}/competition/test_qa.csv")

    if not os.path.exists(test_qa_path):
        print(f" ERROR: File not found! Expected at: {test_qa_path}")
        return  # Exit if file is missing

    # Load CSV into a list of dictionaries
    test_qa = pd.read_csv(test_qa_path)
    test_qa_list = test_qa.to_dict(orient="records")

    # Ensure predictions lists have exactly 522 placeholders
    predictions = ["__INFERENCE_ERROR__: No answer generated"] * TOTAL_QUESTIONS
    predictions_lite = ["__INFERENCE_ERROR__: No answer generated"] * TOTAL_QUESTIONS

    # Step 3: Process each question
    for index, qa in tqdm.tqdm(enumerate(test_qa_list), total=len(test_qa_list), desc="Processing Questions"):
        try:
            question = qa.get("question", f"Unknown Question {index+1}")
            dataset = qa.get("dataset", "unknown_dataset")
            row_id = qa.get("id", str(index + 1))

            print(f"\n🔍 Processing ({index+1}/522): {question}")

            # Step 4: Load FULL dataset
            dataset_path_full = normalize_path(f"{extracted_folder}/competition/{dataset}/all.parquet")

            if os.path.exists(dataset_path_full):
                df_full = pd.read_parquet(dataset_path_full)
                schema_info_full = schema_agent.load_schema(extracted_folder, dataset, lite=False)
                if schema_info_full:
                    code_full = inference_agent.generate_code(question, df_full, row_id)
                    executed_result_full = execution_agent.execute_code(code_full, schema_info_full.get("sample_rows", []))
                    predictions[index] = str(executed_result_full)  # Ensure output is a string
                else:
                    print(f"⚠ Skipping {question} - Full dataset schema missing.")
                    predictions[index] = "__INFERENCE_ERROR__: Full dataset schema missing"

            else:
                predictions[index] = "__INFERENCE_ERROR__: Full dataset not found"

            # Step 5: Load LITE dataset
            dataset_path_lite = normalize_path(f"{extracted_folder}/competition/{dataset}/sample.parquet")

            if os.path.exists(dataset_path_lite):
                df_lite = pd.read_parquet(dataset_path_lite)
                schema_info_lite = schema_agent.load_schema(extracted_folder, dataset, lite=True)
                if schema_info_lite:
                    code_lite = inference_agent.generate_code(question, df_lite, row_id)
                    executed_result_lite = execution_agent.execute_code(code_lite, schema_info_lite.get("sample_rows", []))
                    predictions_lite[index] = str(executed_result_lite)  # Ensure output is a string
                else:
                    print(f"⚠ Skipping {question} - Lite dataset schema missing.")
                    predictions_lite[index] = "__INFERENCE_ERROR__: Lite dataset schema missing"

            else:
                predictions_lite[index] = "__INFERENCE_ERROR__: Lite dataset not found"

        except Exception as e:
            print(f" ERROR processing question '{question}': {e}")
            predictions[index] = f"__INFERENCE_ERROR__: {str(e)}"
            predictions_lite[index] = f"__INFERENCE_ERROR__: {str(e)}"

    # Ensure predictions lists are exactly 522 answers
    if len(predictions) != TOTAL_QUESTIONS:
        print(f" ERROR: Only {len(predictions)} predictions found! Filling missing slots.")
        while len(predictions) < TOTAL_QUESTIONS:
            predictions.append("__INFERENCE_ERROR__: Missing Answer")
        while len(predictions_lite) < TOTAL_QUESTIONS:
            predictions_lite.append("__INFERENCE_ERROR__: Missing Answer")

    # Step 6: Save results
    with open("predictions.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
    print(f" Saved {TOTAL_QUESTIONS} answers to predictions.txt")

    with open("predictions_lite.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(predictions_lite))
    print(f" Saved {TOTAL_QUESTIONS} answers to predictions_lite.txt")

    # Step 7: Zip the predictions into 'archive.zip'
    with zipfile.ZipFile("archive.zip", "w") as zipf:
        zipf.write("predictions.txt")
        zipf.write("predictions_lite.txt")

    print("Created archive.zip with predictions.")


if __name__ == "__main__":
    main()
