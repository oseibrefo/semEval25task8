import os
import csv
import json
import tqdm
import os
import pandas as pd

class SchemaAgent:
    def load_schema(self, extracted_folder,  dataset, lite=False):
        """Loads dataset files and extracts relevant columns & sample rows."""
        filename = "sample.parquet" if lite else "all.parquet"
        file_path = os.path.join(extracted_folder, 'competition', dataset, filename)
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return None

        df = pd.read_parquet(file_path)
        relevant_columns = df.columns.tolist()
        sample_rows = df.head(5).to_dict(orient="records")

        return {"columns": relevant_columns, "sample_rows": sample_rows}
