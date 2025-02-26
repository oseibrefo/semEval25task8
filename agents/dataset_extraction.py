import os
import zipfile

class DatasetExtractionAgent:
    def extract_data(self, zip_path, extract_to="./competition"):
        """Extracts dataset from ZIP file."""
        if os.path.isdir(extract_to) and os.listdir(extract_to):
            print(f"{extract_to} exists. Skipping extraction.")
            return extract_to

        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted data to: {extract_to}")
        return extract_to
