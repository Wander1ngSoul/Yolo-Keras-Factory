import os
from dotenv import load_dotenv

load_dotenv()

class FineTuningConfig:
    BASE_MODEL_PATH = os.getenv('OCR_MODEL_PATH')
    ORIGINAL_DATASET_DIR = os.getenv('DATASET_DIR_PATH_ORIGIN')
    NEW_DATASET_DIR = os.getenv('NEW_DATASET_DIR')
    RESULTS_BASE_DIR = r"C:\keras_recognition\Models\Recognition"