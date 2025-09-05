import os
from dotenv import load_dotenv

load_dotenv()


class RecognitionConfig:
    DATASET_DIR = os.getenv('DATASET_DIR_PATH_ORIGIN')
    NEW_DATASET_DIR = os.getenv('NEW_DATASET_DIR')
    OCR_MODEL_PATH = os.getenv('OCR_MODEL_PATH')
    IMAGE_SIZE = (20, 32)
    NUM_CLASSES = 10

    # Настройки base training
    BASE_BATCH_SIZE = 32
    BASE_EPOCHS = 75
    BASE_MODEL_SAVE_PATH = "meter_recognition.keras"

    # Настройки fine tuning
    FT_BATCH_SIZE = 16
    FT_EPOCHS = 20
    FT_RESULTS_DIR = r"C:\keras_recognition\Models\Recognition"