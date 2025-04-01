<<<<<<< HEAD
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Basic Application Settings ---
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key'

    # --- Data Directories ---
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
    TRAINED_MODELS_DIR = os.path.join(DATA_DIR, 'models')
    COMBINED_DATASET_FILE = os.path.join(DATA_DIR, 'combined_fashion_data.csv') # Path for combined data

    # --- Database Configuration (Keep for potential future use) ---
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Celery Configuration ---
    CELERY = {
        'broker_url': os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0',
        'result_backend': os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0',
    }

    # --- Other Model Settings ---
    MODEL_NAME = "ProsusAI/finbert"

# --- Create Directories ---
os.makedirs(Config.DATA_DIR, exist_ok=True)
=======
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Basic Application Settings ---
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key'

    # --- Data Directories ---
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
    TRAINED_MODELS_DIR = os.path.join(DATA_DIR, 'models')
    COMBINED_DATASET_FILE = os.path.join(DATA_DIR, 'combined_fashion_data.csv') # Path for combined data

    # --- Database Configuration (Keep for potential future use) ---
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Celery Configuration ---
    CELERY = {
        'broker_url': os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0',
        'result_backend': os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0',
    }

    # --- Other Model Settings ---
    MODEL_NAME = "ProsusAI/finbert"

# --- Create Directories ---
os.makedirs(Config.DATA_DIR, exist_ok=True)
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
os.makedirs(Config.TRAINED_MODELS_DIR, exist_ok=True)