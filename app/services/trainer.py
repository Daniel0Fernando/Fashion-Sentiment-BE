<<<<<<< HEAD
# app/services/trainer.py
import os
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import Config
import torch
import pandas as pd
import numpy as np
import shap
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import traceback # Import for detailed error reporting

# --- Create a custom Dataset class ---
class FashionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Ensure all encoding keys are included and are tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Ensure labels are tensors
        return item

    def __len__(self):
        # Use the length of the input_ids tensor
        return len(self.encodings['input_ids'])
# -------------------------------------

def train_model():
    """Trains a FinBERT-based sentiment analysis model from the combined CSV."""
    from app.utils.helpers import preprocess_for_finbert # Local import

    # --- Load Combined Data ---
    combined_csv_path = Config.COMBINED_DATASET_FILE
    try:
        df = pd.read_csv(combined_csv_path)
        print(f"DEBUG: Loaded {len(df)} rows from {combined_csv_path}")
    except FileNotFoundError:
        print(f"ERROR: Combined dataset file not found at {combined_csv_path}. Run the initial train.py first.")
        return "Combined data file not found"
    except pd.errors.EmptyDataError:
         print(f"ERROR: Combined dataset file is empty at {combined_csv_path}.")
         return "Combined data file is empty"
    except Exception as e:
         print(f"Error loading combined CSV: {e}")
         return "Error loading data"

    if df.empty:
        print("ERROR: Combined DataFrame is empty. Check the data loading in train.py.")
        return "No data to train"

    if 'text' not in df.columns or 'brand' not in df.columns:
        print("ERROR: Combined CSV must have 'text' and 'brand' columns.")
        return "Missing required columns"

    # --- Prepare Data ---
    df['text'] = df['text'].astype(str) # Ensure text column is string type
    df.dropna(subset=['text'], inplace=True) # Drop rows where 'text' is still NaN after conversion
    df['label'] = df['text'].apply(assign_label)

    # --- Optionally limit dataframe for faster testing ---
    # df = limit_dataframe(df)
    # --- ------------------------------------------ ---
    print(f"DEBUG: Training with {len(df)} rows.")

    if len(df) == 0:
        print("ERROR: DataFrame is empty after processing. Check assign_label or data.")
        return "No data after processing"

    # --- Model Training ---
    try:
        model_name = Config.MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # 3 labels: pos, neg, neutral

        # --- Prepare Datasets ---
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        # Split data *before* encoding
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels) # Added stratify

        if not train_texts or not val_texts:
             print("ERROR: Not enough data to create train/validation splits.")
             return "Data splitting failed"

        # Encode datasets separately
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        # Create Dataset objects
        train_dataset = FashionDataset(train_encodings, train_labels)
        val_dataset = FashionDataset(val_encodings, val_labels)
        # -------------------------

        # --- Training Arguments ---
        training_args_dict = {
            'output_dir': Config.TRAINED_MODELS_DIR,
            'num_train_epochs': 3,  # Adjust as needed
            'per_device_train_batch_size': 8, # Adjust based on memory
            'per_device_eval_batch_size': 8,  # Adjust based on memory
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': './logs',
            'logging_steps': 10, # Log more frequently
            'evaluation_strategy': "epoch", # Evaluate at the end of each epoch
            'save_strategy': "epoch", # Save at the end of each epoch
            'load_best_model_at_end': True, # Load the best model found during training
        }
        training_args_obj = TrainingArguments(**training_args_dict)

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args_obj,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # --- Train ---
        print("Starting training...")
        trainer.train()
        print("Training finished.")

        # --- Evaluate ---
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation Results: {eval_results}")

        # --- Get Predictions for Classification Report ---
        predictions_output = trainer.predict(test_dataset=val_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)
        evaluate_report(predictions, val_labels) # Use the direct labels

        # --- Model Saving (Best model is already loaded if load_best_model_at_end=True) ---
        model_path = os.path.join(Config.TRAINED_MODELS_DIR, 'finbert_model_best') # Save best model explicitly
        trainer.save_model(model_path) # Saves the best model loaded at the end
        tokenizer.save_pretrained(model_path)
        print(f"Best model and tokenizer saved to {model_path}")

        # --- SHAP Explainer Saving (FIXED) ---
        try:
            print("Creating and saving SHAP explainer...")
            # Re-load the pipeline using the *saved* best model path
            classifier_for_shap = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=-1) # Use device 0 for GPU

            # --- FIX: Explicitly create a Text masker with the tokenizer ---
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(classifier_for_shap, masker)
            # -------------------------------------------------------------

            explainer_path = os.path.join(Config.TRAINED_MODELS_DIR, 'shap_explainer.joblib')
            joblib.dump(explainer, explainer_path)
            print(f"SHAP explainer saved to {explainer_path}")
        except Exception as shap_error:
            print(f"Could not create/save SHAP explainer: {shap_error}")
            traceback.print_exc() # Print traceback for SHAP errors

        return "Training Complete"

    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc() # Print full traceback for detailed error
        return f"Training failed: {e}"

# --- Helper Functions ---

def assign_label(text):
    """Assigns a sentiment label based on keywords."""
    text = str(text).lower() # Ensure text is string and lowercase
    # More refined keywords
    positive_keywords = ['sustainable', 'eco-friendly', 'ethical', 'good quality', 'great quality', 'high-quality', 'impressed', 'love it', 'fantastic', 'excellent', 'comfortable', 'durable', 'highly recommend', 'perfect fit']
    negative_keywords = ['fast fashion', 'poor quality', 'unethical', 'low quality', 'horrendous quality', 'cheap', 'disappointed', 'tear', 'fade', 'shrink', 'bad customer service', 'worst', 'overpriced', 'problem', 'issue', 'damaged', 'not worth']

    is_positive = any(keyword in text for keyword in positive_keywords)
    is_negative = any(keyword in text for keyword in negative_keywords)

    if is_positive and not is_negative:
        return 0 # Positive
    elif is_negative and not is_positive:
        return 1 # Negative
    elif is_negative: # Prioritize negative if both are present
        return 1
    else:
        return 2 # Neutral

def limit_dataframe(df):
    """Limits the number of texts per label for faster testing."""
    max_texts_per_label = 100 # Adjust this number as needed for testing vs full training
    print(f"DEBUG: Limiting dataframe to {max_texts_per_label} per label.")
    # Ensure the label column exists and handle potential missing labels if grouping fails
    if 'label' in df.columns:
        return df.groupby('label', group_keys=False).apply(lambda x: x.head(max_texts_per_label)).reset_index(drop=True)
    else:
        print("WARNING: 'label' column not found for limiting dataframe.")
        return df.head(max_texts_per_label * 3) # Fallback to limiting total rows


def evaluate_report(predictions, labels): # Renamed to avoid conflict
    """Calculates and prints classification report."""
    try:
        # Ensure labels and predictions are aligned and valid
        if len(predictions) != len(labels):
             print("ERROR: Predictions and labels have different lengths.")
             return

        print("\nClassification Report:")
        # Define target names based on your labels
        target_names = ['Positive', 'Negative', 'Neutral']
        # Make sure labels are within the expected range [0, 1, 2]
        valid_labels = all(0 <= lbl <= 2 for lbl in labels)
        valid_preds = all(0 <= p <= 2 for p in predictions)

        if valid_labels and valid_preds:
             report = classification_report(labels, predictions, target_names=target_names, zero_division=0)
             print(report)
        else:
             print("WARN: Invalid values found in labels or predictions. Cannot generate full report.")
             print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")


    except Exception as e:
        print(f"Error generating classification report: {e}")


def is_model_trained():
    """Checks if a trained FinBERT model exists."""
    # Check for the 'best' model saved by the trainer
    model_path = os.path.join(Config.TRAINED_MODELS_DIR, 'finbert_model_best')
    # More robust check: look for config file or pytorch model file
    config_file = os.path.join(model_path, 'config.json')
    model_file = os.path.join(model_path, 'pytorch_model.bin') # Or model.safetensors
=======
# app/services/trainer.py
import os
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import Config
import torch
import pandas as pd
import numpy as np
import shap
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import traceback # Import for detailed error reporting

# --- Create a custom Dataset class ---
class FashionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Ensure all encoding keys are included and are tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Ensure labels are tensors
        return item

    def __len__(self):
        # Use the length of the input_ids tensor
        return len(self.encodings['input_ids'])
# -------------------------------------

def train_model():
    """Trains a FinBERT-based sentiment analysis model from the combined CSV."""
    from app.utils.helpers import preprocess_for_finbert # Local import

    # --- Load Combined Data ---
    combined_csv_path = Config.COMBINED_DATASET_FILE
    try:
        df = pd.read_csv(combined_csv_path)
        print(f"DEBUG: Loaded {len(df)} rows from {combined_csv_path}")
    except FileNotFoundError:
        print(f"ERROR: Combined dataset file not found at {combined_csv_path}. Run the initial train.py first.")
        return "Combined data file not found"
    except pd.errors.EmptyDataError:
         print(f"ERROR: Combined dataset file is empty at {combined_csv_path}.")
         return "Combined data file is empty"
    except Exception as e:
         print(f"Error loading combined CSV: {e}")
         return "Error loading data"

    if df.empty:
        print("ERROR: Combined DataFrame is empty. Check the data loading in train.py.")
        return "No data to train"

    if 'text' not in df.columns or 'brand' not in df.columns:
        print("ERROR: Combined CSV must have 'text' and 'brand' columns.")
        return "Missing required columns"

    # --- Prepare Data ---
    df['text'] = df['text'].astype(str) # Ensure text column is string type
    df.dropna(subset=['text'], inplace=True) # Drop rows where 'text' is still NaN after conversion
    df['label'] = df['text'].apply(assign_label)

    # --- Optionally limit dataframe for faster testing ---
    # df = limit_dataframe(df)
    # --- ------------------------------------------ ---
    print(f"DEBUG: Training with {len(df)} rows.")

    if len(df) == 0:
        print("ERROR: DataFrame is empty after processing. Check assign_label or data.")
        return "No data after processing"

    # --- Model Training ---
    try:
        model_name = Config.MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # 3 labels: pos, neg, neutral

        # --- Prepare Datasets ---
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        # Split data *before* encoding
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels) # Added stratify

        if not train_texts or not val_texts:
             print("ERROR: Not enough data to create train/validation splits.")
             return "Data splitting failed"

        # Encode datasets separately
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        # Create Dataset objects
        train_dataset = FashionDataset(train_encodings, train_labels)
        val_dataset = FashionDataset(val_encodings, val_labels)
        # -------------------------

        # --- Training Arguments ---
        training_args_dict = {
            'output_dir': Config.TRAINED_MODELS_DIR,
            'num_train_epochs': 3,  # Adjust as needed
            'per_device_train_batch_size': 8, # Adjust based on memory
            'per_device_eval_batch_size': 8,  # Adjust based on memory
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': './logs',
            'logging_steps': 10, # Log more frequently
            'evaluation_strategy': "epoch", # Evaluate at the end of each epoch
            'save_strategy': "epoch", # Save at the end of each epoch
            'load_best_model_at_end': True, # Load the best model found during training
        }
        training_args_obj = TrainingArguments(**training_args_dict)

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args_obj,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # --- Train ---
        print("Starting training...")
        trainer.train()
        print("Training finished.")

        # --- Evaluate ---
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation Results: {eval_results}")

        # --- Get Predictions for Classification Report ---
        predictions_output = trainer.predict(test_dataset=val_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)
        evaluate_report(predictions, val_labels) # Use the direct labels

        # --- Model Saving (Best model is already loaded if load_best_model_at_end=True) ---
        model_path = os.path.join(Config.TRAINED_MODELS_DIR, 'finbert_model_best') # Save best model explicitly
        trainer.save_model(model_path) # Saves the best model loaded at the end
        tokenizer.save_pretrained(model_path)
        print(f"Best model and tokenizer saved to {model_path}")

        # --- SHAP Explainer Saving (FIXED) ---
        try:
            print("Creating and saving SHAP explainer...")
            # Re-load the pipeline using the *saved* best model path
            classifier_for_shap = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=-1) # Use device 0 for GPU

            # --- FIX: Explicitly create a Text masker with the tokenizer ---
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(classifier_for_shap, masker)
            # -------------------------------------------------------------

            explainer_path = os.path.join(Config.TRAINED_MODELS_DIR, 'shap_explainer.joblib')
            joblib.dump(explainer, explainer_path)
            print(f"SHAP explainer saved to {explainer_path}")
        except Exception as shap_error:
            print(f"Could not create/save SHAP explainer: {shap_error}")
            traceback.print_exc() # Print traceback for SHAP errors

        return "Training Complete"

    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc() # Print full traceback for detailed error
        return f"Training failed: {e}"

# --- Helper Functions ---

def assign_label(text):
    """Assigns a sentiment label based on keywords."""
    text = str(text).lower() # Ensure text is string and lowercase
    # More refined keywords
    positive_keywords = ['sustainable', 'eco-friendly', 'ethical', 'good quality', 'great quality', 'high-quality', 'impressed', 'love it', 'fantastic', 'excellent', 'comfortable', 'durable', 'highly recommend', 'perfect fit']
    negative_keywords = ['fast fashion', 'poor quality', 'unethical', 'low quality', 'horrendous quality', 'cheap', 'disappointed', 'tear', 'fade', 'shrink', 'bad customer service', 'worst', 'overpriced', 'problem', 'issue', 'damaged', 'not worth']

    is_positive = any(keyword in text for keyword in positive_keywords)
    is_negative = any(keyword in text for keyword in negative_keywords)

    if is_positive and not is_negative:
        return 0 # Positive
    elif is_negative and not is_positive:
        return 1 # Negative
    elif is_negative: # Prioritize negative if both are present
        return 1
    else:
        return 2 # Neutral

def limit_dataframe(df):
    """Limits the number of texts per label for faster testing."""
    max_texts_per_label = 100 # Adjust this number as needed for testing vs full training
    print(f"DEBUG: Limiting dataframe to {max_texts_per_label} per label.")
    # Ensure the label column exists and handle potential missing labels if grouping fails
    if 'label' in df.columns:
        return df.groupby('label', group_keys=False).apply(lambda x: x.head(max_texts_per_label)).reset_index(drop=True)
    else:
        print("WARNING: 'label' column not found for limiting dataframe.")
        return df.head(max_texts_per_label * 3) # Fallback to limiting total rows


def evaluate_report(predictions, labels): # Renamed to avoid conflict
    """Calculates and prints classification report."""
    try:
        # Ensure labels and predictions are aligned and valid
        if len(predictions) != len(labels):
             print("ERROR: Predictions and labels have different lengths.")
             return

        print("\nClassification Report:")
        # Define target names based on your labels
        target_names = ['Positive', 'Negative', 'Neutral']
        # Make sure labels are within the expected range [0, 1, 2]
        valid_labels = all(0 <= lbl <= 2 for lbl in labels)
        valid_preds = all(0 <= p <= 2 for p in predictions)

        if valid_labels and valid_preds:
             report = classification_report(labels, predictions, target_names=target_names, zero_division=0)
             print(report)
        else:
             print("WARN: Invalid values found in labels or predictions. Cannot generate full report.")
             print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")


    except Exception as e:
        print(f"Error generating classification report: {e}")


def is_model_trained():
    """Checks if a trained FinBERT model exists."""
    # Check for the 'best' model saved by the trainer
    model_path = os.path.join(Config.TRAINED_MODELS_DIR, 'finbert_model_best')
    # More robust check: look for config file or pytorch model file
    config_file = os.path.join(model_path, 'config.json')
    model_file = os.path.join(model_path, 'pytorch_model.bin') # Or model.safetensors
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
    return os.path.exists(model_path) and (os.path.exists(config_file) or os.path.exists(model_file))