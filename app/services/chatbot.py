# app/services/chatbot.py
import joblib
import os
from config import Config
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap
import numpy as np
import pandas as pd
import traceback
import re # Make sure re is imported

# --- Configuration for SHAP ---
SHAP_MAX_EVALS = 1024 # Adjust this value (e.g., 128, 256, 512) for speed vs accuracy
MODEL_MAX_LENGTH = 1024
# -----------------------------

# --- Global variables for lazy loading ---
loaded_model = None
loaded_tokenizer = None
loaded_explainer = None
shap_init_error = None
# ---------------------------------------

def load_model_and_tokenizer(device_map='cpu'):
    """Loads the trained FinBERT model and tokenizer, mapping to specified device."""
    global loaded_model, loaded_tokenizer
    if loaded_model is not None and loaded_tokenizer is not None:
        print("DEBUG: Model and tokenizer already loaded.")
        return loaded_model, loaded_tokenizer

    model_path = os.path.join(Config.TRAINED_MODELS_DIR, 'finbert_model_best')
    if not os.path.exists(model_path):
        print(f"ERROR: Model directory not found at {model_path}")
        return None, None

    try:
        print(f"Loading model and tokenizer from {model_path} onto {device_map}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        loaded_model = model
        loaded_tokenizer = tokenizer
        print("Model and tokenizer loaded successfully.")
        return loaded_model, loaded_tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        traceback.print_exc()
        return None, None

def load_or_create_explainer(force_cpu=False):
    """
    Loads the SHAP explainer from file. If loading fails or if it doesn't exist,
    attempts to create it dynamically using the globally loaded model/tokenizer,
    mapping to CPU if needed.
    """
    global loaded_explainer, loaded_model, loaded_tokenizer, shap_init_error

    if loaded_explainer is not None:
        print("DEBUG: Explainer already loaded/created.")
        return loaded_explainer
    if shap_init_error is not None:
        print(f"DEBUG: SHAP initialization previously failed: {shap_init_error}")
        return None

    explainer_path = os.path.join(Config.TRAINED_MODELS_DIR, 'shap_explainer.joblib')
    explainer = None

    if os.path.exists(explainer_path):
        print(f"Attempting to load SHAP explainer from {explainer_path}...")
        try:
            explainer = joblib.load(explainer_path)
            print("SHAP explainer loaded successfully from file.")
            loaded_explainer = explainer
            return loaded_explainer
        except RuntimeError as e:
            if "Attempting to deserialize object on a CUDA device" in str(e):
                print(f"WARN: Loading saved SHAP explainer failed due to CUDA mismatch. Will attempt to recreate on CPU.")
                explainer = None # Reset to trigger recreation
            else:
                print(f"Error loading SHAP explainer: {e}")
                shap_init_error = e
                return None
        except Exception as e:
             print(f"Error loading SHAP explainer: {e}")
             shap_init_error = e
             return None
    else:
        print(f"WARN: SHAP explainer file not found at {explainer_path}. Will attempt to create.")

    if explainer is None:
        print("Attempting to create SHAP explainer dynamically...")
        model, tokenizer = load_model_and_tokenizer(device_map='cpu')

        if model is None or tokenizer is None:
            error_msg = "Cannot create explainer: Model or tokenizer failed to load."
            print(f"ERROR: {error_msg}")
            shap_init_error = RuntimeError(error_msg)
            return None

        try:
            pipeline_device = 0 if torch.cuda.is_available() and not force_cpu else -1
            print(f"Creating SHAP pipeline on device: {pipeline_device}")
            classifier_for_shap = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device
            )
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(classifier_for_shap, masker)
            print("SHAP explainer created successfully.")
            loaded_explainer = explainer
            return loaded_explainer
        except Exception as e:
            print(f"ERROR: Failed to create SHAP explainer dynamically: {e}")
            traceback.print_exc()
            shap_init_error = e
            return None
#--------------------------------------------------------------------------

def get_top_features(tokens, importances, n=5):
    """Gets the top N features and their SIGNED importances, sorted by ABSOLUTE importance."""
    if not isinstance(importances, (list, np.ndarray)) or (isinstance(importances, np.ndarray) and importances.ndim != 1):
         print(f"WARN: Invalid format for SHAP importances: {type(importances)}")
         return []
    if len(tokens) != len(importances):
         print(f"WARN: Mismatch token/importance count ({len(tokens)} vs {len(importances)}) in get_top_features.")
         min_len = min(len(tokens), len(importances))
         tokens = tokens[:min_len]
         importances = importances[:min_len]
         if min_len == 0: return []

    feature_importance_pairs = [
        (token, float(importance)) # Store the original signed importance
        for token, importance in zip(tokens, importances)
        if token not in ['[CLS]', '[SEP]', '[PAD]'] and '##' not in token # Also exclude subword tokens for cleaner output
    ]
    # Sort by absolute value of importance to get most impactful words
    sorted_features = sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)
    # Return word and its signed value
    return sorted_features[:n]

def format_conversational_explanation(predicted_label_index, top_features_list):
    """
    Generates a conversational explanation string based on SHAP features.
    """
    if not top_features_list:
        return "" # Return empty string if no features

    # Extract just the words, removing potential subword markers for readability
    words = [re.sub(r'^##', '', feature[0]) for feature in top_features_list]
    # Filter out any empty strings that might result from subword removal
    words = [word for word in words if word]

    if not words:
         return "" # Return empty if only subword tokens were present

    # Format words for inclusion in the sentence
    if len(words) == 1:
        word_str = f"the word '{words[0]}'"
    elif len(words) == 2:
        word_str = f"words like '{words[0]}' and '{words[1]}'"
    else: # 3 or more
        # Take unique words in case of duplicates after subword removal
        unique_words = []
        for word in words:
             if word not in unique_words:
                  unique_words.append(word)
        if len(unique_words) == 1: word_str = f"the word '{unique_words[0]}'"
        elif len(unique_words) == 2: word_str = f"words like '{unique_words[0]}' and '{unique_words[1]}'"
        else: word_str = f"words like '{unique_words[0]}', '{unique_words[1]}', and '{unique_words[2]}'"


    # Generate explanation based on predicted sentiment
    if predicted_label_index == 0: # Positive
        return f"The positive sentiment seems influenced by mentions of {word_str}."
    elif predicted_label_index == 1: # Negative
        return f"It looks like the negative sentiment might be related to terms such as {word_str}."
    else: # Neutral
        return f"The sentiment appears neutral or mixed, with terms like {word_str} playing a role."


def get_response(conversation_history):
    from app.utils.helpers import clean_text, preprocess_for_finbert

    force_shap_cpu = not torch.cuda.is_available()
    explainer = load_or_create_explainer(force_cpu=force_shap_cpu)
    model = loaded_model
    tokenizer = loaded_tokenizer

    if model is None or tokenizer is None:
        return "Model is not available. Please ensure training is complete.", "" # Return empty explanation string

    user_message = conversation_history[-1]['content']
    cleaned_message = clean_text(user_message)
    processed_message = preprocess_for_finbert(cleaned_message)

    if not processed_message or not isinstance(processed_message, str) or not processed_message.strip():
        print(f"WARN: Invalid processed_message after preprocessing: '{processed_message}'")
        return "I couldn't process that message properly. Could you please rephrase?", ""

    # Truncate message for SHAP if it's too long
    max_len_for_input = getattr(tokenizer, 'model_max_length', MODEL_MAX_LENGTH) - 2
    truncated_message_for_shap = processed_message[:max_len_for_input * 4] # Char heuristic

    brand_name = extract_brand(user_message)
    response_text = ""
    conv_explanation = ""
    shap_feature_list = []
    runtime_device = 0 if torch.cuda.is_available() else -1

    if brand_name:
        sentiment_index, brand_shap_features = get_brand_sentiment(brand_name, model, tokenizer, explainer, runtime_device)
        shap_feature_list = brand_shap_features # Use the list of tuples
        if sentiment_index is not None:
            if sentiment_index == 0: response_text = f"Based on my analysis, {brand_name} generally has a positive sentiment regarding sustainability."
            elif sentiment_index == 1: response_text = f"My analysis suggests {brand_name} faces some negative sentiment concerning sustainability."
            else: response_text = f"The sentiment towards {brand_name} appears to be neutral on sustainability."
            # Generate conversational explanation
            conv_explanation = format_conversational_explanation(sentiment_index, shap_feature_list)
        else:
            response_text = f"I don't have enough specific information about {brand_name} in the dataset to provide a detailed sentiment analysis."
            conv_explanation = "" # No explanation if no sentiment

    else: # General sentiment analysis
        try:
            classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=runtime_device)
            result = classifier(processed_message)[0] # Use original for prediction
            label_map = {0: "positive", 1: "negative", 2: "neutral"}
            predicted_label_index = 2
            try:
                score = result.get('score', 0)
                if isinstance(score, (int, float)): predicted_label_index = np.argmax([score])
                elif isinstance(score, list) and len(score) == 3: predicted_label_index = np.argmax(score)
            except (TypeError, ValueError): pass

            predicted_label = label_map.get(predicted_label_index, "neutral")

            if predicted_label == "positive": response_text = "The overall sentiment seems positive regarding sustainability."
            elif predicted_label == "negative": response_text = "I'm detecting a negative sentiment related to sustainability concerns."
            else: response_text = "The sentiment appears to be neutral at this time."

            # --- SHAP Explanation Part ---
            if explainer:
                try:
                    if not truncated_message_for_shap.strip():
                         conv_explanation = "Cannot explain empty text."
                    else:
                        print(f"DEBUG: Running SHAP explainer with max_evals={SHAP_MAX_EVALS}...")
                        shap_values = explainer([truncated_message_for_shap], max_evals=SHAP_MAX_EVALS)
                        print("DEBUG: SHAP calculation finished.")

                        feature_importances = None
                        if hasattr(shap_values, 'values') and isinstance(shap_values.values, (list, np.ndarray)) and len(shap_values.values) > 0:
                             shap_val_data = shap_values.values[0]
                             if isinstance(shap_val_data, np.ndarray):
                                  if shap_val_data.ndim == 2 and shap_val_data.shape[1] == 3:
                                      if 0 <= predicted_label_index < 3: feature_importances = shap_val_data[:, predicted_label_index]
                                      else: print(f"WARN: Invalid predicted_label_index {predicted_label_index}")
                                  elif shap_val_data.ndim == 1: feature_importances = shap_val_data
                                  else: print(f"WARN: Unexpected SHAP values array shape: {shap_val_data.shape}")
                             else: print("WARN: Element inside shap_values.values is not a numpy array.")
                        else: print("WARN: SHAP values attribute '.values' is not list/array or empty.")

                        if feature_importances is not None:
                            encoded_input = tokenizer(truncated_message_for_shap, return_tensors='pt', truncation=True, max_length=MODEL_MAX_LENGTH)
                            if encoded_input['input_ids'].nelement() > 0:
                                tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
                                if len(tokens) == len(feature_importances):
                                    shap_feature_list = get_top_features(tokens, feature_importances)
                                    conv_explanation = format_conversational_explanation(predicted_label_index, shap_feature_list)
                                else:
                                    print(f"WARN: Mismatch SHAP lengths. Tokens: {len(tokens)}, Features: {len(feature_importances)}")
                                    conv_explanation = "Explanation generation mismatch."
                            else: conv_explanation = "Could not generate explanation (empty input)."
                        else: conv_explanation = "Explanation feature importance calculation failed."

                except Exception as e:
                    print(f"SHAP explanation error during get_response: {e}")
                    traceback.print_exc()
                    conv_explanation = "Explanation not available due to error."
            else:
                 conv_explanation = "SHAP Explainer not available."
            # --------------------------

        except Exception as pipeline_error:
             print(f"Error during sentiment analysis pipeline: {pipeline_error}")
             traceback.print_exc()
             response_text = "Sorry, I encountered an error analysing that message."
             conv_explanation = ""

    return response_text, conv_explanation # Return text and conversational explanation


# --- Modify get_brand_sentiment to return top_features list ---
def get_brand_sentiment(brand_name, model, tokenizer, explainer, device):
    from app.utils.helpers import preprocess_for_finbert

    combined_csv_path = Config.COMBINED_DATASET_FILE
    try: df = pd.read_csv(combined_csv_path)
    except Exception as e: print(f"Error loading CSV for chatbot: {e}"); return None, []
    if 'brand' not in df.columns or 'text' not in df.columns: return None, []
    brand_data = df[df['brand'].str.lower() == brand_name.lower()]
    if brand_data.empty: return None, []

    combined_text = " ".join(brand_data['text'].astype(str).tolist())
    processed_text = preprocess_for_finbert(combined_text)
    if not processed_text or not isinstance(processed_text, str) or not processed_text.strip(): return None, []

    max_len_for_input = getattr(tokenizer, 'model_max_length', MODEL_MAX_LENGTH) - 2
    truncated_text = processed_text[:max_len_for_input * 4] # Use char heuristic
    if not truncated_text.strip(): return None, []

    top_features_list = [] # Initialize list for results

    try:
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        result = classifier(truncated_text)[0] # Use truncated text
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        predicted_label_index = 2 # Default
        try:
            score = result.get('score', 0)
            if isinstance(score, (int, float)): predicted_label_index = np.argmax([score])
            elif isinstance(score, list) and len(score)==3 : predicted_label_index = np.argmax(score)
        except (TypeError, ValueError): pass

        if explainer:
            try:
                print(f"DEBUG: Running SHAP explainer for brand '{brand_name}' with max_evals={SHAP_MAX_EVALS}...")
                shap_values = explainer([truncated_text], max_evals=SHAP_MAX_EVALS) # Use truncated
                print(f"DEBUG: SHAP calculation finished for brand '{brand_name}'.")

                feature_importances = None
                if hasattr(shap_values, 'values') and isinstance(shap_values.values, (list, np.ndarray)) and len(shap_values.values) > 0:
                     shap_val_data = shap_values.values[0]
                     if isinstance(shap_val_data, np.ndarray):
                          if shap_val_data.ndim == 2 and shap_val_data.shape[1] == 3:
                               if 0 <= predicted_label_index < 3: feature_importances = shap_val_data[:, predicted_label_index]
                               else: print(f"WARN: Invalid predicted_label_index {predicted_label_index} (brand: {brand_name}).")
                          elif shap_val_data.ndim == 1: feature_importances = shap_val_data
                          else: print(f"WARN: Unexpected SHAP values array shape (brand: {brand_name}): {shap_val_data.shape}")
                     else: print(f"WARN: Element inside shap_values.values is not a numpy array (brand: {brand_name}).")
                else: print(f"WARN: SHAP values attribute '.values' is not list/array or empty (brand: {brand_name}).")

                if feature_importances is not None:
                    encoded_input = tokenizer(truncated_text, return_tensors='pt', truncation=True, max_length=MODEL_MAX_LENGTH)
                    if encoded_input['input_ids'].nelement() > 0:
                        tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
                        if len(tokens) == len(feature_importances):
                             top_features_list = get_top_features(tokens, feature_importances) # Store list of tuples
                        else: print(f"WARN: Mismatch SHAP lengths (brand: {brand_name})...")
                    # else: print(...) Handle empty input if needed
                # else: print(...) Handle importance calculation failure

            except Exception as e:
                print(f"SHAP explanation error (brand: {brand_name}): {e}")
                traceback.print_exc()
                # Keep top_features_list empty

        # Return the predicted index and the list of (word, value) tuples
        return predicted_label_index, top_features_list

    except Exception as pipeline_error:
        print(f"Error during brand sentiment pipeline for {brand_name}: {pipeline_error}")
        traceback.print_exc()
        return None, [] # Return None and empty list on error


def extract_brand(message):
    known_brands = ["Carnage", "Emerald", "Crocodile", "Dilly & Carlo", "FOA", "GFlock", "Kelly Felder", "Jezza", "Mimosa", "Pepper Street"]
    message = message.lower()
    found_brands = [brand for brand in known_brands if brand.lower() in message]
    if found_brands:
        return max(found_brands, key=len)
    return None
# --------------------------------------------------------