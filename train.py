<<<<<<< HEAD
# train.py
from app.services.trainer import train_model, is_model_trained
from app.services.celery_tasks import train_model_async
import os
import pandas as pd
import glob
from config import Config

if __name__ == '__main__':
    if not is_model_trained():
        print("Model not found. Preparing data and initiating training...")

        # --- Combine CSV files ---
        all_files = glob.glob(os.path.join(Config.DATA_DIR, '*_Reviews_Final.csv'))
        if not all_files:
            print(f"ERROR: No source CSV files found in {Config.DATA_DIR}")
            exit(1)

        li = []
        expected_columns = ['Brand Name', 'Product Name', 'Review']

        for filename in all_files:
            try:
                df = pd.read_csv(filename, index_col=None, header=0)
                print(f"Reading {filename} - {len(df)} rows")

                # Validate columns
                if not all(col in df.columns for col in expected_columns):
                    print(f"WARNING: Skipping {filename}. Missing required columns: {expected_columns}")
                    continue

                # Rename columns for consistency
                df = df.rename(columns={'Brand Name': 'brand', 'Product Name': 'product', 'Review': 'text'})
                df = df[['brand', 'text']] # Keep only relevant columns

                li.append(df)
            except pd.errors.EmptyDataError:
                print(f"WARNING: Skipping empty file: {filename}")
            except Exception as e:
                print(f"ERROR reading {filename}: {e}")

        if not li:
            print("ERROR: No valid data could be read from any source CSV file.")
            exit(1)

        combined_df = pd.concat(li, axis=0, ignore_index=True)
        combined_df.dropna(subset=['text'], inplace=True) # Drop rows where 'text' is missing

        # Save the combined data
        try:
            combined_df.to_csv(Config.COMBINED_DATASET_FILE, index=False)
            print(f"Combined data saved to {Config.COMBINED_DATASET_FILE}")
        except Exception as e:
             print(f"Error saving combined CSV: {e}")
             exit(1)

        # --- Dispatch Training Task ---
        task = train_model_async.delay()
        print(f"Training task initiated. Task ID: {task.id}")
    else:
=======
# train.py
from app.services.trainer import train_model, is_model_trained
from app.services.celery_tasks import train_model_async
import os
import pandas as pd
import glob
from config import Config

if __name__ == '__main__':
    if not is_model_trained():
        print("Model not found. Preparing data and initiating training...")

        # --- Combine CSV files ---
        all_files = glob.glob(os.path.join(Config.DATA_DIR, '*_Reviews_Final.csv'))
        if not all_files:
            print(f"ERROR: No source CSV files found in {Config.DATA_DIR}")
            exit(1)

        li = []
        expected_columns = ['Brand Name', 'Product Name', 'Review']

        for filename in all_files:
            try:
                df = pd.read_csv(filename, index_col=None, header=0)
                print(f"Reading {filename} - {len(df)} rows")

                # Validate columns
                if not all(col in df.columns for col in expected_columns):
                    print(f"WARNING: Skipping {filename}. Missing required columns: {expected_columns}")
                    continue

                # Rename columns for consistency
                df = df.rename(columns={'Brand Name': 'brand', 'Product Name': 'product', 'Review': 'text'})
                df = df[['brand', 'text']] # Keep only relevant columns

                li.append(df)
            except pd.errors.EmptyDataError:
                print(f"WARNING: Skipping empty file: {filename}")
            except Exception as e:
                print(f"ERROR reading {filename}: {e}")

        if not li:
            print("ERROR: No valid data could be read from any source CSV file.")
            exit(1)

        combined_df = pd.concat(li, axis=0, ignore_index=True)
        combined_df.dropna(subset=['text'], inplace=True) # Drop rows where 'text' is missing

        # Save the combined data
        try:
            combined_df.to_csv(Config.COMBINED_DATASET_FILE, index=False)
            print(f"Combined data saved to {Config.COMBINED_DATASET_FILE}")
        except Exception as e:
             print(f"Error saving combined CSV: {e}")
             exit(1)

        # --- Dispatch Training Task ---
        task = train_model_async.delay()
        print(f"Training task initiated. Task ID: {task.id}")
    else:
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
        print("Model already trained.")