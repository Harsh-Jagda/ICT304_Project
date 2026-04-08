import os
import pandas as pd
import joblib
import json

# Конфигурация путей (должна совпадать с твоими файлами)
DATA_DIR = r"c:\Users\INRUI\OneDrive\Desktop\m5-forecasting-accuracy"
REQUIRED_FILES = [
    "data_prep.py",
    "train.py",
    "recommendation.py",
    "config", "lead_times.json",
    "models", "wms_lgbm_model.pkl",
    "data", "processed", "processed_data_ca.parquet",
    "data", "processed", "test_sample.parquet"
]

def check_files():
    print("--- Stage 1: File Integrity Check ---")
    missing = []
    for f in REQUIRED_FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            print(f"[OK] File found: {f}")
        else:
            print(f"[ERROR] File NOT found: {f}")
            missing.append(f)
    return missing

def check_data_sample():
    print("\n--- Stage 2: Data Check (test_sample.parquet) ---")
    path = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")
    try:
        df = pd.read_parquet(path)
        print(f"[OK] Data loaded. Rows: {len(df)}, Columns: {list(df.columns[:5])}...")
        
        # Check for critical columns for recommendations
        required_cols = ['id', 'item_id', 'dept_id', 'sales', 'rolling_std_7']
        for col in required_cols:
            if col in df.columns:
                print(f"  - Column {col}: OK")
            else:
                print(f"  - [WARNING] Column {col} is missing!")
    except Exception as e:
        print(f"[ERROR] Failed to read data: {e}")

def check_model():
    print("\n--- Stage 3: Model Check (wms_lgbm_model.pkl) ---")
    path = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
    try:
        model = joblib.load(path)
        print(f"[OK] Model loaded. Type: {type(model)}")
        
        # Check features the model was trained on
        features = model.feature_name()
        print(f"  - Features in model: {len(features)}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

if __name__ == "__main__":
    missing_files = check_files()
    if not missing_files:
        check_data_sample()
        check_model()
    else:
        print(f"\n[CRITICAL] Missing files: {missing_files}")

if __name__ == "__main__":
    missing_files = check_files()
    if not missing_files:
        check_data_sample()
        check_model()
    else:
        print(f"\n[CRITICAL] Нужно восстановить файлы: {missing_files}")
