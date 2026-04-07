import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .data_prep_service import process_uploaded_csv
from .risk_service import FEATURES, CAT_FEATURES # Import CAT_FEATURES too

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wms_lgbm_model.pkl")

def get_model_metrics(file_path, selected_state='ALL', target='sales'):
    try:
        # 1. Standardized Data Prep
        df, err = process_uploaded_csv(file_path)
        if err:
            return {"error": f"Data Prep Error: {err}"}

        # 2. Regional Filtering
        if selected_state != 'ALL' and 'state_id' in df.columns:
            df = df[df['state_id'].astype(str).str.upper() == selected_state.upper()].copy()

        if df.empty:
            return {"error": f"No data available for region: {selected_state}"}

        # 3. Chronological Split (Last 15%)
        test_size = int(len(df) * 0.15)
        if test_size < 1: test_size = len(df)
        test_df = df.tail(test_size).copy()

        # 4. CRITICAL FIX: Categorical Alignment
        X_test = test_df[FEATURES].copy()
        
        # Explicitly tell pandas/lightgbm which columns are categories
        for col in CAT_FEATURES:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('category')

        # 5. Run Inference
        model = joblib.load(MODEL_PATH)
        
        # We pass the DataFrame directly (not .values) so column names/types are preserved
        y_pred = model.predict(X_test) 
        y_true = test_df[target]

        # 6. Calculate Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 7. Baseline Comparison
        if 'lag_7' in test_df.columns:
            naive_mae = mean_absolute_error(y_true, test_df['lag_7'].fillna(0))
            improvement = ((naive_mae - mae) / naive_mae) * 100 if naive_mae != 0 else 0
        else:
            naive_mae = 0
            improvement = 0

        return {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "baseline_mae": round(float(naive_mae), 3),
            "improvement_pct": round(float(improvement), 1),
            "sample_count": len(test_df),
            "region": selected_state
        }
    except Exception as e:
        return {"error": f"Evaluation Service Error: {str(e)}"}
