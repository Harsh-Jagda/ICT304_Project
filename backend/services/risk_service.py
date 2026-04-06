import joblib
import os
import pandas as pd
import numpy as np
import math
import json
from .data_prep_service import process_uploaded_csv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wms_lgbm_model.pkl")

# Exact feature set from the forecast service
CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'event_name_1', 'event_type_1']
NUM_FEATURES  = ['sell_price', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28']
FEATURES = CAT_FEATURES + NUM_FEATURES

def compute_risk(csv_path, category="FOODS", json_filename="lead_times.json", selected_state="ALL"):
    try:
        # 1. Resolve Config Path (Uploads priority)
        upload_path = os.path.join(os.path.dirname(__file__), "..", "uploads", json_filename)
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "model", json_filename)
        lead_path = upload_path if os.path.exists(upload_path) else fallback_path

        if not os.path.exists(lead_path):
            return None, f"Configuration file {json_filename} missing."

        with open(lead_path, "r") as f:
            lead_times = json.load(f)

        # 2. Process CSV & Get Available Regions for the Frontend
        df, err = process_uploaded_csv(csv_path)
        if err: return None, err

        available_regions = []
        if "state_id" in df.columns:
            available_regions = sorted([str(s).strip() for s in df["state_id"].unique() if pd.notna(s)])

        # 3. Filter by Region & Category
        if selected_state != "ALL" and "state_id" in df.columns:
            df = df[df["state_id"].astype(str).str.upper() == selected_state.upper()].copy()

        if "cat_id" in df.columns:
            category_data = df[df["cat_id"] == category].copy()
        else:
            category_data = df.copy()

        if category_data.empty: 
            return None, f"No data found for {category} in region {selected_state}"

        # 4. ML-Powered Demand Prediction for ROP
        model = joblib.load(MODEL_PATH)
        X = category_data.copy()
        
        for feat in FEATURES:
            if feat not in X.columns:
                X[feat] = 0 if feat in NUM_FEATURES else "Unknown"
        
        for col in CAT_FEATURES:
            X[col] = pd.Categorical(X[col]).codes

        # Use the bypass flag for the 16-feature mismatch
        preds = model.predict(X[FEATURES].values, predict_disable_shape_check=True)
        category_data["ml_forecast"] = np.clip(preds, 0, None)

        # 5. Risk Math
        avg_demand = max(float(category_data["ml_forecast"].mean()), 0.001)
        std_dev = float(category_data["rolling_std_7"].mean()) if "rolling_std_7" in category_data.columns else 0.5

        # Match category from JSON
        matched_key = next((k for k in lead_times if k in category), None)
        lead_cfg = lead_times[matched_key] if matched_key else {"lead_time_days": 3, "service_level_z": 1.645}
        
        lt = lead_cfg.get("lead_time_days", 3)
        z = lead_cfg.get("service_level_z", 1.645)

        # ROP = (Forecasted Demand * Lead Time) + Safety Stock
        safety_stock = z * std_dev * math.sqrt(lt)
        rop = (avg_demand * lt) + safety_stock
        days_of_cover = round(rop / avg_demand, 1) if avg_demand > 0 else 0

        # Recommended Action Logic
        if days_of_cover < lt:
            action = "CRITICAL: Stockout risk. Expedite regional transfer."
        elif days_of_cover > 45:
            action = "OVERSTOCK: Capital frozen. Reduce reorder quantity."
        else:
            action = "OPTIMAL: Inventory levels healthy for this region."

        return {
            "region": selected_state,
            "category": category,
            "rop": round(rop, 2),
            "safety_stock": round(safety_stock, 2),
            "risk_status": "High" if (std_dev / avg_demand) > 1.0 else "Low/Medium",
            "estimated_days_of_cover": days_of_cover,
            "recommended_action": action,
            "available_states": available_regions, # Added for the dropdown loop
            "total_items": len(category_data),
            "lead_time": lt
        }, None

    except Exception as e:
        return None, f"Risk analysis error: {str(e)}"
