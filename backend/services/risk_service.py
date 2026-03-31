import pandas as pd
import numpy as np
import math
import json
import os

def compute_risk(csv_path, category="FOODS", json_filename="lead_times.json"):
    try:
        print(f"🔍 DEBUG RISK: Processing {csv_path} for category {category} using config {json_filename}")

        # ---------------- LOAD DYNAMIC LEAD TIMES ----------------
        # Go up one level from 'services' to 'backend', then into 'model'
        model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
        lead_path = os.path.join(model_dir, json_filename)

        if os.path.exists(lead_path):
            with open(lead_path, "r") as f:
                lead_times = json.load(f)
        else:
            print(f"⚠️ DEBUG RISK: {json_filename} not found at {lead_path}. Using empty dict.")
            lead_times = {}

        # Match category properly (handles PHARMA, SURGICAL, etc.)
        matched_key = None
        for key in lead_times:
            if key in category:
                matched_key = key
                break

        if matched_key:
            lead = lead_times[matched_key]
        else:
            # Global fallback if category isn't in the JSON
            lead = {"lead_time_days": 3, "service_level_z": 1.645}

        # ---------------- LOAD DATA ----------------
        try:
            if csv_path.endswith(".parquet"):
                df = pd.read_parquet(csv_path, columns=["sales", "cat_id", "rolling_std_7"])
            else:
                df = pd.read_csv(csv_path)
        except Exception as e:
            return None, f"Read Error: {str(e)}"

        # ---------------- PREP & FILTER ----------------
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)

        # Ensure numeric types
        for col in ["sales", "rolling_std_7"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype("float32")

        # Filter to the specific category
        if "cat_id" in df.columns:
            category_data = df[df["cat_id"] == category]
        else:
            category_data = df

        # ---------------- CALCULATIONS ----------------
        avg = float(category_data["sales"].mean()) if len(category_data) > 0 else float(df["sales"].mean())
        
        # Use category-specific volatility if available
        if len(category_data) > 0 and "rolling_std_7" in category_data.columns:
            std = float(category_data["rolling_std_7"].mean())
        elif "rolling_std_7" in df.columns:
            std = float(df["rolling_std_7"].mean())
        else:
            std = 0.5

        # Safety fallback for std
        std = 0.5 if (np.isnan(std) or std <= 0) else std

        lead_time = lead.get("lead_time_days", 3)
        z = lead.get("service_level_z", 1.645)

        # ROP Formula: (Avg Demand * Lead Time) + (Z * Std * sqrt(Lead Time))
        safety = z * std * math.sqrt(lead_time)
        rop = (avg * lead_time) + safety

        if math.isnan(rop) or math.isnan(safety):
            return None, "Calculation error: NaN encountered"

        return {
            "category": category,
            "rop": round(rop, 2),
            "safety": round(safety, 2),
            "lead_time": lead_time,
            "service_level_z": z,
            "avg_daily_demand": round(avg, 2),
            "explanation": f"Source: {json_filename} | LT={lead_time}d"
        }, None

    except Exception as e:
        return None, f"Risk analysis error: {str(e)}"
