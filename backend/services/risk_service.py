import pandas as pd
import numpy as np
import math
import json
import os


def compute_risk(csv_path, category="FOODS"):
    try:
        print(f"🔍 DEBUG RISK: Processing {csv_path} for category {category}")

        # ---------------- LOAD LEAD TIMES ----------------
        lead_path = os.path.join(os.path.dirname(__file__), "lead_times.json")

        if os.path.exists(lead_path):
            with open(lead_path, "r") as f:
                lead_times = json.load(f)
        else:
            lead_times = {}

        # Match category properly (handles FOODS_1 etc.)
        matched_key = None
        for key in lead_times:
            if key in category:
                matched_key = key
                break

        if matched_key:
            lead = lead_times[matched_key]
        else:
            print("⚠️ DEBUG RISK: No match found, using defaults")
            lead = {"lead_time_days": 3, "service_level_z": 1.645}

        # ---------------- LOAD DATA (FAST) ----------------
        try:
            df = pd.read_parquet(csv_path, columns=["sales", "cat_id", "rolling_std_7"])
        except Exception:
            df = pd.read_csv(csv_path)

        print(f"✅ DEBUG: Loaded {len(df)} rows")

        # ---------------- SAMPLE EARLY ----------------
        if len(df) > 10000:
            print("⚠️ DEBUG: Sampling 10,000 rows")
            df = df.sample(10000, random_state=42)

        # ---------------- FIX DATA TYPES ----------------
        if "rolling_std_7" in df.columns:
            df["rolling_std_7"] = df["rolling_std_7"].astype("float32")

        if "sales" in df.columns:
            df["sales"] = df["sales"].astype("float32")

        # ---------------- CLEAN ----------------
        if "rolling_std_7" in df.columns and "sales" in df.columns:
            df = df.dropna(subset=["rolling_std_7", "sales"])

        # ---------------- FILTER CATEGORY ----------------
        if "cat_id" in df.columns:
            category_data = df[df["cat_id"] == category]
        else:
            category_data = df

        # ---------------- AVG DEMAND ----------------
        if len(category_data) > 0:
            avg = float(category_data["sales"].mean())
        else:
            avg = float(df["sales"].mean())

        # ---------------- STD DEV ----------------
        if len(category_data) > 0 and "rolling_std_7" in category_data.columns:
            std = float(category_data["rolling_std_7"].mean())
        elif "rolling_std_7" in df.columns:
            std = float(df["rolling_std_7"].mean())
        else:
            std = 0.5

        # Safety fallback
        if np.isnan(std) or std == 0:
            std = 0.5

        # ---------------- CALCULATIONS ----------------
        lead_time = lead["lead_time_days"]
        z = lead["service_level_z"]

        safety = z * std * math.sqrt(lead_time)
        rop = (avg * lead_time) + safety

        # Final NaN protection
        if math.isnan(rop) or math.isnan(safety):
            return None, "Calculation error: NaN encountered"

        result = {
            "category": category,
            "rop": round(rop, 2),
            "safety": round(safety, 2),
            "lead_time": lead_time,
            "service_level_z": z,
            "avg_daily_demand": round(avg, 2),
            "explanation": f"{category}: Lead={lead_time} days, Z={z}"
        }

        print(f"✅ DEBUG RISK: Result: {result}")
        return result, None

    except Exception as e:
        import traceback
        print("💥 DEBUG RISK: Exception:")
        print(traceback.format_exc())
        return None, f"Risk analysis error: {str(e)}"
