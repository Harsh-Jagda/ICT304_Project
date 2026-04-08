import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Make project root importable so config.constants can be found from any CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.constants import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL_Z,
    HOLDING_COST_RATE,
)

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
LEAD_TIME_PATH = os.path.join(DATA_DIR, "config", "lead_times.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "data", "processed", "test_sample.parquet")
OUTPUT_RECS = os.path.join(DATA_DIR, "data", "outputs", "purchase_recommendations.csv")

# Must match train.py exactly
CAT_FEATURES = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'event_name_1', 'event_type_1']


# =========================================================================
# CANONICAL ROP FORMULA — single source of truth for the entire system.
# Import this function in page_procurement.py and prepare_simulation.py.
# NEVER copy-paste the formula elsewhere.
# =========================================================================
def calculate_rop(
    pred_demand: float,
    lead_time: float,
    demand_std: float,
    z_score: float = DEFAULT_SERVICE_LEVEL_Z,
) -> tuple[float, float]:
    """
    Compute Safety Stock and Reorder Point using the industry-standard formula.

    Args:
        pred_demand:  Predicted daily demand (units/day).
        lead_time:    Days from order placement to shelf arrival.
        demand_std:   Rolling 7-day standard deviation of daily demand.
                      Use 0.0 if unavailable (new items, cold-start).
        z_score:      Service level multiplier. Default = 1.645 (95% fill rate).

    Returns:
        (safety_stock, reorder_point) as a tuple of floats.

    Formula reference:
        Safety Stock  = Z × σ × √(LT)
        Reorder Point = (Demand × LT) + Safety Stock
    """
    if demand_std is None or np.isnan(demand_std):
        demand_std = 0.0
    safety_stock = z_score * max(demand_std, 0.0) * np.sqrt(max(lead_time, 1))
    reorder_point = (pred_demand * lead_time) + safety_stock
    return safety_stock, reorder_point

def load_resources():
    print("Loading model and configuration...")
    model = joblib.load(MODEL_PATH)
    with open(LEAD_TIME_PATH, 'r') as f:
        lead_times = json.load(f)
    df = pd.read_parquet(TEST_DATA_PATH)
    return model, lead_times, df

def generate_recommendations(model, lead_time_config, df):
    print("Generating purchase recommendations...")
    
    # Identify features used during training
    features = model.feature_name()
    
    # To bypass categorical metadata mismatch, we convert categories to their underlying codes
    # LightGBM handles this if we pass them as integers.
    X = df[features].copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            # If it's already a category, use codes. If not, it's a bug in data prep.
            if hasattr(X[col], 'cat'):
                X[col] = X[col].cat.codes
            else:
                X[col] = X[col].astype('category').cat.codes
    
    # Predict demand (t+1)
    # Using .values bypasses pandas-specific validation which often fails due to metadata mismatch
    df['predicted_demand'] = model.predict(X.values)
    df['predicted_demand'] = df['predicted_demand'].clip(lower=0)
    
    # 2. Integrate Lead Times & Service Levels
    df['lead_time_days'] = df['dept_id'].apply(lambda x: lead_time_config.get(x, {}).get('lead_time_days', 2))
    df['z_score'] = df['dept_id'].apply(lambda x: lead_time_config.get(x, {}).get('service_level_z', 1.645))
    
    # 3. Calculate Safety Stock & Reorder Point
    # Using the canonical calculate_rop() — do not inline this formula elsewhere.
    rop_results = df.apply(
        lambda row: calculate_rop(
            pred_demand=row['predicted_demand'],
            lead_time=row['lead_time_days'],
            demand_std=row['rolling_std_7'],
            z_score=row['z_score'],
        ),
        axis=1,
        result_type='expand',
    )
    df['safety_stock'] = rop_results[0].clip(lower=0)
    df['reorder_point'] = rop_results[1]
    
    # 4. Mock Current Inventory
    np.random.seed(42)
    df['current_inventory'] = np.random.uniform(0, df['predicted_demand'] * 3, size=len(df)).astype(int)
    
    # 5. Recommendation Logic
    df['order_needed'] = df['current_inventory'] < df['reorder_point']
    df['recommended_order_qty'] = np.where(
        df['order_needed'],
        np.ceil(df['reorder_point'] - df['current_inventory']).astype(int),
        0
    )
    
    # 6. Financial Metrics (Mock)
    df['holding_cost_daily'] = df['current_inventory'] * HOLDING_COST_RATE  # from config.constants
    df['potential_stockout_loss'] = np.where(
        df['predicted_demand'] > df['current_inventory'],
        (df['predicted_demand'] - df['current_inventory']) * df['sell_price'].fillna(0),
        0
    )
    
    # 7. XAI: "Why this order?"
    def create_explanation(row):
        if not row['order_needed']:
            return "Inventory sufficient: Current stock covers ROP."
        return (f"Order {row['recommended_order_qty']} units: "
                f"Predicted demand is {row['predicted_demand']:.1f}/day. "
                f"With {row['lead_time_days']}d lead time and {row['safety_stock']:.1f} buffer, "
                f"we need {row['reorder_point']:.1f} total.")
    
    df['xai_reasoning'] = df.apply(create_explanation, axis=1)
    
    return df

def main():
    model, lead_times, df = load_resources()
    
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].copy()
    
    if len(latest_df) == 0:
        print("No data for latest date, using whole sample.")
        latest_df = df.copy()
        
    recs_df = generate_recommendations(model, lead_times, latest_df)
    
    recs_output = recs_df[['item_id', 'dept_id', 'current_inventory', 'reorder_point', 
                          'recommended_order_qty', 'potential_stockout_loss', 'xai_reasoning']]
    
    print(f"Saving recommendations to {OUTPUT_RECS}...")
    recs_output.to_csv(OUTPUT_RECS, index=False)
    print("Recommendation engine complete!")

if __name__ == "__main__":
    main()
