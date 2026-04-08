import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MEMORY FIX: The dashboard only ever shows store CA_1, so we load the
# CA-only parquet (~142 MB, ~1.2 GB in RAM) instead of all-states
# (~347 MB, ~4 GB in RAM). The all-states file is reserved for retraining.
# This prevents ArrowMemoryError on machines with <= 8 GB RAM.
_CA_DATA   = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
_ALL_DATA  = os.path.join(DATA_DIR, "data", "processed", "processed_data_all.parquet")
DATA_PATH  = _CA_DATA if os.path.exists(_CA_DATA) else _ALL_DATA

MODEL_PATH = os.path.join(DATA_DIR, "models", "wms_lgbm_model.pkl")
LT_PATH    = os.path.join(DATA_DIR, "config", "lead_times.json")
SIM_PATH   = os.path.join(DATA_DIR, "data", "outputs", "sim_results.parquet")

CAT_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
            'wday', 'month', 'event_name_1', 'event_type_1']

@st.cache_resource(show_spinner=False)
def load_model():
    # FIX #7: show a clear actionable error instead of a raw FileNotFoundError
    if not os.path.exists(MODEL_PATH):
        st.error(
            "🔴 **Model file not found.**\n\n"
            f"Expected: `{MODEL_PATH}`\n\n"
            "**To fix this:** run the training pipeline first:\n"
            "```\npython src/mlops/train.py\n```"
        )
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading Data into Memory...")
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            "🔴 **Processed data file not found.**\n\n"
            f"Expected: `{DATA_PATH}`\n\n"
            "**To fix this:** run the data pipeline first:\n"
            "```\npython src/data_pipeline/data_prep.py\n```"
        )
        st.stop()

    try:
        df = pd.read_parquet(DATA_PATH)
    except MemoryError:
        st.error(
            "🔴 **Not enough RAM to load the dataset.**\n\n"
            "Your system ran out of memory loading the parquet file.\n\n"
            "**Solutions:**\n"
            "1. Close other applications to free RAM\n"
            "2. Ensure at least **4 GB free RAM** is available\n"
            "3. If on a low-memory machine, the CA-only file (~142 MB) should be present at:\n"
            f"   `{_CA_DATA}`"
        )
        st.stop()
    except Exception as e:
        st.error(f"🔴 **Failed to load data:** {e}")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    sim_df = None
    if os.path.exists(SIM_PATH):
        sim_df = pd.read_parquet(SIM_PATH)

    with open(LT_PATH) as f:
        lt = json.load(f)
    return df, sim_df, lt

def predict_batch(model, rows: pd.DataFrame) -> np.ndarray:
    features = model.feature_name()
    X = rows[features].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes
    return np.clip(model.predict(X.values), 0, None)

@st.cache_resource(show_spinner="Building Historical Matrix...")
def build_sales_matrix(df: pd.DataFrame, store_id: str):
    store_df = df[df['store_id'] == store_id]
    sales_dict = {}
    for d, grp in store_df.groupby('date', observed=True):
        sales_dict[d.strftime('%Y-%m-%d')] = grp.set_index('item_id')['sales'].to_dict()
    return store_df, sales_dict
