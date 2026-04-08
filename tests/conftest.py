"""
conftest.py — Shared pytest fixtures for the WMS test suite.
Fixtures here are automatically available to all test files.
"""
import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

# Make root importable from any test file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def model():
    """Load the production LightGBM model once for the whole session."""
    from src.ui.data_manager import load_model
    return load_model()


@pytest.fixture(scope="session")
def sample_df():
    """
    Load a small slice of real data (500 rows) for fast tests.
    Using session scope so we only read from disk once.
    """
    path = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
                "wday", "month", "event_name_1", "event_type_1"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df.head(500).copy()


@pytest.fixture(scope="session")
def registry():
    """Load the model registry JSON (read-only)."""
    path = os.path.join(DATA_DIR, "models", "model_registry.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def alien_item_row(sample_df):
    """
    A single data row with a completely fabricated item ID.
    Used to simulate Cold Start / new product arrival.
    """
    row = sample_df.iloc[0:1].copy()
    row["item_id"] = "FOODS_X_9999"
    row["dept_id"] = "FOODS_X"
    return row
