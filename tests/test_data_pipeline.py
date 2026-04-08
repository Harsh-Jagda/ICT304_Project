"""
test_data_pipeline.py — Subsystem 2: Data Pipeline QA
======================================================
Tests every transformation stage of data_prep.py using small
synthetic DataFrames — no giant CSVs required, runs in seconds.

  A) Memory Reduction   — does downcasting actually shrink the data?
  B) Data Melting       — wide-to-long reshape is correct?
  C) Data Cleaning      — pre-launch zeros removed? outliers capped?
  D) Feature Engineering — lags and rolling stats computed correctly?
  E) Output Integrity   — the final processed parquet has no surprises
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.data_prep import (
    reduce_mem_usage,
    melt_sales,
    clean_data,
    create_features,
)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────
# Shared synthetic fixtures
# ─────────────────────────────────────────────────────────

def make_wide_sales(n_items=3, n_days=10):
    """Create a minimal wide-format sales DataFrame (like the raw Kaggle CSV)."""
    rows = []
    for i in range(n_items):
        row = {
            "id":       f"FOODS_1_{i:03d}_CA_1_evaluation",
            "item_id":  f"FOODS_1_{i:03d}",
            "dept_id":  "FOODS_1",
            "cat_id":   "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        }
        for d in range(1, n_days + 1):
            row[f"d_{d}"] = int(np.random.randint(0, 5))
        rows.append(row)
    return pd.DataFrame(rows)


def make_long_sales(n_items=3, n_days=30):
    """
    Create a minimal long-format (already melted) DataFrame with a date column.
    This is the format after melt_sales() + merge_calendar_and_prices().
    """
    base_date = date(2013, 1, 29)
    rows = []
    for i in range(n_items):
        item_id = f"FOODS_1_{i:03d}"
        for d in range(n_days):
            rows.append({
                "id":         f"{item_id}_CA_1_evaluation",
                "item_id":    item_id,
                "dept_id":    "FOODS_1",
                "cat_id":     "FOODS",
                "store_id":   "CA_1",
                "state_id":   "CA",
                "date":       pd.Timestamp(base_date + timedelta(days=d)),
                "sales":      int(np.random.randint(0, 6)),
                "sell_price": float(np.random.uniform(1.0, 5.0)),
                "wday":       ((d % 7) + 1),
                "month":      (base_date + timedelta(days=d)).month,
                "event_name_1": "NoEvent",
                "event_type_1": "NoEvent",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# A) MEMORY REDUCTION
# ─────────────────────────────────────────────────────────

class TestMemoryReduction:
    """reduce_mem_usage() should shrink DataFrame memory without losing values."""

    def test_memory_actually_decreases(self):
        """After downcasting, total memory usage must be strictly smaller."""
        df = pd.DataFrame({
            "a": np.array([1, 2, 3], dtype=np.int64),
            "b": np.array([1.1, 2.2, 3.3], dtype=np.float64),
        })
        mem_before = df.memory_usage(deep=True).sum()
        df_reduced = reduce_mem_usage(df.copy(), verbose=False)
        mem_after  = df_reduced.memory_usage(deep=True).sum()

        assert mem_after < mem_before, (
            f"Memory did not decrease: before={mem_before}B, after={mem_after}B"
        )

    def test_values_unchanged_after_reduction(self):
        """Downcasting must not alter any actual values — only the storage type."""
        df = pd.DataFrame({
            "sales": np.array([0, 1, 10, 100], dtype=np.int64),
            "price": np.array([1.5, 2.0, 3.75, 0.99], dtype=np.float64),
        })
        df_reduced = reduce_mem_usage(df.copy(), verbose=False)

        pd.testing.assert_series_equal(
            df["sales"].astype(float),
            df_reduced["sales"].astype(float),
            check_names=False,
        )

    def test_small_ints_downcast_to_int8(self):
        """Integers in range [-128, 127] should become int8 — the smallest type."""
        df = pd.DataFrame({"x": np.array([0, 5, 100], dtype=np.int64)})
        df_reduced = reduce_mem_usage(df.copy(), verbose=False)
        assert df_reduced["x"].dtype == np.int8, (
            f"Expected int8, got {df_reduced['x'].dtype}"
        )


# ─────────────────────────────────────────────────────────
# B) DATA MELTING
# ─────────────────────────────────────────────────────────

class TestDataMelting:
    """melt_sales() must reshape wide → long correctly."""

    def test_output_row_count(self):
        """Wide table with N items × D day columns = N×D long rows."""
        n_items, n_days = 3, 10
        wide = make_wide_sales(n_items=n_items, n_days=n_days)
        long = melt_sales(wide)
        assert len(long) == n_items * n_days, (
            f"Expected {n_items * n_days} rows, got {len(long)}"
        )

    def test_id_columns_preserved(self):
        """All identifier columns must survive the melt."""
        wide = make_wide_sales()
        long = melt_sales(wide)
        for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            assert col in long.columns, f"Column '{col}' missing after melt"

    def test_sales_column_exists(self):
        """The melted DataFrame must have a 'sales' column."""
        wide = make_wide_sales()
        long = melt_sales(wide)
        assert "sales" in long.columns

    def test_no_day_columns_remain(self):
        """After melting, no 'd_N' columns should survive."""
        wide = make_wide_sales()
        long = melt_sales(wide)
        leftover_day_cols = [c for c in long.columns if c.startswith("d_")]
        assert len(leftover_day_cols) == 0, (
            f"Day columns still present after melt: {leftover_day_cols}"
        )


# ─────────────────────────────────────────────────────────
# C) DATA CLEANING
# ─────────────────────────────────────────────────────────

class TestDataCleaning:
    """clean_data() validates business rules: remove pre-launch zeros, cap outliers."""

    def test_pre_launch_zeros_removed(self):
        """
        Items that have never sold anything yet should be removed.
        Once an item makes its first sale, all subsequent rows (even zeros) are kept.
        """
        # Item A: always zero → should be removed entirely
        # Item B: has a sale on day 5 → rows from day 5 onwards should stay
        n_days = 10
        base = date(2013, 1, 1)
        rows = []
        for d in range(n_days):
            rows.append({
                "id": "ITEM_A", "item_id": "ITEM_A", "cat_id": "FOODS",
                "date": pd.Timestamp(base + timedelta(days=d)),
                "sales": 0, "sell_price": 1.0,
            })
            sale_val = 5 if d >= 4 else 0   # ITEM_B launches on day 5
            rows.append({
                "id": "ITEM_B", "item_id": "ITEM_B", "cat_id": "FOODS",
                "date": pd.Timestamp(base + timedelta(days=d)),
                "sales": sale_val, "sell_price": 1.0,
            })
        df = pd.DataFrame(rows)
        cleaned = clean_data(df)

        assert "ITEM_A" not in cleaned["id"].values, (
            "ITEM_A (never sold anything) should have been removed"
        )
        assert "ITEM_B" in cleaned["id"].values, (
            "ITEM_B (has sales) should remain"
        )

    def test_outliers_are_capped(self):
        """
        Extreme sales outliers (> 2× the 99th percentile) must be capped.
        This prevents the model from learning from data entry errors.
        """
        n_days = 200
        base = date(2013, 1, 1)
        rows = []
        for d in range(n_days):
            rows.append({
                "id": "ITM_1", "item_id": "ITM_1", "cat_id": "FOODS",
                "date": pd.Timestamp(base + timedelta(days=d)),
                "sales": 2,   # normal everyday sales
                "sell_price": 1.0,
            })
        # Inject one massive outlier sale
        rows[-1]["sales"] = 99999
        df = pd.DataFrame(rows)
        cleaned = clean_data(df)

        assert cleaned["sales"].max() < 99999, (
            "Outlier value 99999 was not capped by clean_data()"
        )

    def test_no_nan_sell_price_after_cleaning(self):
        """Price imputation (ffill/bfill) must leave no NaN sell_price values."""
        n_days = 10
        base = date(2013, 1, 1)
        rows = [
            {"id": "ITM_X", "item_id": "ITM_X", "cat_id": "FOODS",
             "date": pd.Timestamp(base + timedelta(days=d)),
             "sales": 1, "sell_price": (1.5 if d > 3 else np.nan)}
            for d in range(n_days)
        ]
        df = pd.DataFrame(rows)
        cleaned = clean_data(df)
        nan_prices = cleaned["sell_price"].isna().sum()
        assert nan_prices == 0, (
            f"clean_data() left {nan_prices} NaN sell_price rows after imputation"
        )


# ─────────────────────────────────────────────────────────
# D) FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """
    create_features() generates lag and rolling-window columns.
    These tests verify the mathematical correctness of each feature.
    """

    @pytest.fixture
    def feature_df(self):
        """A deterministic 30-day single-item DataFrame for exact math tests."""
        base = date(2013, 1, 1)
        sales_series = list(range(1, 31))   # 1, 2, 3, ... 30 — perfectly predictable
        rows = [{
            "id":         "DETERMINISTIC_ITEM",
            "item_id":    "DETERMINISTIC_ITEM",
            "cat_id":     "FOODS",
            "date":       pd.Timestamp(base + timedelta(days=d)),
            "sales":      sales_series[d],
            "sell_price": 1.0,
        } for d in range(30)]
        return pd.DataFrame(rows)

    def test_lag7_column_exists(self, feature_df):
        """create_features() must produce a 'lag_7' column."""
        result = create_features(feature_df)
        assert "lag_7" in result.columns

    def test_lag28_column_exists(self, feature_df):
        """create_features() must produce a 'lag_28' column."""
        result = create_features(feature_df)
        assert "lag_28" in result.columns

    def test_rolling_mean7_column_exists(self, feature_df):
        """create_features() must produce a 'rolling_mean_7' column."""
        result = create_features(feature_df)
        assert "rolling_mean_7" in result.columns

    def test_lag7_mathematical_correctness(self, feature_df):
        """
        For our deterministic item (sales = 1,2,3,...30),
        lag_7 on row 9 (sales=10) should be sales from 7 days ago = 3.
        """
        result = create_features(feature_df.copy()).reset_index(drop=True)
        # Row at index 9 has sales=10; its lag_7 should be row at index 2 → sales=3
        lag_at_9 = result.loc[9, "lag_7"]
        assert lag_at_9 == 3.0, (
            f"lag_7 at index 9 should be 3.0 (sales from 7 days back), got {lag_at_9}"
        )

    def test_rolling_mean7_nonzero_after_warmup(self, feature_df):
        """
        rolling_mean_7 requires at least 7 days of history.
        After the warmup period, values must not be NaN.
        """
        result = create_features(feature_df.copy())
        # Rows after index 14 (day 15) definitely have 7 days of lag history
        late_rows = result.iloc[15:]
        nan_count = late_rows["rolling_mean_7"].isna().sum()
        assert nan_count == 0, (
            f"rolling_mean_7 has {nan_count} NaN values after warmup period"
        )

    def test_no_feature_introduces_future_leak(self, feature_df):
        """
        All lag and rolling features use shift(1) — they look at PAST data only.
        If there is data leakage (using today's sales to predict today's sales),
        the model would achieve impossibly perfect accuracy in production.
        """
        result = create_features(feature_df.copy()).reset_index(drop=True)
        # lag_7 on the FIRST row (no history) must be NaN, not the actual sales value
        first_lag = result.loc[0, "lag_7"]
        assert pd.isna(first_lag), (
            f"lag_7 on the first row should be NaN (no past data), got {first_lag}. "
            "This may indicate data leakage!"
        )


# ─────────────────────────────────────────────────────────
# E) OUTPUT INTEGRITY
# ─────────────────────────────────────────────────────────

class TestOutputIntegrity:
    """
    Quick sanity checks on the actual processed_data_ca.parquet on disk.
    These tests verify that the last pipeline run produced a valid output.
    """

    @pytest.fixture(scope="class")
    def processed_df(self):
        path = os.path.join(DATA_DIR, "data", "processed", "processed_data_ca.parquet")
        if not os.path.exists(path):
            pytest.skip("processed_data_ca.parquet not found — run data_prep.py first")
        return pd.read_parquet(path)

    def test_all_required_columns_present(self, processed_df):
        """The output parquet must contain all columns needed for model training."""
        required = [
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "date", "sales", "sell_price", "wday", "month",
            "lag_7", "lag_28", "rolling_mean_7", "rolling_std_7",
            "rolling_mean_28", "rolling_std_28",
        ]
        missing = [c for c in required if c not in processed_df.columns]
        assert len(missing) == 0, f"Missing columns in processed output: {missing}"

    def test_sales_column_is_non_negative(self, processed_df):
        """Sales can be zero but never negative."""
        neg = (processed_df["sales"] < 0).sum()
        assert neg == 0, f"Found {neg} rows with negative sales in processed data"

    def test_date_column_is_datetime(self, processed_df):
        """The date column must be parsed as datetime, not a string."""
        assert pd.api.types.is_datetime64_any_dtype(processed_df["date"]), (
            f"'date' column is {processed_df['date'].dtype}, expected datetime64"
        )

    def test_no_all_nan_feature_columns(self, processed_df):
        """
        Feature columns (lags, rolling stats) will have some NaN at the start
        of each item's history (warmup period). But they must NOT be entirely NaN.
        """
        feature_cols = ["lag_7", "lag_28", "rolling_mean_7", "rolling_std_7"]
        for col in feature_cols:
            pct_nan = processed_df[col].isna().mean()
            assert pct_nan < 0.5, (
                f"Column '{col}' is {pct_nan:.0%} NaN — feature engineering may have failed"
            )
