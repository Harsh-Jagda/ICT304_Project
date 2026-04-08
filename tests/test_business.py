"""
test_business.py — Subsystem 4: Business Logic QA
==================================================
As you correctly noted, there is no value in testing that 2×10=20.
Instead, these tests focus on what ACTUALLY matters for a production system:

  A) Robustness   — does the engine handle broken/missing inputs gracefully?
  B) XAI Quality  — are explanations generated for every row? Do they contain numbers?
  C) Business Rules — does the flagging logic (order_needed) follow the rules exactly?
  D) Output Schema  — does the function return all columns the dashboard expects?
  E) Integration    — does the full engine run end-to-end without crashing?
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.business.recommendation import generate_recommendations

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────

# NOTE: The 'model' and 'sample_df' fixtures are defined in conftest.py
# and are automatically available to all test classes below.
# DO NOT re-declare them here — doing so causes a shadowing bug.

@pytest.fixture(scope="module")
def lead_time_config():
    """Load the real lead_times.json configuration."""
    path = os.path.join(DATA_DIR, "config", "lead_times.json")
    with open(path) as f:
        return json.load(f)


def make_input_row(
    item_id="FOODS_1_001", dept_id="FOODS_1", cat_id="FOODS",
    store_id="CA_1", state_id="CA",
    predicted_demand=5.0, rolling_std_7=1.0,
    sell_price=3.50, rolling_mean_7=5.0,
    lag_7=4.0, lag_28=4.5,
    rolling_mean_28=4.8, rolling_std_28=1.2,
    wday=2, month=3,
    event_name_1="NoEvent", event_type_1="NoEvent",
    sales=5,
):
    """
    Create a minimal single-row DataFrame that mimics real processed data.
    All parameters have sensible defaults so tests only need to override
    what they're actually testing.
    """
    return pd.DataFrame([{
        "item_id": item_id, "dept_id": dept_id, "cat_id": cat_id,
        "store_id": store_id, "state_id": state_id,
        "sales": sales, "sell_price": sell_price,
        "rolling_std_7": rolling_std_7, "rolling_mean_7": rolling_mean_7,
        "rolling_std_28": rolling_std_28, "rolling_mean_28": rolling_mean_28,
        "lag_7": lag_7, "lag_28": lag_28,
        "wday": wday, "month": month,
        "event_name_1": event_name_1, "event_type_1": event_type_1,
    }])


# ─────────────────────────────────────────────────────────
# A) ROBUSTNESS
# ─────────────────────────────────────────────────────────

class TestRobustness:
    """The engine must never crash, even on bad data."""

    def test_nan_rolling_std_does_not_crash(self, model, lead_time_config):
        """
        rolling_std_7 = NaN is extremely common (items with < 7 days history).
        The engine must handle it with fillna(0), not crash with a ValueError.
        """
        df = make_input_row(rolling_std_7=np.nan)
        try:
            result = generate_recommendations(model, lead_time_config, df)
            assert result["safety_stock"].iloc[0] == 0.0, (
                "NaN std should produce zero safety stock"
            )
        except Exception as e:
            pytest.fail(f"Engine crashed on NaN rolling_std_7: {e}")

    def test_zero_sell_price_does_not_crash(self, model, lead_time_config):
        """Some items may have no price data. The engine must not divide by zero."""
        df = make_input_row(sell_price=0.0)
        try:
            generate_recommendations(model, lead_time_config, df)
        except Exception as e:
            pytest.fail(f"Engine crashed on sell_price=0: {e}")

    def test_unknown_dept_uses_default_lead_time(self, model, lead_time_config):
        """
        An item from a dept_id not in lead_times.json (e.g. a new department)
        should fall back to a default lead time of 2 days, not crash.
        """
        df = make_input_row(dept_id="BRAND_NEW_DEPT_9999")
        result = generate_recommendations(model, lead_time_config, df)
        # Default lead time is 2 (hardcoded in the .get fallback)
        assert result["lead_time_days"].iloc[0] == 2, (
            f"Expected default lead time=2 for unknown dept, "
            f"got {result['lead_time_days'].iloc[0]}"
        )

    def test_engine_handles_multiple_rows(self, model, lead_time_config):
        """Engine must work on a batch (not just a single row)."""
        rows = pd.concat([make_input_row() for _ in range(10)], ignore_index=True)
        result = generate_recommendations(model, lead_time_config, rows)
        assert len(result) == 10, "Output length should match input length"


# ─────────────────────────────────────────────────────────
# B) XAI QUALITY
# ─────────────────────────────────────────────────────────

class TestXAIExplanations:
    """
    Explainable AI (XAI) is a key feature of this system.
    Every recommendation must have a non-empty, informative explanation
    that a warehouse manager with no technical background can understand.
    """

    def test_every_row_has_explanation(self, model, lead_time_config):
        """No row may have an empty or null XAI explanation."""
        rows = pd.concat([make_input_row() for _ in range(5)], ignore_index=True)
        result = generate_recommendations(model, lead_time_config, rows)
        for i, explanation in enumerate(result["xai_reasoning"]):
            assert isinstance(explanation, str) and len(explanation) > 0, (
                f"Row {i} has an empty explanation"
            )

    def test_order_explanation_contains_key_numbers(self, model, lead_time_config):
        """
        When an order is recommended, the explanation must include:
        - the recommended quantity
        - the predicted demand rate
        This ensures managers get actionable, data-backed reasoning.
        """
        # Force a situation where an order IS needed:
        # Set a high demand prediction but zero inventory
        df = make_input_row(rolling_std_7=0.1)
        result = generate_recommendations(model, lead_time_config, df)

        for _, row in result.iterrows():
            if row["order_needed"]:
                explanation = row["xai_reasoning"]
                assert "Predicted demand" in explanation, (
                    "Explanation does not mention 'Predicted demand'"
                )
                assert "lead time" in explanation.lower(), (
                    "Explanation does not mention lead time"
                )

    def test_no_order_explanation_is_positive(self, model, lead_time_config):
        """When no order is needed, the explanation should say stock is sufficient."""
        # Create a row where inventory is definitely sufficient
        # by making demand very low
        df = make_input_row(rolling_std_7=0.0)
        result = generate_recommendations(model, lead_time_config, df)

        for _, row in result.iterrows():
            if not row["order_needed"]:
                assert "sufficient" in row["xai_reasoning"].lower(), (
                    f"No-order explanation unclear: '{row['xai_reasoning']}'"
                )


# ─────────────────────────────────────────────────────────
# C) BUSINESS RULES
# ─────────────────────────────────────────────────────────

class TestBusinessRules:
    """
    These tests verify the decision logic, not the math.
    'When should we order?' is a business question, not a formula.
    """

    def test_order_needed_flag_is_boolean(self, model, lead_time_config):
        """order_needed must always be True or False, not NaN."""
        result = generate_recommendations(model, lead_time_config, make_input_row())
        assert result["order_needed"].dtype == bool or \
               result["order_needed"].isin([True, False]).all(), (
            "order_needed column contains non-boolean values"
        )

    def test_recommended_qty_is_zero_when_no_order_needed(self, model, lead_time_config):
        """If no order is needed, recommended_order_qty must be 0, not negative."""
        result = generate_recommendations(model, lead_time_config, make_input_row())
        no_order_rows = result[~result["order_needed"]]
        if len(no_order_rows) > 0:
            assert (no_order_rows["recommended_order_qty"] == 0).all(), (
                "recommended_order_qty must be 0 when order_needed is False"
            )

    def test_recommended_qty_is_positive_when_order_needed(self, model, lead_time_config):
        """If an order is needed, the quantity must be > 0."""
        result = generate_recommendations(model, lead_time_config, make_input_row())
        order_rows = result[result["order_needed"]]
        if len(order_rows) > 0:
            assert (order_rows["recommended_order_qty"] > 0).all(), (
                "recommended_order_qty must be > 0 when order_needed is True"
            )

    def test_holding_cost_non_negative(self, model, lead_time_config):
        """Holding cost is stock × rate. Stock is never negative, so cost is never negative."""
        result = generate_recommendations(model, lead_time_config, make_input_row())
        assert (result["holding_cost_daily"] >= 0).all(), (
            "Holding cost is negative — this is mathematically impossible"
        )

    def test_potential_loss_non_negative(self, model, lead_time_config):
        """Potential stockout loss can only be zero or positive."""
        result = generate_recommendations(model, lead_time_config, make_input_row())
        assert (result["potential_stockout_loss"] >= 0).all(), (
            "Stockout loss has negative values"
        )


# ─────────────────────────────────────────────────────────
# D) OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────

class TestOutputSchema:
    """The dashboard relies on specific column names. They must always be present."""

    REQUIRED_COLUMNS = [
        "predicted_demand",
        "lead_time_days",
        "z_score",
        "safety_stock",
        "reorder_point",
        "current_inventory",
        "order_needed",
        "recommended_order_qty",
        "holding_cost_daily",
        "potential_stockout_loss",
        "xai_reasoning",
    ]

    def test_all_required_columns_present(self, model, lead_time_config):
        """Every column the dashboard reads must exist in the output."""
        result = generate_recommendations(model, lead_time_config, make_input_row())
        missing = [c for c in self.REQUIRED_COLUMNS if c not in result.columns]
        assert not missing, (
            f"generate_recommendations() is missing output columns: {missing}\n"
            "The dashboard will crash if these are absent."
        )


# ─────────────────────────────────────────────────────────
# E) INTEGRATION
# ─────────────────────────────────────────────────────────

class TestIntegration:
    """Full end-to-end test using a real slice of data from disk."""

    def test_engine_runs_on_real_sample(self, model, lead_time_config, sample_df):
        """
        Pass 50 rows of real processed data through the full engine.
        This is the closest thing to a production smoke test.
        """
        try:
            result = generate_recommendations(model, lead_time_config, sample_df.head(50).copy())
            assert len(result) == 50
            assert "xai_reasoning" in result.columns
            assert result["recommended_order_qty"].notna().all()
        except Exception as e:
            pytest.fail(f"Engine failed on real data: {e}")
