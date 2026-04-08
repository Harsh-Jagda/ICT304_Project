"""
test_mlops.py — Subsystem 1: MLOps Quality Assurance
=====================================================
Tests every layer of the ML lifecycle:
  A) Prediction Sanity      — is the model output trustworthy?
  B) Model Accuracy         — does the model meet our error thresholds?
  C) Cold Start Resilience  — what happens with a brand-new unknown product?
  D) Retrain Trigger Logic  — do the 4 trigger checks fire correctly?
  E) Model Registry         — does versioning & promotion logic work?
"""
import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.ui.data_manager import predict_batch
import src.mlops.retrain_manager as rm
from src.mlops.model_registry import (
    get_production_model_entry,
    get_best_model_entry,
    get_baseline_mae,
)


# ─────────────────────────────────────────────────────────
# A) PREDICTION SANITY
# ─────────────────────────────────────────────────────────

class TestPredictionSanity:
    """Basic sanity checks: does the model produce valid numbers?"""

    def test_output_length_matches_input(self, model, sample_df):
        """The number of predictions must equal the number of input rows."""
        preds = predict_batch(model, sample_df)
        assert len(preds) == len(sample_df), (
            f"Expected {len(sample_df)} predictions, got {len(preds)}"
        )

    def test_no_nan_in_predictions(self, model, sample_df):
        """No prediction should be NaN — NaN would crash the dashboard."""
        preds = predict_batch(model, sample_df)
        nan_count = int(np.isnan(preds).sum())
        assert nan_count == 0, f"Found {nan_count} NaN predictions"

    def test_no_negative_predictions(self, model, sample_df):
        """Sales cannot be negative. The model clips values at 0."""
        preds = predict_batch(model, sample_df)
        neg_count = int((preds < 0).sum())
        assert neg_count == 0, f"Found {neg_count} negative predictions"

    def test_predictions_are_finite(self, model, sample_df):
        """No prediction should be +/-inf."""
        preds = predict_batch(model, sample_df)
        assert np.all(np.isfinite(preds)), "Found infinite values in predictions"


# ─────────────────────────────────────────────────────────
# B) MODEL ACCURACY
# ─────────────────────────────────────────────────────────

class TestModelAccuracy:
    """
    Regression quality gates.
    These thresholds represent the minimum acceptable performance.
    If a retrained model fails these, it should NOT go to production.
    """

    MAE_THRESHOLD  = 3.0   # average error must be under 3 units/day
    RMSE_THRESHOLD = 5.0   # worst-case error must be under 5 units/day

    def test_mae_within_threshold(self, model, sample_df):
        """
        MAE (Mean Absolute Error) is the core accuracy metric.
        An MAE of 3.0 means on average the forecast is off by ≤ 3 units/day.
        """
        preds   = predict_batch(model, sample_df)
        actuals = sample_df["sales"].values
        mae     = float(np.mean(np.abs(preds - actuals)))

        # Attach the actual value to the test report
        pytest.mae_result = mae

        assert mae < self.MAE_THRESHOLD, (
            f"MAE {mae:.4f} exceeds threshold {self.MAE_THRESHOLD}. "
            "Model performance has degraded — consider retraining."
        )

    def test_rmse_within_threshold(self, model, sample_df):
        """
        RMSE (Root Mean Squared Error) penalises large prediction errors heavily.
        A high RMSE means the model occasionally predicts very badly.
        """
        preds   = predict_batch(model, sample_df)
        actuals = sample_df["sales"].values
        rmse    = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        assert rmse < self.RMSE_THRESHOLD, (
            f"RMSE {rmse:.4f} exceeds threshold {self.RMSE_THRESHOLD}."
        )

    def test_zero_sales_days_not_massively_overforecast(self, model, sample_df):
        """
        On days when actual sales = 0, the model should not predict huge numbers.
        This catches the 'over-ordering' problem we identified earlier.
        Max acceptable forecast on a zero-sale day = 3.0 units.
        """
        zero_days = sample_df[sample_df["sales"] == 0].copy()
        if len(zero_days) == 0:
            pytest.skip("No zero-sale rows in sample")

        preds = predict_batch(model, zero_days)
        mean_overforecast = float(preds.mean())

        assert mean_overforecast < 3.0, (
            f"Model over-forecasts on zero-sale days: {mean_overforecast:.2f} avg. "
            "This leads to unnecessary stock ordering."
        )


# ─────────────────────────────────────────────────────────
# C) COLD START RESILIENCE
# ─────────────────────────────────────────────────────────

class TestColdStart:
    """
    What happens when a brand new product arrives in the warehouse?
    The model must not crash. It should fall back gracefully using
    category-level knowledge (as specified in FR2 of Requirements).
    """

    def test_unknown_item_does_not_crash(self, model, alien_item_row):
        """Model must return a result even for a completely unknown item ID."""
        try:
            preds = predict_batch(model, alien_item_row)
            assert len(preds) == 1
        except Exception as e:
            pytest.fail(
                f"Model crashed on unknown item 'FOODS_X_9999': {e}\n"
                "Cold Start fallback is not working. See FR2 in Requirements."
            )

    def test_unknown_item_prediction_is_non_negative(self, model, alien_item_row):
        """Even for unknown items, the forecast must be ≥ 0."""
        preds = predict_batch(model, alien_item_row)
        assert preds[0] >= 0, (
            f"Negative prediction {preds[0]:.2f} for unknown item."
        )

    def test_unknown_item_prediction_is_not_nan(self, model, alien_item_row):
        """Unknown item must not produce NaN — that would crash the dashboard."""
        preds = predict_batch(model, alien_item_row)
        assert not np.isnan(preds[0]), (
            "Cold Start produced NaN. The model has no fallback for unknown categories."
        )

    def test_cold_start_uses_category_knowledge(self, model, sample_df, alien_item_row):
        """
        The unknown item (FOODS_X) shares the parent category 'FOODS' with real items.
        LightGBM uses category-level splits, so the prediction should be in a
        realistic range (not 0, not 1000+).
        """
        real_foods = sample_df[sample_df["cat_id"] == "FOODS"].copy()
        if len(real_foods) == 0:
            pytest.skip("No FOODS items in sample")

        real_preds  = predict_batch(model, real_foods)
        alien_pred  = predict_batch(model, alien_item_row)[0]
        real_median = float(np.median(real_preds))

        # The cold-start prediction should be within 10x of the category median.
        # If it's wildly different, our fallback is broken.
        assert alien_pred < real_median * 10, (
            f"Cold start prediction {alien_pred:.2f} is unrealistically high "
            f"vs. FOODS category median {real_median:.2f}. Fallback may be misconfigured."
        )


# ─────────────────────────────────────────────────────────
# D) RETRAIN TRIGGER LOGIC
# ─────────────────────────────────────────────────────────

class TestRetrainTriggers:
    """
    Each of the 4 triggers (time, drift, volume, new-state) must
    fire exactly when expected and stay silent otherwise.
    """

    def test_volume_trigger_no_file(self, tmp_path):
        """If real_time_sales.csv doesn't exist → volume trigger = False."""
        original  = rm.REALTIME_DATA
        rm.REALTIME_DATA = str(tmp_path / "nonexistent.csv")
        fired, reason = rm.check_volume_trigger()
        rm.REALTIME_DATA = original
        assert not fired
        assert "No real-time data file" in reason

    def test_volume_trigger_fires_when_large(self, tmp_path):
        """If the file has >= 1000 rows → volume trigger = True."""
        fake_csv = tmp_path / "real_time_sales.csv"
        pd.DataFrame({"sale": range(1001)}).to_csv(fake_csv, index=False)
        original = rm.REALTIME_DATA
        rm.REALTIME_DATA = str(fake_csv)
        fired, reason = rm.check_volume_trigger()
        rm.REALTIME_DATA = original
        assert fired, f"Volume trigger should have fired. Reason: {reason}"

    def test_drift_trigger_fires_on_bad_mae(self):
        """If current MAE is 50% above baseline → drift trigger = True."""
        baseline = get_baseline_mae()
        if baseline is None:
            pytest.skip("No baseline MAE in registry")
        bad_mae = baseline * 1.6   # 60% worse than baseline
        fired, reason = rm.check_performance_drift(bad_mae)
        assert fired, (
            f"Drift trigger should have fired for MAE {bad_mae:.4f} "
            f"vs baseline {baseline:.4f}. Reason: {reason}"
        )

    def test_drift_trigger_silent_on_good_mae(self):
        """If current MAE is only 5% above baseline → no drift trigger."""
        baseline = get_baseline_mae()
        if baseline is None:
            pytest.skip("No baseline MAE in registry")
        good_mae = baseline * 1.05  # only 5% worse — acceptable
        fired, reason = rm.check_performance_drift(good_mae)
        assert not fired, (
            f"Drift trigger fired incorrectly for acceptable MAE {good_mae:.4f}. "
            f"Reason: {reason}"
        )

    def test_new_state_trigger_known_states(self, tmp_path):
        """If all state_ids are known (CA, TX, WI) → no trigger."""
        fake_csv = tmp_path / "real_time_sales.csv"
        pd.DataFrame({"state_id": ["CA", "TX", "WI"]}).to_csv(fake_csv, index=False)
        original = rm.REALTIME_DATA
        rm.REALTIME_DATA = str(fake_csv)
        fired, reason = rm.check_new_state_trigger()
        rm.REALTIME_DATA = original
        assert not fired

    def test_new_state_trigger_unknown_state(self, tmp_path):
        """If an unknown state_id appears (e.g. NY) → trigger fires."""
        fake_csv = tmp_path / "real_time_sales.csv"
        pd.DataFrame({"state_id": ["CA", "NY"]}).to_csv(fake_csv, index=False)
        original = rm.REALTIME_DATA
        rm.REALTIME_DATA = str(fake_csv)
        fired, reason = rm.check_new_state_trigger()
        rm.REALTIME_DATA = original
        assert fired, "New-state trigger should fire for unknown state 'NY'"


# ─────────────────────────────────────────────────────────
# E) MODEL REGISTRY
# ─────────────────────────────────────────────────────────

class TestModelRegistry:
    """
    The registry is our 'source of truth' for model versions.
    It must always be consistent — exactly one production model.
    """

    def test_registry_is_not_empty(self, registry):
        """At least one model must be registered to run the dashboard."""
        assert len(registry) > 0, (
            "Model registry is empty. Run train.py to register a model."
        )

    def test_exactly_one_production_model(self, registry):
        """There must be exactly one model flagged as is_production=True."""
        prod_models = [r for r in registry if r.get("is_production")]
        assert len(prod_models) == 1, (
            f"Expected exactly 1 production model, found {len(prod_models)}. "
            "Registry is corrupt. Check model_registry.py > register_model()."
        )

    def test_production_model_has_best_mae(self, registry):
        """The production model should be the one with the lowest MAE."""
        prod    = get_production_model_entry()
        best    = get_best_model_entry()
        if prod is None or best is None:
            pytest.skip("Registry empty")
        assert prod["version"] == best["version"], (
            f"Production model is {prod['version']} (MAE {prod['mae']:.4f}) "
            f"but best model is {best['version']} (MAE {best['mae']:.4f}). "
            "Auto-promotion logic may be broken."
        )

    def test_all_registry_entries_have_required_fields(self, registry):
        """Every registry entry must have the standard set of fields."""
        required = {"version", "date", "mae", "rmse", "is_production", "model_file"}
        for entry in registry:
            missing = required - set(entry.keys())
            assert not missing, (
                f"Registry entry {entry.get('version')} is missing fields: {missing}"
            )
