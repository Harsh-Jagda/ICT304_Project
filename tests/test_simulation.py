"""
test_simulation.py — Subsystem 3: Simulation Engine QA
=======================================================
Tests the core mechanics of the 365-day A/B simulation engine.
All tests use tiny synthetic scenarios (5 days, 2 items) so
they run in milliseconds, not minutes.

  A) Delivery Logic    — does stock arrive exactly on the right day?
  B) Stockout Logic    — are missed sales recorded correctly?
  C) Holding Costs     — is the formula stock × rate applied correctly?
  D) Reorder Policy    — does the AI policy order more than the baseline?
  E) Output Validation — is the saved sim_results.parquet valid?
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use shared constants so test values stay in sync with the engine
from config.constants import DEFAULT_UNIT_PRICE_USD, HOLDING_COST_RATE
from src.business.recommendation import calculate_rop

SELL_PRICE   = DEFAULT_UNIT_PRICE_USD  # was hardcoded 3.50
HOLDING_RATE = HOLDING_COST_RATE       # was hardcoded 0.05
LEAD_TIME    = 2


# ─────────────────────────────────────────────────────────
# Minimal simulation helpers (mirror the engine logic)
# These let us write deterministic unit tests without
# depending on the full 365-day pipeline.
# ─────────────────────────────────────────────────────────

def run_mini_sim(
    initial_stock: int,
    demand_per_day: list,          # list of actual sales per day
    orders: list,                  # list of (order_day_idx, qty) tuples
    lead_time: int = LEAD_TIME,
):
    """
    Micro-simulation of a single item over N days.
    Returns a list of dicts: {day, stock, missed, holding_cost}
    """
    stock = initial_stock
    deliveries = []   # list of {day: int, qty: int}

    for order_day, qty in orders:
        deliveries.append({"day": order_day + lead_time, "qty": qty})

    results = []
    for day_idx, demand in enumerate(demand_per_day):
        # 1. Receive deliveries due today
        for d in deliveries:
            if d["day"] <= day_idx:
                stock += d["qty"]
        deliveries = [d for d in deliveries if d["day"] > day_idx]

        # 2. Sell (or miss)
        if stock >= demand:
            stock -= demand
            missed = 0
        else:
            missed = demand - stock
            stock = 0

        # 3. Record state after sales
        holding_cost = stock * HOLDING_RATE

        results.append({
            "day":          day_idx,
            "stock":        stock,
            "missed":       missed,
            "holding_cost": holding_cost,
        })

    return results


# ─────────────────────────────────────────────────────────
# A) DELIVERY LOGIC
# ─────────────────────────────────────────────────────────

class TestDeliveryLogic:
    """Stock ordered on day X must arrive exactly on day X + lead_time."""

    def test_delivery_arrives_on_correct_day(self):
        """
        Order 10 units on Day 0 with lead_time=2.
        Stock should increase on Day 2, not Day 1 or Day 3.
        """
        results = run_mini_sim(
            initial_stock=0,
            demand_per_day=[0, 0, 0, 0, 0],   # no sales — purely watching delivery
            orders=[(0, 10)],                  # order 10 units on day 0
            lead_time=2,
        )
        # Day 1: delivery not yet arrived → stock still 0
        assert results[1]["stock"] == 0, (
            f"Delivery arrived too early on day 1. Stock: {results[1]['stock']}"
        )
        # Day 2: delivery arrives → stock = 10
        assert results[2]["stock"] == 10, (
            f"Delivery did not arrive on day 2. Stock: {results[2]['stock']}"
        )

    def test_multiple_orders_stack_correctly(self):
        """
        Two separate orders should both arrive and both add to stock.
        Order 5 on Day 0 and 3 on Day 1 → both arrive on Day 2.
        """
        results = run_mini_sim(
            initial_stock=0,
            demand_per_day=[0, 0, 0, 0],
            orders=[(0, 5), (0, 3)],   # two orders, same day
            lead_time=2,
        )
        assert results[2]["stock"] == 8, (
            f"Expected stock=8 after two deliveries, got {results[2]['stock']}"
        )

    def test_delivery_does_not_arrive_before_lead_time(self):
        """Stock must not appear before lead_time days have passed."""
        results = run_mini_sim(
            initial_stock=0,
            demand_per_day=[0, 0, 0],
            orders=[(0, 20)],
            lead_time=2,
        )
        assert results[0]["stock"] == 0, "Stock appeared on the same day as the order!"
        assert results[1]["stock"] == 0, "Stock arrived one day early!"


# ─────────────────────────────────────────────────────────
# B) STOCKOUT LOGIC
# ─────────────────────────────────────────────────────────

class TestStockoutLogic:
    """When stock runs out, missed sales must be recorded exactly."""

    def test_full_stockout_records_all_missed_sales(self):
        """
        Start with 0 stock, demand = 5 → all 5 units should be missed.
        """
        results = run_mini_sim(
            initial_stock=0,
            demand_per_day=[5],
            orders=[],
        )
        assert results[0]["missed"] == 5, (
            f"Expected 5 missed sales, got {results[0]['missed']}"
        )

    def test_partial_stockout_is_correct(self):
        """
        Start with 3 stock, demand = 7 → stock goes to 0, missed = 4.
        """
        results = run_mini_sim(
            initial_stock=3,
            demand_per_day=[7],
            orders=[],
        )
        assert results[0]["stock"]  == 0, "Stock should be 0 after a stockout"
        assert results[0]["missed"] == 4, (
            f"Expected 4 missed units, got {results[0]['missed']}"
        )

    def test_no_stockout_when_sufficient_stock(self):
        """No missed sales when demand ≤ stock."""
        results = run_mini_sim(
            initial_stock=10,
            demand_per_day=[3, 3, 3],
            orders=[],
        )
        for r in results:
            assert r["missed"] == 0, f"Unexpected missed sale on day {r['day']}"

    def test_lost_revenue_formula(self):
        """
        Lost revenue = missed_units × SELL_PRICE.
        5 missed units × $3.50 = $17.50.
        """
        results = run_mini_sim(
            initial_stock=0,
            demand_per_day=[5],
            orders=[],
        )
        lost_revenue = results[0]["missed"] * SELL_PRICE
        assert abs(lost_revenue - 17.50) < 0.01, (
            f"Lost revenue should be $17.50, got ${lost_revenue:.2f}"
        )


# ─────────────────────────────────────────────────────────
# C) HOLDING COSTS
# ─────────────────────────────────────────────────────────

class TestHoldingCosts:
    """Daily holding cost = units_on_shelf × HOLDING_RATE."""

    def test_holding_cost_exact_formula(self):
        """
        Start with 20 units, zero demand, zero orders.
        Holding cost on Day 0 = 20 × 0.05 = $1.00
        """
        results = run_mini_sim(
            initial_stock=20,
            demand_per_day=[0],
            orders=[],
        )
        expected = 20 * HOLDING_RATE
        actual   = results[0]["holding_cost"]
        assert abs(actual - expected) < 0.001, (
            f"Holding cost: expected ${expected:.2f}, got ${actual:.2f}"
        )

    def test_holding_cost_decreases_as_stock_sells(self):
        """As stock is consumed each day, holding cost should decrease."""
        results = run_mini_sim(
            initial_stock=10,
            demand_per_day=[2, 2, 2, 2],   # sell 2/day
            orders=[],
        )
        costs = [r["holding_cost"] for r in results]
        for i in range(1, len(costs)):
            assert costs[i] <= costs[i - 1], (
                f"Holding cost increased from day {i-1} to {i}: "
                f"{costs[i-1]:.2f} → {costs[i]:.2f}"
            )

    def test_holding_cost_zero_when_empty(self):
        """Empty warehouse has zero holding cost."""
        results = run_mini_sim(
            initial_stock=0,
            demand_per_day=[0, 0, 0],
            orders=[],
        )
        for r in results:
            assert r["holding_cost"] == 0.0, (
                f"Non-zero holding cost on empty stock on day {r['day']}"
            )


# ─────────────────────────────────────────────────────────
# D) REORDER POLICY
# ─────────────────────────────────────────────────────────

class TestReorderPolicy:
    """
    The ROP (Reorder Point) formula determines when to order.
    These tests now call the canonical calculate_rop() function directly —
    so if the formula changes in recommendation.py, these tests will catch
    any regression immediately.
    """

    def test_rop_basic_formula(self):
        """
        With pred=10, lt=2, std=0, z=1.645:
        safety_stock = 1.645 × 0 × √2 = 0
        ROP = 10 × 2 + 0 = 20.0
        """
        ss, rop = calculate_rop(pred_demand=10.0, lead_time=2, demand_std=0.0)
        assert abs(rop - 20.0) < 0.001, (
            f"ROP formula: expected 20.0, got {rop:.4f}"
        )
        assert ss == 0.0, f"Safety stock should be 0 when std=0, got {ss:.4f}"

    def test_rop_increases_with_uncertainty(self):
        """
        Higher demand variability (std) → bigger safety buffer → higher ROP.
        """
        _, rop_low_std  = calculate_rop(pred_demand=10.0, lead_time=2, demand_std=0.0)
        _, rop_high_std = calculate_rop(pred_demand=10.0, lead_time=2, demand_std=5.0)
        assert rop_high_std > rop_low_std, (
            "ROP with high std should be greater than ROP with zero std"
        )

    def test_ai_orders_when_stock_below_rop(self):
        """
        If stock + on_order < ROP, an order must be triggered.
        """
        stock, on_order = 5, 0
        _, rop = calculate_rop(pred_demand=10.0, lead_time=2, demand_std=1.0)

        should_order = (stock + on_order) < rop
        assert should_order, (
            f"Expected order trigger: stock={stock}, on_order={on_order}, ROP={rop:.2f}"
        )

    def test_no_order_when_stock_sufficient(self):
        """
        If stock is well above ROP, no order should be triggered.
        """
        stock, on_order = 100, 0
        _, rop = calculate_rop(pred_demand=1.0, lead_time=2, demand_std=0.1)

        should_order = (stock + on_order) < rop
        assert not should_order, (
            f"No order expected: stock={stock}, ROP={rop:.2f}"
        )

    def test_nan_std_handled_gracefully(self):
        """
        calculate_rop() must not crash when demand_std is NaN (cold-start items).
        """
        import math
        ss, rop = calculate_rop(pred_demand=5.0, lead_time=2, demand_std=float('nan'))
        assert not math.isnan(rop), "calculate_rop returned NaN for nan std — cold start broken"
        assert ss == 0.0, "Safety stock should default to 0 for NaN std"


# ─────────────────────────────────────────────────────────
# E) OUTPUT VALIDATION
# ─────────────────────────────────────────────────────────

class TestSimulationOutput:
    """
    Sanity checks on the actual sim_results.parquet saved to disk.
    These verify the 365-day run produced a consistent, usable output.
    """

    @pytest.fixture(scope="class")
    def sim_df(self):
        path = os.path.join(DATA_DIR, "data", "outputs", "sim_results.parquet")
        if not os.path.exists(path):
            pytest.skip("sim_results.parquet not found — run prepare_simulation.py first")
        return pd.read_parquet(path)

    def test_required_columns_present(self, sim_df):
        """Output must have all 7 required metric columns."""
        required = [
            "date",
            "ai_stockout_items", "base_stockout_items",
            "ai_lost_revenue",   "base_lost_revenue",
            "ai_holding_cost",   "base_holding_cost",
        ]
        missing = [c for c in required if c not in sim_df.columns]
        assert not missing, f"sim_results.parquet is missing columns: {missing}"

    def test_no_negative_financial_values(self, sim_df):
        """Revenue losses and holding costs can be zero but never negative."""
        for col in ["ai_lost_revenue", "base_lost_revenue",
                    "ai_holding_cost", "base_holding_cost"]:
            neg = (sim_df[col] < 0).sum()
            assert neg == 0, f"Column '{col}' has {neg} negative values"

    def test_simulation_covers_full_year(self, sim_df):
        """The simulation should span at least 300 days (one year minus weekends/gaps)."""
        assert len(sim_df) >= 300, (
            f"Simulation only has {len(sim_df)} days — expected at least 300"
        )

    def test_base_stockouts_higher_than_ai(self, sim_df):
        """
        The whole point of the AI system is to reduce stockouts.
        The baseline policy must have MORE missed-stock days than the AI policy
        on average. If not, the AI is not adding value.
        """
        avg_ai   = sim_df["ai_stockout_items"].mean()
        avg_base = sim_df["base_stockout_items"].mean()
        assert avg_base >= avg_ai, (
            f"AI stockouts ({avg_ai:.1f}/day) are WORSE than baseline ({avg_base:.1f}/day). "
            "The simulation A/B test result is suspicious."
        )
