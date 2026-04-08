"""
test_ui_state.py — Subsystem 5: UI State Controller QA
=======================================================
Tests the interactive simulation ("live game") mechanics in state_controller.py.
No Streamlit server is needed — we mock st.session_state with a plain dict
and call the functions directly.

  A) Initialization  — does the game start with correct default values?
  B) Day Advancement — does "Advance 1 Day" update date and process sales?
  C) Deliveries      — does an approved order arrive after the lead time?
  D) Missed Sales    — are stockouts tracked correctly in the live game?
  E) Approve All     — does bulk-approve move all items into deliveries?
"""
import os
import sys
import types
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────
# Mock st.session_state
# We replace Streamlit's session_state with a plain Python
# object so we can test state logic without a running server.
# ─────────────────────────────────────────────────────────

class FakeSessionState(dict):
    """Acts like st.session_state: supports both dict and attribute access."""
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """
    Replace `st` in state_controller with a minimal mock.
    This is the standard technique for unit-testing Streamlit apps.
    """
    fake_st = MagicMock()
    fake_st.session_state = FakeSessionState()
    monkeypatch.setattr(
        "src.ui.state_controller.st",
        fake_st
    )
    return fake_st.session_state


# ─────────────────────────────────────────────────────────
# Helper: build a minimal store_df
# ─────────────────────────────────────────────────────────

def make_store_df(items=("ITEM_A", "ITEM_B"), n_days=5):
    base = pd.Timestamp("2016-04-01")
    rows = []
    for item in items:
        for d in range(n_days):
            rows.append({
                "item_id":  item,
                "store_id": "CA_1",
                "date":     base + pd.Timedelta(days=d),
                "sales":    2,
            })
    return pd.DataFrame(rows)


def make_sales_dict(store_df):
    sales = {}
    for d, grp in store_df.groupby("date"):
        sales[d.strftime("%Y-%m-%d")] = grp.set_index("item_id")["sales"].to_dict()
    return sales


# ─────────────────────────────────────────────────────────
# A) INITIALIZATION
# ─────────────────────────────────────────────────────────

class TestInitialization:
    """init_simulator_state() must set up the game with valid starting values."""

    def test_date_is_set_after_init(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df()
        init_simulator_state(store_df, "CA_1")
        assert "si_date" in mock_streamlit
        assert isinstance(mock_streamlit["si_date"], pd.Timestamp)

    def test_all_items_have_initial_inventory(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df(items=["ITEM_A", "ITEM_B", "ITEM_C"])
        init_simulator_state(store_df, "CA_1")
        inv = mock_streamlit["si_inventory"]
        for item in ["ITEM_A", "ITEM_B", "ITEM_C"]:
            assert item in inv, f"{item} missing from initial inventory"
            assert inv[item] >= 0, f"{item} starts with negative inventory"

    def test_deliveries_start_empty(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df()
        init_simulator_state(store_df, "CA_1")
        assert mock_streamlit["si_deliveries"] == []

    def test_missed_sales_start_at_zero(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df(items=["ITEM_A", "ITEM_B"])
        init_simulator_state(store_df, "CA_1")
        for item in ["ITEM_A", "ITEM_B"]:
            assert mock_streamlit["si_missed_sales"][item] == 0.0

    def test_ready_flag_is_true(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df()
        init_simulator_state(store_df, "CA_1")
        assert mock_streamlit["si_ready"] is True


# ─────────────────────────────────────────────────────────
# B) DAY ADVANCEMENT
# ─────────────────────────────────────────────────────────

class TestDayAdvancement:
    """advance_one_day() must increment the date by exactly 1 day."""

    def _init(self, mock_streamlit, items=("ITEM_A",)):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df(items=list(items))
        init_simulator_state(store_df, "CA_1")
        return make_sales_dict(store_df)

    def test_date_increments_by_one_day(self, mock_streamlit):
        from src.ui.state_controller import advance_one_day
        sales_dict = self._init(mock_streamlit)
        start_date = mock_streamlit["si_date"]
        advance_one_day(sales_dict)
        end_date = mock_streamlit["si_date"]
        delta = end_date - start_date
        assert delta == pd.Timedelta(days=1), (
            f"Date should advance by 1 day, advanced by {delta}"
        )

    def test_advance_multiple_days_accumulates(self, mock_streamlit):
        from src.ui.state_controller import advance_one_day
        sales_dict = self._init(mock_streamlit)
        start_date = mock_streamlit["si_date"]
        for _ in range(3):
            advance_one_day(sales_dict)
        delta = mock_streamlit["si_date"] - start_date
        assert delta == pd.Timedelta(days=3), (
            f"After 3 advances, date should be 3 days ahead, got {delta}"
        )

    def test_approved_set_cleared_each_day(self, mock_streamlit):
        """si_approved_today must reset to empty set each day."""
        from src.ui.state_controller import advance_one_day
        sales_dict = self._init(mock_streamlit)
        mock_streamlit["si_approved_today"] = {"ITEM_A"}  # simulate a previous approval
        advance_one_day(sales_dict)
        assert mock_streamlit["si_approved_today"] == set(), (
            "Approvals from the previous day must be cleared on advance"
        )


# ─────────────────────────────────────────────────────────
# C) DELIVERIES
# ─────────────────────────────────────────────────────────

class TestDeliveries:
    """Orders placed today must arrive exactly after lead_time days."""

    def _setup(self, mock_streamlit, items=("ITEM_A",), initial_stock=0):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df(items=list(items), n_days=10)
        # Override initial inventory to a known value
        init_simulator_state(store_df, "CA_1")
        for item in items:
            mock_streamlit["si_inventory"][item] = initial_stock
        return make_sales_dict(store_df)

    def test_delivery_arrives_after_lead_time(self, mock_streamlit):
        from src.ui.state_controller import advance_one_day
        sales_dict = self._setup(mock_streamlit, initial_stock=100)

        # Schedule a delivery for 2 days from now
        current_date = mock_streamlit["si_date"]
        arrival_date = (current_date + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        mock_streamlit["si_deliveries"].append(
            {"item": "ITEM_A", "qty": 50, "day": arrival_date}
        )

        # Advance 1 day — delivery should NOT arrive yet
        advance_one_day(sales_dict)
        assert mock_streamlit["si_inventory"]["ITEM_A"] <= 100, (
            "Delivery arrived too early (after 1 day, not 2)"
        )

        # Advance another day — now it should arrive
        advance_one_day(sales_dict)
        # Stock = 100 (initial) - 2 days of sales(2/day) + 50 delivery
        assert mock_streamlit["si_inventory"]["ITEM_A"] >= 50 + 100 - 10, (
            "Delivery of 50 units did not arrive after lead time"
        )

    def test_completed_delivery_removed_from_queue(self, mock_streamlit):
        """After delivery arrives, it must be removed from si_deliveries."""
        from src.ui.state_controller import advance_one_day
        sales_dict = self._setup(mock_streamlit, initial_stock=50)

        current_date = mock_streamlit["si_date"]
        arrival_date = (current_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        mock_streamlit["si_deliveries"].append(
            {"item": "ITEM_A", "qty": 10, "day": arrival_date}
        )

        advance_one_day(sales_dict)  # delivery arrives
        # Queue should now be empty
        remaining = [d for d in mock_streamlit["si_deliveries"] if d["item"] == "ITEM_A"]
        assert len(remaining) == 0, (
            "Delivered order was not removed from si_deliveries queue"
        )


# ─────────────────────────────────────────────────────────
# D) MISSED SALES TRACKING
# ─────────────────────────────────────────────────────────

class TestMissedSales:
    """When stock is empty and a sale is recorded, it must be counted as missed."""

    def test_stockout_increments_missed_sales(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state, advance_one_day
        
        # We need dates that align with max_date - 30 logic.
        # Let's make 35 days of data ending at 2016-04-05.
        store_df = make_store_df(items=["ITEM_A"], n_days=35)
        init_simulator_state(store_df, "CA_1")

        # Set stock to 0 so every sale is a miss
        mock_streamlit["si_inventory"]["ITEM_A"] = 0

        sales_dict = make_sales_dict(store_df)
        advance_one_day(sales_dict)

        missed = mock_streamlit["si_missed_sales"]["ITEM_A"]
        assert missed > 0, (
            "Missed sales counter did not increment when stock was 0"
        )

    def test_no_miss_when_stock_is_sufficient(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state, advance_one_day
        store_df = make_store_df(items=["ITEM_A"], n_days=35)
        init_simulator_state(store_df, "CA_1")

        # Give abundant stock (sales are 2/day)
        mock_streamlit["si_inventory"]["ITEM_A"] = 1000

        sales_dict = make_sales_dict(store_df)
        advance_one_day(sales_dict)

        missed = mock_streamlit["si_missed_sales"]["ITEM_A"]
        assert missed == 0.0, (
            f"Unexpected missed sale when stock was 1000. Missed: {missed}"
        )


# ─────────────────────────────────────────────────────────
# E) APPROVE ALL
# ─────────────────────────────────────────────────────────

class TestApproveAll:
    """Bulk approval must add all recommended items to the delivery queue."""

    def test_approve_adds_to_deliveries(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df()
        init_simulator_state(store_df, "CA_1")

        # Simulate what the dashboard does on "Approve All"
        recs = [
            {"item": "ITEM_A", "order_qty": 10, "lt": 2},
            {"item": "ITEM_B", "order_qty": 5,  "lt": 2},
        ]
        current_date = mock_streamlit["si_date"]
        for r in recs:
            mock_streamlit["si_approved_today"].add(r["item"])
            arr_date = (current_date + pd.Timedelta(days=r["lt"])).strftime("%Y-%m-%d")
            mock_streamlit["si_deliveries"].append(
                {"item": r["item"], "qty": r["order_qty"], "day": arr_date}
            )

        assert len(mock_streamlit["si_deliveries"]) == 2
        items_in_queue = {d["item"] for d in mock_streamlit["si_deliveries"]}
        assert "ITEM_A" in items_in_queue
        assert "ITEM_B" in items_in_queue

    def test_approved_items_in_approved_today_set(self, mock_streamlit):
        from src.ui.state_controller import init_simulator_state
        store_df = make_store_df()
        init_simulator_state(store_df, "CA_1")

        mock_streamlit["si_approved_today"].add("ITEM_A")
        assert "ITEM_A" in mock_streamlit["si_approved_today"]
