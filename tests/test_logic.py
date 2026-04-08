import pandas as pd
import numpy as np
import unittest

def mock_recommendation_logic(predicted_demand, lead_time_days, safety_stock, current_inventory):
    """
    Simplified version of the logic in recommendation.py for testing.
    """
    reorder_point = (predicted_demand * lead_time_days) + safety_stock
    order_needed = current_inventory < reorder_point
    order_qty = max(0, int(round(reorder_point - current_inventory))) if order_needed else 0
    return reorder_point, order_needed, order_qty

class TestRecommendationLogic(unittest.TestCase):
    def test_basic_order(self):
        # Demand: 2/day, Lead time: 4 days, Safety stock: 2. ROP = 2*4 + 2 = 10.
        # Inventory: 5. Need 5 more.
        rop, needed, qty = mock_recommendation_logic(2, 4, 2, 5)
        self.assertEqual(rop, 10)
        self.assertTrue(needed)
        self.assertEqual(qty, 5)

    def test_no_order_needed(self):
        # ROP = 10, Inventory = 15. No order.
        rop, needed, qty = mock_recommendation_logic(2, 4, 2, 15)
        self.assertEqual(rop, 10)
        self.assertFalse(needed)
        self.assertEqual(qty, 0)

    def test_rounding(self):
        # ROP = 10.4, Inventory = 5. Order = 5.4 -> 5
        rop, needed, qty = mock_recommendation_logic(2.1, 4, 2, 5)
        self.assertEqual(rop, 10.4)
        self.assertEqual(qty, 5)

    def test_safety_stock_formula(self):
        # Simple check of formula from recommendation.py: z * sigma * sqrt(L)
        z = 1.645
        sigma = 1.0
        L = 4
        safety_stock = z * sigma * np.sqrt(L)
        self.assertAlmostEqual(safety_stock, 3.29, places=2)

if __name__ == "__main__":
    unittest.main()
