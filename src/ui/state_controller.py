import streamlit as st
import numpy as np
from datetime import timedelta
import pandas as pd

def init_simulator_state(store_df: pd.DataFrame, store_id: str):
    max_date = store_df['date'].max()
    start_date = max_date - timedelta(days=30)
    
    items = store_df['item_id'].unique()
    np.random.seed(42) 
    init_stocks = {item: int(np.random.randint(15, 30)) for item in items}
    
    st.session_state.si_date = start_date
    st.session_state.si_inventory = init_stocks
    st.session_state.si_deliveries = []
    st.session_state.si_approved_today = set()
    st.session_state.si_ready = True
    st.session_state.si_missed_sales = {item: 0.0 for item in items}
    st.session_state.si_history_sales = {item: 0.0 for item in items}

def advance_one_day(sales_dict):
    st.session_state.si_date += timedelta(days=1)
    new_date_str = st.session_state.si_date.strftime('%Y-%m-%d')
    
    for d in list(st.session_state.si_deliveries):
        if d['day'] <= new_date_str:
            st.session_state.si_inventory[d['item']] += d['qty']
    st.session_state.si_deliveries = [d for d in st.session_state.si_deliveries if d['day'] > new_date_str]
    
    actual_sales_today = sales_dict.get(new_date_str, {})
    for item, true_demand in actual_sales_today.items():
        if true_demand > 0:
            st.session_state.si_history_sales[item] += true_demand
            if st.session_state.si_inventory[item] >= true_demand:
                st.session_state.si_inventory[item] -= true_demand
            else:
                missed = true_demand - st.session_state.si_inventory[item]
                st.session_state.si_missed_sales[item] += missed
                st.session_state.si_inventory[item] = 0
    
    st.session_state.si_approved_today = set()

def advance_n_days(sales_dict, n=1):
    for _ in range(n):
        advance_one_day(sales_dict)
