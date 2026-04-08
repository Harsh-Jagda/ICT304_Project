import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from src.ui.data_manager import predict_batch
from src.ui.state_controller import init_simulator_state, advance_n_days
from config.constants import DEFAULT_UNIT_PRICE_USD, DEFAULT_LEAD_TIME_DAYS
# FIX #1: import the canonical ROP formula — no more copy-paste
from src.business.recommendation import calculate_rop

def render_page(model, store_df: pd.DataFrame, sales_dict: dict, lt_cfg: dict):
    if 'si_ready' not in st.session_state:
        # Pass the selected store based on the active dataframe instead of hardcoding 'CA_1'
        active_store = store_df['store_id'].iloc[0] if len(store_df) > 0 else 'CA_1'
        init_simulator_state(store_df, active_store)
        
    date_current = st.session_state.si_date
    date_str = date_current.strftime('%Y-%m-%d')
    date_real_world = date_current.strftime('%b %d, %Y')
    
    st.markdown(f"### 🟢 Live Procurement Hub | <span style='color:#8B949E; font-size: 20px;'>Virtual Date: {date_real_world}</span>", unsafe_allow_html=True)
    st.markdown("Interactive human-in-the-loop dashboard. AI actively monitors inventory and forecasts demand.")
    
    with st.expander("Glossary & Key Concepts (Click to expand)", expanded=False):
        st.markdown("""
        **For Business & Operations (Sales & Ops):**
        * **Virtual Date:** Represents the current date within our 365-day fast-forward simulation. You are playing the role of a manager on this specific day in history.
        * **Virtual Lost Revenue:** The exact dollar amount of sales you missed because the warehouse ran out of stock (Stockouts). The AI's primary goal is to minimize this to near zero.
        * **Actionable Alerts & Budget:** How many items realistically need reordering *today* to avoid empty shelves, and the capital required.
        
        **For Tech & Supply Chain Strategy:**
        * **Explainable AI (XAI):** We do not use "Black Box" models. For every recommendation, the AI generates a human-readable mathematical proof explaining exactly *why* the order is needed.
        * **ROP (Reorder Point):** The algorithmic threshold. If `Stock + Incoming < ROP`, we order. Formula: `Predicted Demand × Lead Time + Safety Stock`.
        * **Safety Stock:** A statistical buffer calculated dynamically using the 7-day rolling standard deviation of demand (`1.645 × σ × √LT`) to cover 95% of uncertainty.
        * **Lead Time (LT):** The physical delay (in days) between authorizing a Purchase Order and the stock arriving on the shelf.
        """)
    
    col_a, col_b, col_c = st.columns([1, 1, 1])
    
    day_df = store_df[store_df['date'] == date_current].copy()
    if len(day_df) == 0:
        st.error("Reached End of Database.")
        return
        
    preds = predict_batch(model, day_df)
    day_df['pred_demand'] = preds
    
    recs = []
    total_budget_needed = 0
    inv = st.session_state.si_inventory
    dels = st.session_state.si_deliveries
    
    for _, row in day_df.iterrows():
        item = row['item_id']
        if item in st.session_state.si_approved_today:
            continue
            
        stock = inv.get(item, 0)
        on_order = sum([d['qty'] for d in dels if d['item'] == item and d['day'] > date_str])
        
        # Read per-category config (lead time + z-score)
        # dept is typically 'HOBBIES_1', so split to get 'HOBBIES'
        dept = row.get('dept_id', '')
        cat_key = dept.split('_')[0] if isinstance(dept, str) else str(dept).split('_')[0]
        lt_cfg_dept = lt_cfg.get(cat_key, {})
        lt_val = lt_cfg_dept.get('lead_time_days', DEFAULT_LEAD_TIME_DAYS)
        z_score = lt_cfg_dept.get('service_level_z', 1.645)

        std_7 = row['rolling_std_7'] if not pd.isnull(row['rolling_std_7']) else 0.5
        pred = row['pred_demand']

        # FIX #1: use the canonical function — single source of truth
        _safety_stock, rop = calculate_rop(pred, lt_val, std_7, z_score)

        if stock + on_order < rop:
            order_qty = max(0, int(rop - stock - on_order + (pred * lt_val)))
            if order_qty > 0:
                recs.append({
                    'item': item, 'stock': stock, 'on_order': on_order,
                    'pred': pred, 'rop': rop, 'order_qty': order_qty,
                    'lt': lt_val
                })
                total_budget_needed += (order_qty * DEFAULT_UNIT_PRICE_USD)  # FIX #2: use constant
                
    recs.sort(key=lambda x: (x['stock'] + x['on_order']) - x['rop'])
    
    col_a.markdown(f"""
    <div class='metric-box'>
        <div class='metric-title'>Actionable Alerts</div>
        <div class='metric-value-base'>{len(recs)} SKUs</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_b.markdown(f"""
    <div class='metric-box'>
        <div class='metric-title'>Recommended Budget</div>
        <div class='metric-value-neutral'>${total_budget_needed:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    total_losses = sum(st.session_state.si_missed_sales.values()) * DEFAULT_UNIT_PRICE_USD  # FIX #2
    total_sales = sum(st.session_state.si_history_sales.values()) * DEFAULT_UNIT_PRICE_USD  # FIX #2
    loss_ratio = (total_losses / (total_sales + total_losses + 1e-6)) * 100
    
    col_c.markdown(f"""
    <div class='metric-box'>
        <div class='metric-title'>Virtual Lost Revenue</div>
        <div class='metric-value-base'>${total_losses:,.0f} <span style='font-size:16px;'>({loss_ratio:.1f}%)</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1, 3])
    with act_col1:
        if st.button("+1 Day", type="primary", use_container_width=True):
            advance_n_days(sales_dict, int(1))
            st.rerun()
    with act_col2:
        if st.button("+7 Days", type="primary", use_container_width=True):
            advance_n_days(sales_dict, int(7))
            st.rerun()
    with act_col3:
        if st.button("+28 Days", type="primary", use_container_width=True):
            advance_n_days(sales_dict, int(28))
            st.rerun()

    with act_col4:
        if len(recs) > 0:
            if st.button("Approve All Current Recommendations", use_container_width=True):
                for r in recs:
                    st.session_state.si_approved_today.add(r['item'])
                    arr_date = (date_current + timedelta(days=r['lt'])).strftime('%Y-%m-%d')
                    st.session_state.si_deliveries.append({
                        'item': r['item'], 'qty': r['order_qty'], 'day': arr_date
                    })
                st.rerun()

    st.markdown("---")
    
    # NEW FEATURE: Show ordered items and time until arrival
    with st.expander(f"Pending Incoming Deliveries ({len(st.session_state.si_deliveries)})", expanded=False):
        if not st.session_state.si_deliveries:
            st.info("No incoming deliveries at the moment.")
        else:
            del_df = pd.DataFrame(st.session_state.si_deliveries)
            del_df['Days Until Arrival'] = (pd.to_datetime(del_df['day']) - pd.to_datetime(date_current)).dt.days
            del_df = del_df.rename(columns={'item': 'SKU', 'qty': 'Quantity', 'day': 'Arrival Date'})
            del_df = del_df[['SKU', 'Quantity', 'Arrival Date', 'Days Until Arrival']].sort_values('Days Until Arrival')
            st.dataframe(del_df, use_container_width=True, hide_index=True)
            
    st.markdown("---")
    
    if len(recs) == 0:
        st.success("✅ Excellent! Warehouse is fully stocked and optimized for current demand.")
    
    for r in recs[:20]:
        with st.expander(f"🔴 SKU: {r['item']}  |  Stock: {r['stock']}  |  AI Recommended Order: {r['order_qty']}"):
            x_col1, x_col2 = st.columns([1, 1])
            with x_col1:
                st.markdown("**Explainable AI (XAI)**")
                st.info(f"Inventory (+Pending) drops below Safety Threshold.\n\n"
                        f"• Safety Threshold (ROP): **{r['rop']:.1f}**\n"
                        f"• Predicted Avg. Demand: **{r['pred']:.2f} per day**\n"
                        f"• Lead Time: **{r['lt']} days**\n"
                        f"The AI suggests ordering exactly **{r['order_qty']}** units to prevent stockout.")
            with x_col2:
                st.markdown("**Human Override**")
                form_key = f"form_{r['item']}"
                with st.form(form_key):
                    qty = st.number_input("Final Order Quantity:", value=r['order_qty'], min_value=0, step=1)
                    if st.form_submit_button("Approve Purchase Order", type="primary"):
                        st.session_state.si_approved_today.add(r['item'])
                        arr_date = (date_current + timedelta(days=r['lt'])).strftime('%Y-%m-%d')
                        st.session_state.si_deliveries.append({
                            'item': r['item'], 'qty': qty, 'day': arr_date
                        })
                        st.rerun()
