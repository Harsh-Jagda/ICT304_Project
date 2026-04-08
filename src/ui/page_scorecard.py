import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_page(sim_df: pd.DataFrame):
    st.markdown("###Strategic ROI & Financial Simulator")
    st.markdown("Comparing Traditional Baseline Ordering against our LightGBM AI Engine over a 365-day backtest.")
    
    with st.expander("Why this matters? (Value Proposition)", expanded=False):
        st.markdown("""
        **The Business Pitch (Why buy this?):**
        This dashboard perfectly illustrates the ultimate financial value of Machine Learning in supply chain.
        Traditional baseline algorithms (like Simple Moving Averages) order blindly. Our AI engine continuously predicts demand, preventing two devastating supply chain failures:
        
        1. **Stockouts (Lost Revenue):** The AI orders just in time, capturing sales that the baseline missed.
        2. **Overstock (Holding Cost):** The AI doesn't hold excess buffer, freeing up working capital.
        
        **This exact screen proves the direct ROI of the AI system to stakeholders by comparing dollars saved.**
        """)
    
    if sim_df is None:
        st.warning("Simulation data not found. Please run `prepare_simulation.py` to generate the 365-day backtest.")
        return
        
    sim_df['date'] = pd.to_datetime(sim_df['date'])
    total_ai_lost = sim_df['ai_lost_revenue'].sum()
    total_base_lost = sim_df['base_lost_revenue'].sum()
    
    total_ai_holding = sim_df['ai_holding_cost'].sum()
    total_base_holding = sim_df['base_holding_cost'].sum()
    
    saved_lost = total_base_lost - total_ai_lost
    total_savings = (total_base_lost + total_base_holding) - (total_ai_lost + total_ai_holding)

    st.markdown(f"<h3 style='text-align: center; margin-bottom:30px;'>Total Bottom-line Benefit: <span style='color:#2EA043'>+${total_savings:,.0f} / year</span></h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>AI DYNAMIC FORECASTING</div>
            <div style='margin-bottom: 5px;'>Lost Revenue (Stockouts): <br><span class='metric-value-ai'>${total_ai_lost:,.0f}</span></div>
            <div>Capital Tied in Inventory: <br><span class='metric-value-ai'>${total_ai_holding:,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>HISTORICAL BASELINE</div>
            <div style='margin-bottom: 5px;'>Lost Revenue (Stockouts): <br><span class='metric-value-base'>${total_base_lost:,.0f}</span></div>
            <div>Capital Tied in Inventory: <br><span class='metric-value-base'>${total_base_holding:,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Daily Items Affected by Stockout")
    fig_st = go.Figure()
    fig_st.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['base_stockout_items'], mode='lines', name='Baseline Strategy', line=dict(color='#F85149', width=1.5)))
    fig_st.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['ai_stockout_items'], mode='lines', name='AI Empowered', line=dict(color='#2EA043', width=2.5)))
    fig_st.update_layout(height=400, template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_st, use_container_width=True)
    
    st.subheader("Daily Inventory Overstock Liability ($)")
    fig_hc = go.Figure()
    fig_hc.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['base_holding_cost'], mode='lines', name='Baseline Strategy', line=dict(color='#F85149', width=1.5)))
    fig_hc.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['ai_holding_cost'], mode='lines', name='AI Empowered', line=dict(color='#2EA043', width=2.5)))
    fig_hc.update_layout(height=400, template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_hc, use_container_width=True)

    st.markdown("---")
    st.subheader("Methodology & Analytical Assumptions")
    st.markdown("""
    When evaluating the simulation results, please consider the following business realities and assumptions incorporated into the calculation:
    
    1. **Target Metric (Sales vs True Demand):** 
       The M5 dataset records *historical sales*. If an item was out of stock, sales were zero, which masks the true lost demand. During simulation, if stock falls to zero, we assume the true missed demand matches the AI's predicted demand for that day. 
    2. **Product Perishability (Shrinkage & Spoilage):**
       The current logic assumes non-perishable goods (e.g., `HOBBIES` or `HOUSEHOLD` items). Inventory only decreases when a sale happens. In reality, for `FOODS`, items expire and are written off. This means the model's actual holding cost might be slightly higher in a real supermarket due to spoilage, but since the AI minimizes holding time, it would simultaneously decrease food waste drastically.
    3. **Holding Cost Variations (Cold-Chain Logistics):**
       A flat holding cost rate (approx $0.05 per unit/day) is applied uniformly. However, storing frozen goods requires energy-intensive cold-chain logistics, while dry goods just require shelf space. The flat rate serves as an academic baseline. 
    4. **Universal Unit Price:**
       For visual clarity, a flat profit margin multiplier was applied based on the average retail item. 
    """)
