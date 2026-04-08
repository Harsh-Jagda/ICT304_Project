import streamlit as st
from src.ui.data_manager import load_model, load_data, build_sales_matrix
from src.ui.page_procurement import render_page as render_procurement
from src.ui.page_scorecard import render_page as render_scorecard

st.set_page_config(
    page_title="Supply Chain AI Hub",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background: rgba(240, 242, 246, 0.8); border: 1px solid #E0E4E8;
        border-radius: 10px; padding: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); color: #1F2937;
    }
    .metric-title { font-size: 16px; color: #4B5563; text-transform: uppercase; letter-spacing: 1px; font-weight: bold; margin-bottom: 10px;}
    .metric-value-ai { font-size: 28px; font-weight: 700; color: #2EA043; } /* Green */
    .metric-value-base { font-size: 28px; font-weight: 700; color: #F85149; } /* Red */
    .metric-value-neutral { font-size: 28px; font-weight: 700; color: #58A6FF; } /* Blue */
</style>
""", unsafe_allow_html=True)

def login_screen():
    st.markdown("<h2 style='text-align: center'>WMS Access Control</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("**Hint:** manager/123, director/123, admin/123")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if pwd == "123":
                if user in ["manager", "director", "admin"]:
                    st.session_state.role = user
                    st.rerun()
            st.error("Invalid credentials")

def main():
    if "role" not in st.session_state:
        st.session_state.role = None

    if st.session_state.role is None:
        login_screen()
        return

    model = load_model()
    df, sim_df, lt = load_data()
    
    st.sidebar.markdown(f"### WMS Dashboard v2.1")
    
    # UI Note about CA Data and Store Selection
    st.sidebar.info("📌 **Note**: To optimize memory usage, this dashboard currently only loads stores from California (CA).")
    
    store_options = sorted([s for s in df['store_id'].unique() if s.startswith('CA_')])
    selected_store = st.sidebar.selectbox("🏠 Select Store:", store_options)
    
    # If the user changes the store, reset the simulation state
    if "current_store" not in st.session_state:
        st.session_state.current_store = selected_store
    elif st.session_state.current_store != selected_store:
        st.session_state.current_store = selected_store
        # Delete state keys to force re-init in page_procurement
        for k in ['si_ready', 'si_date', 'si_inventory', 'si_deliveries', 'si_approved_today', 'si_missed_sales', 'si_history_sales']:
            if k in st.session_state:
                del st.session_state[k]

    store_df, sales_dict = build_sales_matrix(df, selected_store)

    try:
        from src.mlops.model_registry import get_production_model_entry
        prod_entry = get_production_model_entry()
        if prod_entry:
            st.sidebar.markdown(f"**Model Version:** `{prod_entry.get('version', 'Unknown')}`")
            st.sidebar.markdown(f"**Model Health (MAE):** `{prod_entry.get('mae', 'N/A')}`")
        else:
            st.sidebar.markdown("**Model Version:** `Local Fallback`")
    except Exception as e:
        # FIX #6: was 'except: pass' — silently hid corrupt registry errors
        st.sidebar.warning(f"⚠️ Registry unavailable: {e}")
    
    st.sidebar.markdown(f"**Role:** {st.session_state.role.capitalize()}")
    if st.sidebar.button("Logout"):
        st.session_state.role = None
        st.rerun()
    
    pages = []
    if st.session_state.role in ["manager", "admin"]:
        pages.append("🟢 Live Procurement Hub")
    if st.session_state.role in ["director", "admin"]:
        pages.append("📈 ROI Simulation Tool")
        
    page = st.sidebar.radio("Navigation:", pages)
    
    if page == "🟢 Live Procurement Hub":
        render_procurement(model, store_df, sales_dict, lt)
    elif page == "📈 ROI Simulation Tool":
        render_scorecard(sim_df)

if __name__ == "__main__":
    main()
