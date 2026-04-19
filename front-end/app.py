import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import backend

# Page Configuration
st.set_page_config(
    page_title="AEnergy", 
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional White/Pink Light Mode Theme & Native Container Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(120deg, #FFE4E6 0%, #FAFAFA 25%, #E0E7FF 50%, #FFF0F5 75%, #FEF3C7 100%);
        background-size: 200% 200%;
        background-attachment: fixed;
        animation: auroraGradient 15s ease infinite;
        font-family: 'Inter', sans-serif;
        color: #334155;
    }

    @keyframes auroraGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1E293B !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    p, label, span {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(24px) saturate(160%);
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
    }

    /* Holographic Frosted Glass Containers (Glassmorphism) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(20px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(225, 29, 72, 0.08), 0 4px 12px 0 rgba(99, 102, 241, 0.05) !important;
        padding: 1rem !important;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 12px 40px 0 rgba(225, 29, 72, 0.15), 0 6px 16px 0 rgba(99, 102, 241, 0.1) !important;
        transform: translateY(-2px);
    }

    div[data-baseweb="slider"] {
        color: #E11D48 !important;
    }
    
    .stButton>button {
        background-color: #BE185D;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button * {
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #9D174D;
        box-shadow: 0 4px 6px -1px rgba(157, 23, 77, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(225, 29, 72, 0.05);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        font-weight: 600;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #BE185D !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #1E293B !important;
    }
    </style>
""", unsafe_allow_html=True)

# LOAD DATA & TRAIN IF NEEDED
df = backend.load_data()
if not backend.os.path.exists(backend.MODEL_FILE):
    backend.train_and_save_model()

# SIDEBAR NAV
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:20px 0;">
            <div style="font-size: 3rem; margin-bottom: -10px;">⚡️</div>
            <h2 style="color: #BE185D !important; margin-bottom: 0;">AEnergy</h2>
            <p style="color:#64748B !important; font-size:0.9rem;">Enterprise Energy Platform</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigation", ["Dashboard", "Insights", "Control Center"])

if menu == "Dashboard":
    st.markdown("<h1>Energy Cost Estimator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem; color: #64748B;'>Optimize your device power usage with Machine Learning algorithms.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='color: #1E293B !important; margin-bottom: 1rem;'>Device Configuration</h3>", unsafe_allow_html=True)
            p_w = st.number_input("Wattage (W)", 10, 10000, 1000, step=50)
            p_h = st.slider("Daily Usage (Hours)", 0.0, 24.0, 8.0)
            p_d = st.slider("Days Active per Month", 1, 31, 30)
            p_e = st.select_slider("Efficiency Label (5 is Best)", options=[1, 2, 3, 4, 5], value=5)

    with col2:
        predicted_cost = backend.predict_cost(p_w, p_h, p_d, p_e)
        base_kwh = backend.calculate_kwh(p_w, p_h, p_d)
        standard_cost = base_kwh * 4.5
        
        if predicted_cost is not None:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E11D48, #BE185D); padding: 32px; border-radius: 16px; text-align: center; box-shadow: 0 10px 15px -3px rgba(225, 29, 72, 0.25); color: white; margin-bottom: 24px;">
                    <div style="color: #FFFFFF !important; text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.95rem; margin-bottom: 8px; font-weight: 700;">Estimated Monthly Cost</div>
                    <div style="color: #FFFFFF !important; font-size: 3.5rem; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1); font-weight: 700; line-height: 1.2;">฿ {predicted_cost:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown("<p style='font-size: 0.9rem; margin-bottom: -10px; color:#64748B;'>This gauge compares your AI-predicted cost against the standard mathematical rate. Staying below the black line means you're operating efficiently!</p>", unsafe_allow_html=True)
                
                # Pink/Slate Gauge Chart with Threshold for clear reading
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = predicted_cost,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction vs Standard Rate", 'font': {'color': 'rgba(0,0,0,0)', 'size': 1}},
                    delta = {'reference': standard_cost, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
                    gauge = {
                        'axis': {'range': [None, max(standard_cost, predicted_cost) * 1.5], 'tickcolor': "#94A3B8"},
                        'bar': {'color': "#E11D48"},
                        'bgcolor': "#F1F5F9",
                        'threshold': {
                            'line': {'color': "#1E293B", 'width': 4},
                            'thickness': 0.75,
                            'value': standard_cost
                        }
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#334155"}, margin=dict(l=10, r=10, t=20, b=10), height=200)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("AI Model not found. Please train the model in the Control Center.")

elif menu == "Insights":
    st.markdown("<h1>Data Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1rem; color: #64748B;'>Overview of the relationship between energy consumption (kWh) and appliance efficiency.</p>", unsafe_allow_html=True)
    
    # Process for charting
    if 'kwh' not in df.columns:
        df['kwh'] = df.apply(lambda r: backend.calculate_kwh(r['wattage'], r['hours'], r['days']), axis=1)
    
    # Map for easy reading
    df_chart = df.copy()
    df_chart['Efficiency Label'] = 'Label ' + df_chart['efficiency'].astype(str)
    
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("### Cost vs Consumption (kWh)", help="This plot demonstrates that higher efficiency (dark pink) consistently results in a lower cost for the same amount of power used.")
            
            # Using Discrete Colors for better separation instead of hard-to-read continuous scales
            color_discrete_map = {
                'Label 5': '#831843', # Darkest Pink (Most Efficient)
                'Label 4': '#BE185D',
                'Label 3': '#E11D48',
                'Label 2': '#F43F5E',
                'Label 1': '#FB7185'  # Lightest Pink (Least Efficient)
            }
            
            fig1 = px.scatter(df_chart, x='kwh', y='cost', color='Efficiency Label',
                              color_discrete_map=color_discrete_map,
                              labels={'kwh': 'Total Consumption (kWh)', 'cost': 'Net Cost (THB)'})
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155"))
            st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        with st.container(border=True):
            st.markdown("### Average Cost by Efficiency Label", help="Clearly compares the average operating cost across different appliance efficiency ratings.")
            avg_cost = df_chart.groupby('Efficiency Label')['cost'].mean().reset_index()
            
            # Sort order
            avg_cost = avg_cost.sort_values(by='Efficiency Label')

            fig2 = px.bar(avg_cost, x='Efficiency Label', y='cost', text='cost', color='Efficiency Label',
                          color_discrete_map=color_discrete_map,
                          labels={'Efficiency Label': 'Efficiency Rating', 'cost': 'Average Cost (THB)'})
            fig2.update_traces(texttemplate='฿%{text:.2f}', textposition='outside')
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155"))
            st.plotly_chart(fig2, use_container_width=True)

elif menu == "Control Center":
    st.markdown("<h1>Platform Control Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1rem; color: #64748B;'>For administrators to seed real-world data and tune the machine learning algorithms.</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Collect Real Data", "Model Engine"])
    
    with tab1:
        with st.container(border=True):
            st.markdown("### Record Real Electricity Bill Data")
            with st.form("data_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                w_in = c1.number_input("Wattage", min_value=10)
                h_in = c2.number_input("Hours/Day", min_value=0.5)
                d_in = c3.number_input("Days/Month", min_value=1, max_value=31)
                
                c4, c5 = st.columns([1, 2])
                e_in = c4.selectbox("Efficiency Label", [1,2,3,4,5], index=4)
                cost_in = c5.number_input("Actual Bill Cost (THB)", min_value=0.0)
                
                if st.form_submit_button("Submit Record"):
                    backend.add_record(w_in, h_in, d_in, e_in, cost_in)
                    st.toast("Record saved and system updated!")
                    df = backend.load_data() 

    with tab2:
        with st.container(border=True):
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.markdown("### AI Engine Status")
                st.metric("Trained Records", len(df))
                st.write("Algorithm: **RandomForestRegressor** (Captures non-linear efficiency dynamics)")
                
            with c2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🔄 Retrain AI Engine", use_container_width=True):
                    metrics, err = backend.train_and_save_model()
                    if err:
                        st.error(f"Engine failure: {err}")
                    else:
                        st.success(f"Successfully retrained! Accuracy (R²): {metrics['r2']:.2%}")
                        st.balloons()

st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>© 2026 AEnergy Platform | Precision AI Engine</div>", unsafe_allow_html=True)
