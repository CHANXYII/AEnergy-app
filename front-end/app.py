import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="AEnergy", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Glassmorphism & Custom Pink Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&display=swap');
    
    .stApp {
        background: linear-gradient(
            135deg,
            #FFF0F6 0%,
            #FFE4EC 25%,
            #FDE2FF 60%,
            #E8F0FF 100%
        );
        background-attachment: fixed;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 182, 193, 0.5);
    }

    .hero-card {
        background: linear-gradient(135deg, #FF69B4, #FF1493);
        padding: 50px;
        border-radius: 35px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(255, 20, 147, 0.45);
        margin-bottom: 30px;
    }


    /* Predicted Cost */
    .hero-card h1 {
        color: #FFFFFF !important; 
        font-size: 4.5rem !important;
        font-weight: 700 !important;
        margin: 15px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Hero Card */
    .hero-card p {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    .hero-card {
        background: linear-gradient(135deg, #FF69B4, #FF1493, #FF85C1);
        background-size: 200% 200%;
        animation: gradientMove 6s ease infinite;
        padding: 50px;
        border-radius: 35px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(255, 20, 147, 0.35);
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    h1, h2, h3, h4, h5, h6 {
        color: #D23669 !important;
        font-weight: 600 !important;
    }

    /* Input Labels */
    label {
        color: #D23669 !important;
        font-weight: 600 !important;
    }

    /* Radio Label */
    .stRadio > label {
        color: #D23669 !important;
        font-weight: 600 !important;
    }

    /* Label ของ input ทุกประเภท */
    label {
        color: #D23669 !important;
        font-weight: 500 !important;
    }

    /* slider text */
    div[data-baseweb="slider"] {
        color: #D23669 !important;
    }

    /* Number Input Label */
    .stNumberInput label {
        color: #D23669 !important;
    }

    /* Select Slider Label */
    .stSelectSlider label {
        color: #D23669 !important;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF69B4, #FF1493);
        color: #000000;
        border: none;
        border-radius: 20px;
        padding: 15px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 20, 147, 0.4);
        color: #FFFFFF;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 15px 15px 0 0;
        padding: 10px 25px;
        color: #D23669;
    }
    .stTabs [aria-selected="true"] {
        background: #FF69B4 !important;
        color: #FFE4ED !important;
    }
    </style>
    """, unsafe_allow_html=True)

# LOGIC FUNCTIONS
CSV_FILE = 'collected_data.csv'
MODEL_FILE = 'model.pkl'

def load_data():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['wattage', 'hours', 'days', 'efficiency', 'cost'])
        df.to_csv(CSV_FILE, index=False)
    return pd.read_csv(CSV_FILE)

def train_and_save_model(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 2: return None, "Insufficient data"
        X, y = df[['wattage', 'hours', 'days', 'efficiency']], df['cost']
        model = LinearRegression().fit(X, y)
        with open(MODEL_FILE, 'wb') as f: pickle.dump(model, f)
        return {"r2": r2_score(y, model.predict(X))}, None
    except Exception as e: return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/616/616494.png" width="90">
            <h2 style="color:#D23669; margin-bottom:5px;">AEnergy</h2>
            <p style="color:#FF69B4; font-size:0.9rem;">Smart Saving Assistant</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    menu = st.radio("Navigation", ["Dashboard", "AI Center"])
    st.markdown("---")


# --- MAIN CONTENT ---
df = load_data()

if "Dashboard" in menu:
    st.markdown("# วิเคราะห์และพยากรณ์ค่าไฟ")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ คุณยังไม่ได้เทรนโมเดล AI กรุณาไปที่ AI Center เพื่อบันทึกข้อมูลก่อนนะคะ")
    
    col1, col2 = st.columns([1, 1.4], gap="large")
    
    with col1:
        st.markdown("#### ตั้งค่าอุปกรณ์")
        p_w = st.number_input("กำลังไฟฟ้า (Watts)", 10, 10000, 1000, step=50)
        p_h = st.slider("การใช้งาน (ชั่วโมง/วัน)", 0.0, 24.0, 8.0)
        p_d = st.number_input("จำนวนวัน (ต่อเดือน)", 1, 31, 30)
        p_e = st.select_slider("ฉลากประหยัดไฟ", options=[1, 2, 3, 4, 5], value=5)

    with col2:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f: 
                model = pickle.load(f)
            
            # Prediction & Base Cost Calculation
            base_cost = (p_w / 1000) * p_h * p_d * 4.5
            
            # Prepare input safely
            input_data = np.array([[p_w, p_h, p_d, p_e]])
            pred = model.predict(input_data)[0]
            predicted_cost = max(0, pred)
            
            # HERO CARD
            st.markdown(f"""
                <div class="hero-card">
                    <p style="text-transform: uppercase; letter-spacing: 2px;">
                        Estimated Monthly Cost
                    </p>
                    <h1>฿ {predicted_cost:,.2f}</h1>
                    <p>คำนวณโดย AI ส่วนตัวของคุณ</p>
                </div>
            """, unsafe_allow_html=True)

            # Glass comparison
            difference = predicted_cost - base_cost
            percent = (difference / base_cost * 100) if base_cost != 0 else 0

            if difference > 0:
                st.warning(f"AI สูงกว่าสูตรพื้นฐาน {abs(percent):.1f}%")
            else:
                st.success(f"AI ต่ำกว่าสูตรพื้นฐาน {abs(percent):.1f}%")

            st.markdown('</div>', unsafe_allow_html=True)

            # Chart
            #chart_data = pd.DataFrame({
            #    "Source": ["Math Formula", "AI Prediction"],
            #    "Value": [base_cost, predicted_cost]
            #})
            #fig = px.bar(chart_data, x="Value", y="Source", orientation='h', color="Source",
            #             color_discrete_map={"Math Formula": "#B6FFCA", "AI Prediction": "#FF7091"})
            #fig.update_layout(height=200, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', 
            #                  plot_bgcolor='rgba(0,0,0,0)', font=dict(family="DM Sans", color="#D23669"))
            #st.plotly_chart(fig, use_container_width=True)
            #st.markdown('</div>', unsafe_allow_html=True)

# AI Center
else:
    st.markdown("# การจัดการข้อมูล AI")
    tab1, tab2 = st.tabs(["📝 บันทึกข้อมูลใหม่", "📉 ตรวจสอบสถานะ"])
    
    with tab1:
        with st.form("data_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1: w_in = st.number_input("Wattage", min_value=10)
            with c2: h_in = st.number_input("Hours/Day", min_value=0.5)
            with c3: d_in = st.number_input("Days/Month", min_value=1, max_value=31)
            
            c4, c5 = st.columns([1, 2])
            with c4: e_in = st.selectbox("Label", [1,2,3,4,5], index=4)
            with c5: cost_in = st.number_input("Actual Cost (ยอดจากบิลจริง)", min_value=0.0)
            
            if st.form_submit_button("บันทึกข้อมูล"):
                new_row = pd.DataFrame([[w_in, h_in, d_in, e_in, cost_in]], 
                                       columns=['wattage', 'hours', 'days', 'efficiency', 'cost'])
                new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
                st.toast("บันทึกสำเร็จ! ข้อมูลถูกจัดเก็บแล้ว")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        col_list, col_train = st.columns([2, 1])
        with col_list:
            st.markdown("##### ประวัติข้อมูล")
            st.dataframe(df, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_train:
            st.markdown("##### 🚀 AI Training")
            st.metric("Total Records", len(df))
            if st.button("🔄 Retrain AI"):
                metrics, err = train_and_save_model(CSV_FILE)
                if not err:
                    st.success(f"AI ฉลาดขึ้นแล้ว! (R²: {metrics['r2']:.2%})")
                    st.balloons()
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #000000; font-size: 0.8rem;'>© 2026 Energy Minty Pro | Designed for Clean Energy & Clean UI</div>", 
            unsafe_allow_html=True)
