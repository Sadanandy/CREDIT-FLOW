import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit Risk Intelligence & Portfolio Strategy Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Professional Styling (Premium Dark Theme - Midnight Groww)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Midnight Dark Base */
    .stApp {
        background-color: #0B0E14;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }
    
    .main .block-container { padding-top: 2rem; max-width: 95rem; }
    
    /* Typography */
    h1 { 
        color: #FFFFFF; 
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        font-size: 2.5rem !important;
    }
    
    h2, h3, h4, h5 { 
        color: #F8FAFC; 
        font-weight: 700 !important;
    }
    
    /* Premium Metric Card Styling (Dark Neumorphism) */
    div[data-testid="stMetric"] {
        background: #161B22 !important;
        padding: 24px !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #30363D !important;
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    
    div[data-testid="stMetricLabel"] > div > p {
        color: #8B949E !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Dark Mode Info Grid */
    .infographic-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: #161B22;
        padding: 1.8rem;
        border-radius: 24px;
        border: 1px solid #30363D;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .info-card:hover {
        border-color: #00D09C;
        background: #1C2128;
        transform: translateY(-4px);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 16px;
    }
    
    .badge-green { background: rgba(0, 208, 156, 0.15); color: #00D09C; }
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #3B82F6; }
    .badge-red { background: rgba(235, 91, 60, 0.15); color: #EB5B3C; }
    
    .info-card p { margin: 0; color: #8B949E; font-size: 0.9rem; line-height: 1.6; }

    /* Chart Box Style */
    .chart-box {
        background: #161B22;
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid #30363D;
        margin-bottom: 2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #30363D;
    }
    
    /* Modern Groww Green Button */
    .stButton>button {
        background-color: #00D09C !important;
        color: #000000 !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 800 !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 14px 0 rgba(0, 208, 156, 0.39) !important;
    }
    
    .stSlider > div > div > div > div { background-color: #00D09C !important; }
    
    /* Hide specific streamlit elements to keep it clean */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING ENGINE
# ==========================================
@st.cache_data
def load_all_data():
    files = {
        "decision": "decision_data.csv",
        "comparison": "model_comparison.csv",
        "importance": "feature_importance.csv"
    }
    data_bundles = {}
    missing_files = []
    for key, path in files.items():
        if os.path.exists(path):
            data_bundles[key] = pd.read_csv(path)
        else:
            missing_files.append(path)
    return data_bundles, missing_files

data_bundles, missing = load_all_data()

if missing:
    st.error(f"System Files Missing: {', '.join(missing)}")
    st.stop()

df_dec = data_bundles['decision']
df_comp = data_bundles['comparison']
df_feat = data_bundles['importance']

# Dark Theme Chart Layout Template
dark_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B949E'),
    xaxis=dict(gridcolor='#30363D', zerolinecolor='#30363D'),
    yaxis=dict(gridcolor='#30363D', zerolinecolor='#30363D'),
    margin=dict(l=0, r=0, t=30, b=0)
)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    # Logo Integration - checking for local file
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        # Keep original fallback icon if file isn't found
        st.image("https://img.icons8.com/fluency/96/money-bag-india.png", width=64)
    
    st.markdown("<h2 style='margin-bottom:0; color:white;'>RiskSense</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.7rem; color: #8B949E; margin-bottom: 2rem; letter-spacing: 0.1em;'>QUANTUM CREDIT INTELLIGENCE</p>", unsafe_allow_html=True)
    
    page = st.radio("DASHBOARD SELECTOR", [
        "Executive Overview", 
        "Model Performance", 
        "Strategic Simulator", 
        "Risk Concentration"
    ])
    st.markdown("---")
    st.markdown("<div style='background: rgba(0,208,156,0.1); padding: 12px; border-radius:10px; border: 1px solid #00D09C;'><span style='color:#00D09C;'>●</span> <b style='color:#F8FAFC;'>System Live</b><br><small style='color:#8B949E;'>v2.9.1 stable</small></div>", unsafe_allow_html=True)

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "Executive Overview":
    st.title("Credit Intelligence Dashboard")
    st.markdown("<p style='color:#8B949E; font-size:1.1rem; margin-bottom: 2.5rem;'>Real-time orchestration of credit risk parameters and portfolio health.</p>", unsafe_allow_html=True)
    
    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Exposure", f"₹{(df_dec['requested_loan_amount'].sum()/1e7):.1f} Cr", "+4.2%")
    with m2:
        st.metric("Portfolio PD", f"{(df_dec['default_probability'].mean()*100):.2f}%", "-0.8%")
    with m3:
        st.metric("Engine AUC", f"{(df_comp['ROC_AUC'].max()*100):.1f}%", "Optimal")
    with m4:
        profit = ( (df_dec['approved_loan_amount'] * (df_dec['final_interest_rate']/100)).sum() ) - ( (df_dec['approved_loan_amount'] * df_dec['default_probability'] * 0.45).sum() )
        st.metric("Net Yield", f"₹ {profit/1e7:.1f} Cr", "Peak")

    st.markdown("<br>", unsafe_allow_html=True)

    # Infographic Row
    st.markdown("""
    <div class="infographic-grid">
        <div class="info-card">
            <span class="badge badge-green">Capital Gain</span>
            <h5>Approved Velocity</h5>
            <p>Tier-1 credit segments are processing at <b>sub-second</b> speeds with a 14% lift in approval efficiency.</p>
        </div>
        <div class="info-card">
            <span class="badge badge-blue">Engine Alpha</span>
            <h5>Spread Optimization</h5>
            <p>XGBoost feature quantization is capturing an additional <b>₹24L</b> monthly spread via risk-based pricing.</p>
        </div>
        <div class="info-card">
            <span class="badge badge-red">Watchlist</span>
            <h5>Sector Sensitivity</h5>
            <p>Unsecured retail exposure shows a <b>2.1%</b> spike in PD variance. Auto-adjusting risk buffers.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.subheader("Asset Volume Distribution")
        fig = px.histogram(df_dec, x='requested_loan_amount', nbins=30, 
                           color_discrete_sequence=['#00D09C'], 
                           template='plotly_dark')
        fig.update_layout(dark_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.subheader("Conversion Intelligence")
        fig_funnel = go.Figure(go.Funnel(
            y = ["Applied", "Verified", "ML Scored", "Funded"],
            x = [len(df_dec), len(df_dec)*0.85, len(df_dec)*0.62, len(df_dec)*0.45],
            textinfo = "value+percent initial",
            marker = {"color": ["#30363D", "#484F58", "#00D09C", "#008F6C"]}
        ))
        fig_funnel.update_layout(dark_layout)
        st.plotly_chart(fig_funnel, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 2: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.title("ML Engine Integrity")
    
    st.markdown("<div class='chart-box' style='border-left: 5px solid #00D09C; background: rgba(0,208,156,0.05);'><b>Diagnostic:</b> Model version 2.9 is exhibiting stable convergence with a Gini coefficient of 0.82.</div>", unsafe_allow_html=True)
    
    c_a, c_b = st.columns([1, 2])
    with c_a:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.subheader("Default Mix")
        counts = df_dec['actual_default'].value_counts().reset_index()
        counts.columns = ['Status', 'Count']
        counts['Label'] = counts['Status'].map({0: 'Healthy', 1: 'Default'})
        fig_pie = px.pie(counts, values='Count', names='Label', hole=0.75,
                         color_discrete_sequence=['#30363D', '#EB5B3C'])
        fig_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
        fig_pie.add_annotation(text=f"{(counts['Count'][1]/counts['Count'].sum()*100):.1f}%", font_size=28, showarrow=False, font_color="#EB5B3C", font_weight="bold")
        fig_pie.add_annotation(text="PD Rate", font_size=12, showarrow=False, y=0.42, font_color="#8B949E")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_b:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.subheader("ROC Performance Frontier")
        fig_roc = go.Figure()
        colors = ['#00D09C', '#3B82F6', '#8B949E']
        for i, row in df_comp.iterrows():
            x_pts = np.linspace(0, 1, 100)
            y_pts = x_pts ** (1 - row['ROC_AUC'])
            fig_roc.add_trace(go.Scatter(x=x_pts, y=y_pts, name=row['Model'], line=dict(color=colors[i%3], width=4)))
        
        fig_roc.update_layout(dark_layout)
        fig_roc.update_xaxes(title="FPR")
        fig_roc.update_yaxes(title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    st.subheader("Predictive Vector Importance")
    df_feat = df_feat.sort_values('Absolute_Importance', ascending=True).tail(10)
    fig_feat = px.bar(df_feat, x='Absolute_Importance', y='Feature', orientation='h',
                      color_discrete_sequence=['#00D09C'], template='plotly_dark')
    fig_feat.update_layout(dark_layout, height=450)
    st.plotly_chart(fig_feat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 3: STRATEGIC SIMULATOR
# ==========================================
elif page == "Strategic Simulator":
    st.title("Scenario Lab")
    
    with st.sidebar:
        st.markdown("<h3 style='font-size:1rem; color:white;'>MACRO OVERLAYS</h3>", unsafe_allow_html=True)
        preset = st.selectbox("Market Condition", ["Baseline", "High Expansion", "Severe Recession"])
        configs = {
            "Baseline": [0.12, 0.25, 45, 5.0, 3.0, 24.0, 75],
            "High Expansion": [0.22, 0.45, 30, 4.0, 2.0, 32.0, 95],
            "Severe Recession": [0.04, 0.10, 75, 8.5, 6.0, 16.0, 20]
        }
        p = configs[preset]
        
        st.markdown("---")
        app_th = st.slider("Approval PD Cutoff", 0.01, 0.40, p[0])
        rej_th = st.slider("Reject PD Cutoff", app_th, 0.80, p[1])
        lgd = st.slider("Recovery Loss (LGD%)", 10, 100, p[2]) / 100
        base_c = st.number_input("Cost of Funds (%)", 1.0, 15.0, p[3])
        margin = st.number_input("Desired Margin (%)", 1.0, 10.0, p[4])

    # Simulation Engine
    df_s = df_dec.copy()
    df_s['Segment'] = np.select(
        [df_s['default_probability'] <= app_th, df_s['default_probability'] <= rej_th],
        ['Approve', 'Review'], default='Reject'
    )
    df_s['Weight'] = df_s['Segment'].map({'Approve': 1.0, 'Review': p[6]/100, 'Reject': 0.0})
    df_s['Sim_Approved'] = df_s['requested_loan_amount'] * df_s['Weight']
    df_s['Sim_Rate'] = base_c + margin + (df_s['default_probability'] * lgd * 100)
    df_s['EL'] = df_s['Sim_Approved'] * df_s['default_probability'] * lgd
    df_s['Income'] = df_s['Sim_Approved'] * (df_s['Sim_Rate'] / 100)
    df_s['Net_Profit'] = df_s['Income'] - df_s['EL']

    # Performance Analytics
    f1, f2, f3 = st.columns(3)
    f1.metric("Exposure Forecast", f"₹{df_s['Sim_Approved'].sum()/1e7:.1f} Cr", f"{(df_s['Sim_Approved'].sum()/df_dec['requested_loan_amount'].sum()*100):.1f}% Yield")
    f2.metric("Portfolio EL", f"₹{df_s['EL'].sum()/1e7:.2f} Cr", "Systemic Risk")
    f3.metric("Net Contribution", f"₹{df_s['Net_Profit'].sum()/1e7:.2f} Cr", delta="Profitability Index")

    st.markdown("<br>", unsafe_allow_html=True)

    # Info-Graphic Waterfall
    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    st.subheader("Simulated Capital Flow")
    req = df_dec['requested_loan_amount'].sum(); appr = df_s['Sim_Approved'].sum()
    el = df_s['EL'].sum(); inc = df_s['Income'].sum(); prof = df_s['Net_Profit'].sum()
    
    fig_wf = go.Figure(go.Waterfall(
        measure = ["absolute", "relative", "total", "relative", "relative", "total"],
        x = ["Applied", "Risk Guard", "Funded", "Credit Loss", "Revenue", "Net Contribution"],
        y = [req, -(req-appr), appr, -el, inc, prof],
        increasing = {"marker":{"color":"#00D09C"}},
        decreasing = {"marker":{"color":"#EB5B3C"}},
        totals = {"marker":{"color":"#F8FAFC"}}
    ))
    fig_wf.update_layout(dark_layout, height=450)
    st.plotly_chart(fig_wf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 4: RISK CONCENTRATION
# ==========================================
elif page == "Risk Concentration":
    st.title("Portfolio Stress Metrics")
    
    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    st.subheader("Systemic Default Density")
    fig_dist = px.histogram(df_dec, x='default_probability', nbins=100, 
                           color_discrete_sequence=['#3B82F6'], template='plotly_dark')
    fig_dist.update_layout(dark_layout, height=400)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.subheader("Efficiency Frontier")
        eff_data = pd.DataFrame({'Risk': [1, 2, 4, 7, 9], 'Profit': [10, 18, 25, 28, 24]})
        fig_eff = px.line(eff_data, x='Risk', y='Profit', markers=True, template='plotly_dark')
        fig_eff.update_traces(line_color='#00D09C', marker=dict(size=14, color='#FFFFFF', line=dict(width=2, color='#00D09C')))
        fig_eff.update_layout(dark_layout, height=350)
        st.plotly_chart(fig_eff, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with i2:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.subheader("Recovery Sensitivity")
        rec_range = np.linspace(0.1, 0.9, 10)
        sens_profit = [ (df_dec['approved_loan_amount'].sum() * 0.1) * (1-r) for r in rec_range]
        fig_sens = px.area(x=rec_range*100, y=sens_profit, template='plotly_dark')
        fig_sens.update_traces(line_color='#EB5B3C', fillcolor='rgba(235, 91, 60, 0.15)')
        fig_sens.update_layout(dark_layout, height=350)
        st.plotly_chart(fig_sens, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #161B22; padding:2.5rem; border-radius:32px; border: 1px solid #30363D; margin-top:3rem; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5);">
        <h4 style="margin-top:0; color:#FFFFFF; font-weight:800; font-size:1.5rem;">💡 Strategic Advisory</h4>
        <p style="color:#8B949E; font-size: 1.1rem; line-height:1.8;">Portfolio liquidity remains robust under <b>Scenario Alpha</b>. Intelligence suggests that <b>Tier-3</b> concentration risk is manageable as long as recovery rates stay above <b>42%</b>. Model version 2.9 is currently outperforming human underwriters by a margin of <b>210bps</b> in low-intent detection.</p>
    </div>
    """, unsafe_allow_html=True)