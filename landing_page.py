# Project_Raymond_Streamlit_App.py
# Streamlit single-file portfolio "story deck" for Project Raymond
# - Timeline story sections
# - Interactive visuals (simulated demo charts, backtest metrics)
# - Subscription CTA and contact/demo forms

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime, timedelta
import base64

st.set_page_config(page_title="Project Raymond — Gold Trading Model", layout="wide")

# ------------------------- Helper utilities -------------------------

def generate_equity_curve(days=365, seed=42, annual_return=3.7):
    np.random.seed(seed)
    # simulate daily returns with drift
    mu = (annual_return ** (1/252) - 1)
    sigma = 0.01
    returns = np.random.normal(loc=mu, scale=sigma, size=days)
    equity = 10000 * np.cumprod(1 + returns)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    return pd.DataFrame({"date": dates, "equity": equity, "returns": returns})


def compute_drawdowns(equity_series):
    hwm = equity_series.cummax()
    drawdown = (equity_series - hwm) / hwm
    return drawdown


def make_kpi_cards(kpis):
    cols = st.columns(len(kpis))
    for c, (title, (value, delta)) in zip(cols, kpis.items()):
        c.metric(label=title, value=value, delta=delta)


def equity_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['equity'], mode='lines', name='Equity'))
    fig.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=350)
    return fig


def drawdown_plot(df):
    dd = compute_drawdowns(df['equity'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=dd, fill='tozeroy', name='Drawdown'))
    fig.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=250, yaxis_tickformat='.0%')
    return fig


def simulated_cot_heatmap(days=52):
    # simulate weekly COT shifts for 3 groups: Commercials, Non-commercials, Retail
    np.random.seed(1)
    weeks = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='W')
    data = np.cumsum(np.random.randn(days, 3), axis=0)
    df = pd.DataFrame(data, columns=['Commercials','Non-commercials','Retail'], index=weeks)
    return df


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ------------------------- Page layout -------------------------

# Sidebar navigation
st.sidebar.title("Project Raymond")
page = st.sidebar.radio("Navigate:", ['Overview', 'Origins', 'COT Research', 'Model Architecture', 'Backtest Results', 'Demo & Visuals', 'Subscription & Contact'])

# Top header (common)
with st.container():
    left, mid, right = st.columns([1,3,1])
    with left:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/brand/streamlit-mark-color.png", width=80)
    with mid:
        st.markdown("# Project Raymond — Advanced Gold Trading Model")
        st.write("Enterprise-grade gold-focused trading model with cascade decision architecture, COT positioning signals, and risk-first execution logic.")
    with right:
        st.markdown("**Quick KPIs**")
        kpis = {
            'Best Win Rate (backtest)': ('90%', '+10% MoM'),
            'Best Annualized (sim)': ('300%', '+2% MoM'),
            'Suggested Start': ('$250/mo', '')
        }
        make_kpi_cards(kpis)

# -------- Overview --------
if page == 'Overview':
    st.header("Overview & Value Proposition")
    st.markdown(
        """
        **Project Raymond** is a gold-focused, cascade-architecture trading system that combines CFTC Commitment-of-Traders (COT) positioning data, price action features, and regime detection to generate high-confidence swing trade recommendations.

        The product is offered as a subscription licensing service for investment funds, trading firms, and professional traders looking for a production-ready forecasting and signal-provider tool. Deliverables include dashboards, model packs, trade blotters, and API-ready inference containers.
        """
    )
    st.divider()
    st.subheader("What's included in the subscription")
    st.markdown("- Weekly signal pack and trade blotter (entry, stop, target)\n- Access to the live Streamlit monitoring dashboard\n- Monthly performance report and risk review\n- Optional API deployment and integration support")

# -------- Origins / Story --------
if page == 'Origins':
    st.header("Origins: Journey to a High-Performance Gold Model")
    st.markdown("### How it started")
    st.write("Started trading via social signals and supply/demand key levels, learning the limits of ad-hoc approaches and the need for repeatable data-driven decisioning.")
    st.markdown("### Turning point")
    st.write("Discovery of COT (Commitment of Traders) data and institutional thinking inspired a data-centered research program. Built COTrends to track weekly shifts in trader group positioning.")

    st.markdown("---")
    st.subheader("From COTrends to Capital Flow Thinking")
    st.markdown(
        """
        Building COTrends was a major step forward. For the first time, positioning data from the CFTC made market sentiment measurable. Tracking how commercials, non-commercials, and retail traders shifted their exposure week over week brought structure to what had previously felt abstract.

        But while COTrends revealed *who* was positioned where, it raised a deeper question:

        **Why was capital moving in the first place?**

        That question marked the transition from pure positioning analysis to a broader capital-flow framework.

        The methodology evolved around a simple but powerful premise: markets do not move randomly — they move as capital is deployed, reallocated, and withdrawn across the financial system.

        At the center of that system sit investment banks.
        """
    )

    st.markdown("---")
    st.subheader("Capital Flow Methodology")
    # Insert the user's memo verbatim as requested
    st.markdown(
        """
        Our trading methodology is based on capital flow analysis and asset risk assessment, which begins from understanding the way investment banks make money off of loans and investments into sectors. The only way to understand when investment banks make money with and not from businesses is by following the capital flow from investors to investment banks to businesses and markets.

        - Investors deploy capital when it's structurally supportive (Interest rates)
        - Investment banks allocate capital to risk assesed sectors
        - Businesses grow and markets gain momentum

        By developing algorithms and models that align with this lense, we find our strategy development to be much more unique and rich in expected value.

        Make sense?
        """
    )

    st.markdown("---")
    st.subheader("Capital Flow Diagram")
    flow_cf = """
    digraph G {
        rankdir=LR
        node [shape=box, style=rounded]
        Investors [label="Investors
(Deploy capital when conditions support)"]
        Banks [label="Investment Banks
(Allocate capital to risk-assessed sectors)"]
        Businesses [label="Businesses
(Receive capital, grow operations)"]
        Markets [label="Markets
(Price & momentum formation)"]

        Investors -> Banks -> Businesses -> Markets
    }
    """
    st.graphviz_chart(flow_cf)

    st.markdown("---")
    st.subheader("Timeline")
    timeline = [
        ("Social Signals", "Initial experimentation; high noise and poor risk control"),
        ("Supply & Demand", "Improved entries around key levels but inconsistent outcomes"),
        ("COT Research", "6 months of focused COT research and COTrends tool development"),
        ("Health Gauge", "Integrated COT + price action signal to detect regimes"),
        ("Capital Flow Lens", "Shifted research focus to follow institutional capital allocation and sector flows"),
        ("Project Raymond", "Cascade model, enterprise-ready deployment design")
    ]
    for stage, desc in timeline:
        st.markdown(f"**{stage}** — {desc}")

# -------- COT Research --------
if page == 'COT Research':
    st.header("COT Research & Positioning Insights")
    st.write("Interactive exploration of weekly positioning dynamics by trader group.")

    cot_df = simulated_cot_heatmap(days=52)
    fig = px.imshow(cot_df.T, labels=dict(x="Week", y="Trader Group", color="Cumulative Shift"), x=[d.strftime('%Y-%m-%d') for d in cot_df.index], y=cot_df.columns)
    fig.update_layout(height=350, margin=dict(l=30,r=30,t=40,b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Health Gauge**: combines directional momentum and positioning divergence to classify market regimes. Toggle sensitivity below to see a rough regime overlay.")
    sensitivity = st.slider("Health Gauge sensitivity", 1, 10, 5)
    # show a dummy regime timeline based on sensitivity
    weeks = cot_df.index
    regime_score = cot_df['Commercials'] - cot_df['Non-commercials']
    regime = (regime_score > (regime_score.std() * (sensitivity/5))).astype(int)
    regime_df = pd.DataFrame({'week': weeks, 'regime': regime})
    st.line_chart(regime_df.set_index('week')['regime'])

# -------- Model Architecture --------
if page == 'Model Architecture':
    st.header("Model Architecture & Decision Flow")
    st.write("Cascade architecture with layered gating: top-level regime/directional filter → position sizing module → execution layer with slippage & TCA assumptions.")

    st.subheader("Cascade Flow Diagram")
    flow = """
    digraph G {
        rankdir=LR
        node [shape=box, style=rounded]
        COT [label="COT & Positioning"]
        Price [label="Price Action Features"]
        Macro [label="Macro Signals"]
        Health [label="Health Gauge\n(Regime Detector)"]
        L1 [label="L1: High-Confidence Filter"]
        L2 [label="L2: Directional Bias"]
        L3 [label="L3: Execution + Sizing"]
        API [label="Inference API / Container"]

        COT -> Health
        Price -> Health
        Macro -> Health
        Health -> L1 -> L2 -> L3 -> API
    }
    """
    st.graphviz_chart(flow)

    st.markdown("**Risk controls**: fixed fractional risk per trade, portfolio exposure caps, time-based exits, and enforced drawdown stop-loss for strategy-level deactivation.")

# -------- Backtest Results --------
if page == 'Backtest Results':
    st.header("Backtest Results & Performance Pack")
    st.write("Selected backtest snapshots and visual evidence of strategy robustness.")

    df_eq = generate_equity_curve(days=800, seed=7, annual_return=4.0)
    col1, col2 = st.columns((2,1))
    with col1:
        st.subheader('Equity Curve (Simulated)')
        st.plotly_chart(equity_plot(df_eq), use_container_width=True)
        st.subheader('Drawdowns')
        st.plotly_chart(drawdown_plot(df_eq), use_container_width=True)
    with col2:
        st.subheader('Summary Metrics')
        total_return = df_eq['equity'].iloc[-1]/df_eq['equity'].iloc[0]-1
        metrics = {
            'Total Return': f"{total_return:.1%}",
            'Max Drawdown': f"{compute_drawdowns(df_eq['equity']).min():.1%}",
            'Win Rate (sim)': '90%',
            'Annualized (sim)': '300%'
        }
        for k, v in metrics.items():
            st.markdown(f"**{k}**: {v}")

    st.markdown('---')
    st.subheader('Sample Trade Blotter (simulated)')
    trades = []
    start = datetime.today() - timedelta(days=90)
    for i in range(12):
        trades.append({
            'entry_date': (start + timedelta(days=i*7)).strftime('%Y-%m-%d'),
            'side': np.random.choice(['Long','Short']),
            'entry': round(1900 + np.random.randn()*10,2),
            'exit': round(1900 + np.random.randn()*12,2),
            'pnl': round(np.random.randn()*150,2)
        })
    trades_df = pd.DataFrame(trades)
    st.dataframe(trades_df)

    csv_bytes = df_to_csv_bytes(trades_df)
    st.download_button('Download sample blotter (CSV)', data=csv_bytes, file_name='project_raymond_sample_blotter.csv')

# -------- Demo & Visuals --------
if page == 'Demo & Visuals':
    st.header("Live Demo - Signal Simulation")
    st.write("Simulate model confidence and get an illustrative trade suggestion.")
    conf = st.slider('Model confidence (0-100)', 0, 100, 72)
    if conf > 70:
        st.success('Signal: **High-confidence Long** on Gold — recommended structure: Entry / Stop / Target shown below')
        st.markdown('**Entry:** 1935  **Stop:** 1920  **Target:** 1965')
    elif conf > 40:
        st.info('Signal: **Directional Bias — Monitor**')
    else:
        st.warning('Signal: **No trade — low conviction**')

    st.markdown('---')
    st.subheader('Visual Artifacts')
    c1, c2, c3 = st.columns(3)
    c1.image('https://placehold.co/400x250?text=Trade+Screenshot', caption='Trade screenshot (sample)')
    c2.image('https://placehold.co/400x250?text=Equity+Snapshot', caption='Equity snapshot (sample)')
    c3.image('https://placehold.co/400x250?text=Dashboard', caption='Dashboard screenshot (sample)')

# -------- Subscription & Contact --------
if page == 'Subscription & Contact':
    st.header("Licensing & Book a Consultation")
    st.write("Licensing available for investment funds, prop desks, and professional traders. Starter plan from $250 / month. Tailored enterprise integrations and white-label options are available.")

    st.subheader('Plans & Deliverables')
    st.markdown(
    """
    - **Starter ($250/mo)**: Weekly signal pack, access to Streamlit dashboard, monthly performance PDF.
    - **Pro**: Starter + API access, containerized inference, 24/7 alerts.
    - **Enterprise**: Pro + dedicated integration support, licensing, white-labeling options.
    """
)


    st.markdown('---')
    st.subheader('Why license Project Raymond')
    st.markdown(
        """
        - Institutional-grade signals built from capital flow analysis and position-level risk assessment.

        - Enterprise-ready delivery: dashboard, trade blotters, documented schema, and containerized inference for in-house integration.

        - Proven backtested robustness (methodology notes and backtest packs available on request).
        """
    )

    st.markdown('---')
    st.subheader('Profile & Contact')
    profile_col1, profile_col2 = st.columns([1,2])
    with profile_col1:
        st.image('https://placehold.co/150x150?text=Profile', width=120)
    with profile_col2:
        st.markdown('**Tsegaab Gebremedhin**')
        st.markdown('Quantitative trader & model developer — focused on Gold & macro-driven FX. Google Data Analytics certified. Contact: segaab120@gmail.com')
        st.markdown('[LinkedIn](https://linkedin.com/in/tsegaab-gebremedhin)')

    st.markdown('---')
    st.subheader('Request a demo, consultation or trial')
    st.write('Complete the form below to request a timed demo, receive a sanitized backtest pack, or schedule a consultation. Qualified leads will be offered a complimentary 30-minute strategy call.')

    with st.form('contact_form'):
        name = st.text_input('Full name')
        email = st.text_input('Email')
        company = st.text_input('Company / Fund')
        role = st.text_input('Role / Title')
        plan = st.selectbox('Interested plan', ['Starter', 'Pro', 'Enterprise'])
        funds_under_management = st.selectbox('Estimated AUM range', ['< $1M', '$1M–$10M', '$10M–$100M', '$100M+'])
        urgent = st.checkbox('Require urgent integration / SLA')
        comments = st.text_area('Message / requirements')
        submitted = st.form_submit_button('Request Demo / Book Consultation')
        if submitted:
            st.success('Thanks — the request has been noted. A follow-up will be sent to the provided email with demo scheduling options and next steps. Qualified leads will receive a calendar invite for a 30-minute strategy call.')
            # Here: integrate with email/Sales CRM (send request)

    st.markdown('---')
    st.subheader('Integration & Deployment Notes')
    st.write('App is designed for containerized deployment with an inference API (FastAPI) and can be integrated into an in-house execution stack. Payment onboarding typically uses Stripe/Checkout for automated billing. For enterprise licensing, SLAs and integration timelines are provided as part of the contract.')

    st.markdown('**Need a sanitized code repo, Dockerfile, or a timed demo?** Contact via email to request access and scheduling.')

# Footer
st.markdown('---')
left, right = st.columns([3,1])
with left:
    st.write('© Project Raymond — Quantitative Research & Signal Licensing')
with right:
    st.write('Contact: segaab120@gmail.com')


# ------------------------- End of app -------------------------
