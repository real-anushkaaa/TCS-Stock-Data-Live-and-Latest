import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set Streamlit page config
st.set_page_config(
    page_title="TCS Stock Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for red/pink theme
st.markdown("""
    <style>
    .main {
        background-color: #F6F7F8;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stMetric {
        background: #802547;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #FF3366;
    }
    .stSelectbox>div>div>div>div {
        
        color: #FF3366 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_TCS_stock_history.csv", parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime('%B')
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["Weekday"] = df["Date"].dt.day_name()
    df["52W_High"] = df["High"].rolling(window=252, min_periods=1).max()
    df["52W_Low"] = df["Low"].rolling(window=252, min_periods=1).min()
    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.title("TCS Stock Dashboard")
    day = st.selectbox("Day", options=["All"] + sorted(df["Day"].unique().tolist()))
    month = st.selectbox("Month", options=["All"] + sorted(df["MonthName"].unique(), key=lambda x: datetime.strptime(x, '%B').month))
    quarter = st.selectbox("Quarter", options=["All"] + sorted(df["Quarter"].unique()))
    year = st.selectbox("Year", options=["All"] + sorted(df["Year"].unique()))

# Filter data based on selections
df_filtered = df.copy()
if day != "All":
    df_filtered = df_filtered[df_filtered["Day"] == int(day)]
if month != "All":
    df_filtered = df_filtered[df_filtered["MonthName"] == month]
if quarter != "All":
    df_filtered = df_filtered[df_filtered["Quarter"] == int(quarter)]
if year != "All":
    df_filtered = df_filtered[df_filtered["Year"] == int(year)]

# Calculate 52W High/Low (rolling window)
df["52W_High"] = df["High"].rolling(window=252, min_periods=1).max()
df["52W_Low"] = df["Low"].rolling(window=252, min_periods=1).min()

# KPI Metrics
avg_52w_high = df_filtered["52W_High"].mean()
avg_52w_low = df_filtered["52W_Low"].mean()
total_volume = df_filtered["Volume"].sum()
total_open = df_filtered["Open"].sum()
total_close = df_filtered["Close"].sum()

# KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Average Of 52W High", f"{avg_52w_high/1000:.2f}K")
kpi2.metric("Average Of 52W Low", f"{avg_52w_low/1000:.2f}K")
kpi3.metric("Volume Total", f"{total_volume/1e6:.0f}M")
kpi4.metric("Open Total", f"{total_open/1000:.2f}K")
kpi5.metric("Close Total", f"{total_close/1000:.2f}K")

st.markdown("---")

# Quarterly Aggregates: 52W Highs & Lows
quarterly = df_filtered.groupby("Quarter").agg({"52W_High": "sum", "52W_Low": "sum"}).reset_index()
fig_quarter = go.Figure()
fig_quarter.add_trace(go.Bar(x=quarterly["Quarter"], y=quarterly["52W_High"], name="Sum of 52W H", marker_color="#52B8F3"))
fig_quarter.add_trace(go.Bar(x=quarterly["Quarter"], y=quarterly["52W_Low"], name="Sum of 52W L", marker_color="#ff3366"))
fig_quarter.update_layout(barmode='stack', title="Quarterly Aggregates: Summing 52W Highs & Lows", plot_bgcolor="#fff0f3")

# Daily Totals by Day
fig_daily = px.line(
    df_filtered,
    x="Date",              # ✅ string, not list
    y=["Close", "Open"],   # ✅ multiple Y allowed
    labels={"value": "Price", "Date": "Date"},
    title="Daily Prices by Day"
)

fig_daily.update_traces(line=dict(width=3))
fig_daily.update_layout(plot_bgcolor="#F6F7F8", legend_title_text='')

# Ensure month order is correct
df_filtered["MonthName"] = pd.Categorical(
    df_filtered["MonthName"],
    categories=[datetime(2000, m, 1).strftime('%B') for m in range(1, 13)],
    ordered=True
)

# 📊 Aggregate by month
monthly = df_filtered.groupby("MonthName").agg({
    "52W_High": "sum",
    "52W_Low": "sum",
    "Volume": "sum"
}).reset_index().sort_values("MonthName")

# 📈 Create Plotly figure
fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(x=monthly["MonthName"], y=monthly["52W_High"], name="Sum of 52W H", marker_color="#FF3366"))
fig_monthly.add_trace(go.Bar(x=monthly["MonthName"], y=monthly["52W_Low"], name="Sum of 52W L", marker_color="#1DA2F0"))
fig_monthly.add_trace(go.Bar(x=monthly["MonthName"], y=monthly["Volume"], name="Sum of VOLUME", marker_color="#8E6BAB"))

# ⚙️ Layout with log scale
fig_monthly.update_layout(
    barmode='group',
    title="Monthly Totals: Aggregating 52W Highs & Lows And Volume",
    plot_bgcolor="#F6F7F8",
    yaxis=dict(type='log', title='Log Scale of Values')
)

# Weekday Trend: 52W High, Low, Open, Close
weekday = df_filtered.groupby("Weekday").agg({"52W_High": "sum", "52W_Low": "sum", "Close": "sum", "Open": "sum"}).reset_index()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday["Weekday"] = pd.Categorical(weekday["Weekday"], categories=weekday_order, ordered=True)
weekday = weekday.sort_values("Weekday")
fig_weekday = go.Figure()
fig_weekday.add_trace(go.Scatter(x=weekday["Weekday"], y=weekday["52W_High"], mode="lines+markers", name="Sum of 52W H", line=dict(color="#52B8F3", width=3)))
fig_weekday.add_trace(go.Scatter(x=weekday["Weekday"], y=weekday["52W_Low"], mode="lines+markers", name="Sum of 52W L", line=dict(color="#2EC4B6", width=3)))
fig_weekday.add_trace(go.Scatter(x=weekday["Weekday"], y=weekday["Close"], mode="lines+markers", name="Sum of close", line=dict(color="#b71c1c", width=3)))
fig_weekday.add_trace(go.Scatter(x=weekday["Weekday"], y=weekday["Open"], mode="lines+markers", name="Sum of OPEN", line=dict(color="#f8bbd0", width=3)))
fig_weekday.update_layout(title="Sum of 52W H, 52W L, close and OPEN by Day", plot_bgcolor="#F6F7F8")

# Volume by Month (Horizontal Bar)
fig_vol_month = px.bar(monthly, x="Volume", y="MonthName", orientation="h", color_discrete_sequence=["#1DA2F0"], title="Sum of VOLUME by Month")
fig_vol_month.update_layout(plot_bgcolor="#F6F7F8")

# Layout
row1, row2 = st.columns([2, 3])
with row1:
    st.plotly_chart(fig_quarter, use_container_width=True)
with row2:
    st.plotly_chart(fig_daily, use_container_width=True)

row3, row4 = st.columns([2, 3])
with row3:
    st.plotly_chart(fig_monthly, use_container_width=True)
with row4:
    st.plotly_chart(fig_weekday, use_container_width=True)

st.plotly_chart(fig_vol_month, use_container_width=True)

st.markdown("<br><sub>TCS Dashboard Design | Design by Oikantik</sub>", unsafe_allow_html=True)