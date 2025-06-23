import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import skew, kurtosis

# ——— Kalman Filter ———
def kalman_filter(observed, x=None, delta=1e-4, R=0.01, initial_beta=0, initial_P=1.0):
    if x is None:
        x = np.ones_like(observed)

    n = len(observed)
    beta = np.zeros(n)
    P = np.zeros(n)
    e = np.zeros(n)

    beta[0] = initial_beta
    P[0] = initial_P

    for t in range(1, n):
        beta[t] = beta[t-1]
        P[t] = P[t-1] + delta

        y = observed[t] - beta[t] * x[t]
        K = P[t] * x[t] / (x[t]**2 * P[t] + R)
        beta[t] = beta[t] + K * y
        P[t] = (1 - K * x[t]) * P[t]
        e[t] = y

    return beta, P, e

# ——— Dashboard plotting ———
def plot_dashboard(weekly_df, price_paths, revenue_paths=None, kalman_data=None, revenue_metrics=None,
                   freq="W-FRI", price_unit="USD/kg", revenue_unit="USD", kalman_delta=0.0001):
    last_date = pd.to_datetime(weekly_df["Date"].iloc[-1])
    periods = price_paths.shape[0] - 1
    dates = pd.date_range(last_date + timedelta(weeks=1), periods=periods, freq=freq)

    # Extract Kalman filter results
    beta_hat = kalman_data['beta']
    P_t = kalman_data['Pt']
    e_t = kalman_data['e_t']
    
    # Clean errors (skip first element)
    e_t_clean = e_t[1:]
    
    # Simplified Kalman metrics (without VaR/CVaR)
    kalman_metrics = {
        "Current Hedge Ratio": beta_hat[-1],
        "Current Variance (P_t)": P_t[-1],
        "Error Mean": np.mean(e_t_clean),
        "Error Std": np.std(e_t_clean),
    }

    final_prices = price_paths.iloc[-1].values
    price_mean = np.mean(final_prices)
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    ret_end = final_prices / weekly_df["Price"].iloc[-1] - 1
    VaR = -np.percentile(ret_end, 5) * weekly_df["Price"].iloc[-1]
    CVaR = -np.mean(ret_end[ret_end <= np.percentile(ret_end, 5)]) * weekly_df["Price"].iloc[-1]
    expected_vol = np.std(np.log(price_paths.values[1:] / price_paths.values[:-1]), axis=0).mean() * np.sqrt(52)
    skew_p = skew(final_prices)
    kurt_p = kurtosis(final_prices, fisher=False)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])

    # ——— Kalman Projection ———
    n_projection = len(dates)
    if n_projection > 0:
        last_beta = beta_hat[-1]
        last_P = P_t[-1]
        
        # Project with increasing uncertainty
        projected_beta = np.full(n_projection, last_beta)
        projected_Pt = last_P + kalman_delta * np.arange(1, n_projection + 1)
    else:
        projected_beta = np.array([])
        projected_Pt = np.array([])

    # Combine historical and projected
    full_dates = np.concatenate([weekly_df["Date"], dates])
    full_beta = np.concatenate([beta_hat, projected_beta])
    full_Pt = np.concatenate([P_t, projected_Pt])

    # Price paths
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(weekly_df["Date"], weekly_df["Price"], label="Historical", color="#1f77b4")
    ax1.plot(dates, price_paths.mean(axis=1)[1:], '--', label="Mean Forecast", color="orange")
    ax1.fill_between(dates, price_paths.quantile(0.05, axis=1)[1:], 
                     price_paths.quantile(0.95, axis=1)[1:], alpha=0.2)
    ax1.legend()
    ax1.set_title("Gold Price Forecast")
    ax1.set_ylabel(f"Price ({price_unit})")

    # Hedge ratio evolution with projection
    ax2 = fig.add_subplot(gs[0, 1])
    # Historical
    ax2.plot(weekly_df["Date"], beta_hat, label="Historical Hedge Ratio", color="purple")
    ax2.fill_between(weekly_df["Date"], 
                     beta_hat - 2*np.sqrt(P_t), 
                     beta_hat + 2*np.sqrt(P_t), 
                     alpha=0.2, color="purple")
    # Projection
    if n_projection > 0:
        ax2.plot(dates, projected_beta, ':', label="Projected Hedge Ratio", color="red")
        ax2.fill_between(dates, 
                         projected_beta - 2*np.sqrt(projected_Pt), 
                         projected_beta + 2*np.sqrt(projected_Pt), 
                         alpha=0.1, color="red")
    
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.set_title("Hedge Ratio Evolution with Projection")
    ax2.set_ylabel("Beta")

    # Final price distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(final_prices, bins=30, alpha=0.7, color="#1f77b4", edgecolor="white")
    ax3.axvline(np.mean(final_prices), color='black', linestyle='--', label="Mean")
    ax3.axvline(p5, color='red', linestyle='--', label="5th pct")
    ax3.axvline(p95, color='green', linestyle='--', label="95th pct")
    ax3.legend()
    ax3.set_title("Final Price Distribution")

    # Revenue forecast
    ax4 = fig.add_subplot(gs[1, 1])
    if revenue_paths is not None:
        ax4.plot(dates, revenue_paths.mean(axis=1)[1:], '--', color="green", label="Mean Revenue")
        ax4.fill_between(dates, np.percentile(revenue_paths, 5, axis=1)[1:], 
                         np.percentile(revenue_paths, 95, axis=1)[1:], alpha=0.2, color="green")
        ax4.legend()
        ax4.set_title("Revenue Forecast")
        ax4.set_ylabel(f"Revenue ({revenue_unit})")
    else:
        # Kalman filter errors
        ax4.hist(e_t_clean, bins=30, alpha=0.7, color="green", edgecolor="white")
        ax4.axvline(np.mean(e_t_clean), color='black', linestyle='--', label="Mean")
        ax4.axvline(np.percentile(e_t_clean, 5), color='red', linestyle='--', label="5th pct")
        ax4.axvline(np.percentile(e_t_clean, 95), color='blue', linestyle='--', label="95th pct")
        ax4.legend()
        ax4.set_title("Kalman Filter Error Distribution")

    # Price stats table
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    data5 = [
        ["Current Price", f"{weekly_df['Price'].iloc[-1]:,.4f}"],
        ["Forecast Mean", f"{price_mean:,.4f}"],
        ["90% CI", f"{p5:,.4f} - {p95:,.4f}"],
        ["Exp. Volatility", f"{expected_vol*100:.2f}% p.a."],
        ["VaR (5%)", f"{VaR:,.4f}"],
        ["CVaR (5%)", f"{CVaR:,.4f}"],
        ["Skewness", f"{skew_p:.2f}"],
        ["Kurtosis", f"{kurt_p:.2f}"],
    ]
    tbl5 = ax5.table(cellText=data5, cellLoc='left', loc='center')
    tbl5.auto_set_font_size(False)
    tbl5.set_fontsize(10)
    tbl5.scale(1, 1.5)

    # Kalman/revenue stats table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    if revenue_metrics:
        # Combine Kalman and revenue metrics
        combined_metrics = {**kalman_metrics, **revenue_metrics}
        data6 = [[k, f"{v:,.6f}"] for k, v in combined_metrics.items()]
    else:
        data6 = [[k, f"{v:,.6f}"] for k, v in kalman_metrics.items()]
    
    tbl6 = ax6.table(cellText=data6, cellLoc='left', loc='center')
    tbl6.auto_set_font_size(False)
    tbl6.set_fontsize(10)
    tbl6.scale(1, 1.5)

    plt.tight_layout()
    return fig

# ——— Streamlit app ———
st.set_page_config(layout="wide")
st.title("Gold Price & Hedge Ratio Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("merged.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

df = load_data()

# Sidebar
st.sidebar.header("Filters & Settings")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date])
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
if df.empty:
    st.error("No data in selected range.")
    st.stop()

n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, 100)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 4, 52, 12)
ci_var = st.sidebar.slider("VaR Confidence (%)", 80, 99, 95)
ci_cvar = st.sidebar.slider("CVaR Confidence (%)", 80, 99, 97)
if ci_cvar <= ci_var:
    st.sidebar.error("CVaR confidence must be greater than VaR confidence.")
    st.stop()

# Kalman parameters
st.sidebar.subheader("Kalman Filter Parameters")
kalman_R = st.sidebar.slider("Measurement Noise (R)", 0.001, 0.1, 0.01, 0.001)
kalman_delta = st.sidebar.slider("Process Noise (δ)", 0.0001, 0.01, 0.0001, 0.0001)

# Revenue parameters
st.sidebar.subheader("Revenue Simulation")
volume = st.sidebar.number_input("Volume (units)", min_value=1, value=1000)
show_revenue = st.sidebar.checkbox("Show Revenue Forecast", value=True)

# Prepare returns - FIXED: Compute log returns AFTER date filtering
df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))
rets = df["LogReturn"].dropna().values

# Kalman filter with adjustable parameters - FIXED: Align with original DataFrame
beta_hat, P_t, e_t = kalman_filter(rets, R=kalman_R, delta=kalman_delta)

# FIXED: Properly align Kalman results with DataFrame
# We have 1 less observation for Kalman results than original df
kalman_df = df.iloc[1:].copy()
kalman_df["HedgeRatio"] = beta_hat
kalman_df["Pt"] = P_t
kalman_df["e_t"] = e_t

# Monte Carlo
last_price = kalman_df["Price"].iloc[-1]  # Use last price from Kalman-aligned DF
paths = np.zeros((horizon + 1, n_sims))
paths[0] = last_price
mu, sigma = rets.mean(), rets.std()
for t in range(1, horizon + 1):
    eps = np.random.normal(mu, sigma, n_sims)
    paths[t] = paths[t - 1] * np.exp(eps)
price_paths = pd.DataFrame(paths)

# Revenue simulation
revenue_paths = None
revenue_metrics = None

if show_revenue:
    # Dynamic volume based on hedge ratio
    dynamic_beta = beta_hat[-1] if len(beta_hat) else 0
    dynamic_volume = volume * (1 - np.clip(dynamic_beta, 0, 1))
    revenue_paths = price_paths.values * dynamic_volume

    # Risk metrics
    alpha_var = 1 - ci_var / 100
    alpha_cvar = 1 - ci_cvar / 100
    VaR_rev = np.percentile(revenue_paths[-1], alpha_var * 100)
    threshold_cvar = np.percentile(revenue_paths[-1], alpha_cvar * 100)
    CVaR_rev = revenue_paths[-1][revenue_paths[-1] <= threshold_cvar].mean() if np.any(revenue_paths[-1] <= threshold_cvar) else np.nan

    revenue_metrics = {
        f"VaR_{ci_var}%": VaR_rev,
        f"CVaR_{ci_cvar}%": CVaR_rev,
        "ExpectedRevenue": revenue_paths[-1].mean(),
        "HedgeRatioApplied": dynamic_beta,
        "EffectiveVolume": dynamic_volume
    }

# Plot using Kalman-aligned dataframe
fig = plot_dashboard(
    weekly_df=kalman_df,  # Use Kalman-aligned DF for plotting
    price_paths=price_paths,
    revenue_paths=revenue_paths,
    kalman_data={"beta": beta_hat, "Pt": P_t, "e_t": e_t},
    revenue_metrics=revenue_metrics,
    kalman_delta=kalman_delta  # Pass delta for projection
)
st.pyplot(fig)

# Display raw data - FIXED: Use Kalman-aligned dataframe
st.subheader("Filtered Data Preview")
col1, col2 = st.columns(2)
with col1:
    st.write("**Price & Hedge Ratio Data**")
    st.dataframe(kalman_df[["Date", "Price", "HedgeRatio", "Pt", "e_t"]].tail(10))

# Kalman metrics summary
st.subheader("Hedge Ratio Metrics")
kalman_metrics = {
    "Current Hedge Ratio": beta_hat[-1],
    "Hedge Ratio Stability": P_t[-1],
    "Error Mean": np.mean(e_t[1:]),
    "Error Std Dev": np.std(e_t[1:]),
}

metrics_df = pd.DataFrame(list(kalman_metrics.items()), columns=["Metric", "Value"])
st.dataframe(metrics_df.style.format({"Value": "{:.6f}"}))

# Download buttons - FIXED: Use Kalman-aligned dataframe
st.subheader("Data Export")
csv1 = kalman_df[["Date", "Price", "HedgeRatio", "Pt", "e_t"]].to_csv(index=False)  # Kalman results
csv2 = price_paths.transpose().to_csv()  # Simulation paths

col1, col2 = st.columns(2)
col1.download_button(
    label="Download Kalman Data",
    data=csv1,
    file_name="kalman_results.csv",
    mime="text/csv"
)
col2.download_button(
    label="Download Price Simulations",
    data=csv2,
    file_name="price_simulations.csv",
    mime="text/csv"
)