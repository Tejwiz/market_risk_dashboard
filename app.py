import streamlit as st
import pandas as pd
import numpy as np
from nsepython import equity_history
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.parse

# -------------------------------
# Ticker Mapping Helper
# -------------------------------
def map_to_nse_symbol(ticker):
    """
    Map a plain-text ticker from holdings file to NSE-compatible symbol.
    """
    ticker = ticker.strip().upper()
    # Replace & with URL-encoded %26
    ticker = ticker.replace("&", "%26")
    # Remove spaces
    ticker = ticker.replace(" ", "")
    return ticker

# -------------------------------
# VaR Calculation Functions
# -------------------------------
def calculate_historical_var(portfolio_returns, confidence_level):
    var_percentile = np.percentile(portfolio_returns, 100 - confidence_level)
    return -var_percentile

def calculate_parametric_var(portfolio_returns, confidence_level):
    mean = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    z_score = norm.ppf(confidence_level / 100)
    return -(mean - z_score * std_dev)

def calculate_monte_carlo_var(mean_return, std_return, confidence_level, num_simulations=100000):
    simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
    return -np.percentile(simulated_returns, 100 - confidence_level)

# -------------------------------
# NSE Price Fetch Function
# -------------------------------
def fetch_nse_prices(ticker, start_date, end_date):
    try:
        start_str = pd.to_datetime(start_date).strftime("%d-%m-%Y")
        end_str = pd.to_datetime(end_date).strftime("%d-%m-%Y")
        
        df = equity_history(ticker, "EQ", start_str, end_str)
        if df is None or df.empty:
            return None
        
        df['CH_TIMESTAMP'] = pd.to_datetime(df['CH_TIMESTAMP'])
        df.set_index('CH_TIMESTAMP', inplace=True)
        df.sort_index(inplace=True)
        return df['CH_CLOSING_PRICE']
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Market Risk Dashboard - VaR", layout="wide")
st.title("📊 Market Risk Dashboard")
st.subheader("Value at Risk (VaR) - 90%, 95%, 99% Confidence Levels")

uploaded_file = st.file_uploader("Upload your holdings Excel file", type=["xlsx"])

if uploaded_file is not None:
    df_holdings = pd.read_excel(uploaded_file)

    if not all(col in df_holdings.columns for col in ["Ticker", "Quantity"]):
        st.error("❌ The file must have columns: 'Ticker', 'Quantity'")
    else:
        # Clean and map tickers
        df_holdings["Ticker_Clean"] = df_holdings["Ticker"].astype(str).str.strip().str.upper()
        df_holdings["Ticker_Mapped"] = df_holdings["Ticker_Clean"].apply(map_to_nse_symbol)

        st.write("📌 Mapped Tickers for NSE API:", df_holdings[["Ticker", "Ticker_Mapped"]])

        # Date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)

        # Fetch prices from NSE
        prices = pd.DataFrame()
        failed_tickers = []
        for _, row in df_holdings.iterrows():
            t_clean = row["Ticker_Clean"]
            t_mapped = row["Ticker_Mapped"]
            series = fetch_nse_prices(t_mapped, start_date, end_date)
            if series is not None and not series.empty:
                prices[t_clean] = series  # store under original name
            else:
                failed_tickers.append(t_clean)

        if prices.empty:
            st.error("❌ No valid price data fetched for any ticker from NSE. Please check tickers.")
            st.stop()

        if failed_tickers:
            st.warning(f"⚠️ No data found for: {', '.join(failed_tickers)}")

        # Drop failed tickers from holdings
        df_holdings = df_holdings[df_holdings["Ticker_Clean"].isin(prices.columns)]

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Portfolio weights
        quantities = df_holdings.set_index("Ticker_Clean")["Quantity"]

        latest_prices = prices.iloc[-1]
        position_values = latest_prices * quantities
        portfolio_value = position_values.sum()
        weights = position_values / portfolio_value

        portfolio_returns = (returns * weights).sum(axis=1)

        # Mean & Std Dev
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # Calculate VaR
        confidence_levels = [90, 95, 99]
        results = []
        for cl in confidence_levels:
            hist_var = calculate_historical_var(portfolio_returns, cl) * portfolio_value
            para_var = calculate_parametric_var(portfolio_returns, cl) * portfolio_value
            mc_var = calculate_monte_carlo_var(mean_return, std_return, cl) * portfolio_value
            results.append({
                "Confidence Level": f"{cl}%",
                "Historical VaR": round(hist_var, 2),
                "Parametric VaR": round(para_var, 2),
                "Monte Carlo VaR": round(mc_var, 2)
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        st.info(f"💰 Portfolio Value: ₹{portfolio_value:,.2f}")

        # -------------------------------
        # Volatility Shock Stress Testing
        # -------------------------------
        st.subheader("⚡ Volatility Shock Stress Testing")
        vol_shock = st.slider(
            "Select volatility shock (%)",
            min_value=0, max_value=200, value=30, step=5
        )

        shocked_std_dev = std_return * (1 + vol_shock / 100)
        stress_test_results = []
        for cl in confidence_levels:
            stressed_mc_var = calculate_monte_carlo_var(
                mean_return, shocked_std_dev, cl
            ) * portfolio_value
            stress_test_results.append({
                "Confidence Level": f"{cl}%",
                f"MC VaR (+{vol_shock}% Vol)": round(stressed_mc_var, 2)
            })

        stress_df = pd.DataFrame(stress_test_results)
        st.dataframe(stress_df)

else:
    st.warning("📂 Please upload your holdings file first.")
