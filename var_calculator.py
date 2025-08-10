import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from nsepython import equity_history

# ===== CONFIG =====
HOLDINGS_FILE = "holdings.xlsx"   # Your holdings file
HOLDINGS_SHEET = "holdings"       # Sheet name in Excel
MC_SIMULATIONS = 100000           # Monte Carlo simulations
HIST_DAYS = 365                   # Number of past days for history
# ==================

def load_holdings():
    """Load holdings and calculate weights."""
    df = pd.read_excel(HOLDINGS_FILE, sheet_name=HOLDINGS_SHEET)
    df.columns = [c.strip() for c in df.columns]

    # Calculate market value & weights
    df["MarketValue"] = df["Quantity"] * df["Price"]
    total_value = df["MarketValue"].sum()
    weights = df.set_index("Ticker")["MarketValue"] / total_value

    return df, weights, total_value

def get_nse_history(ticker, days=HIST_DAYS):
    """Fetch historical daily closing prices from NSE."""
    today = datetime.today()

    # Roll back to last Friday if weekend
    if today.weekday() >= 5:  
        days_to_subtract = today.weekday() - 4
        today = today - timedelta(days=days_to_subtract)

    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    df = equity_history(symbol=ticker, start=start_date, end=end_date)

    if not df.empty:
        df = df[["CH_TIMESTAMP", "CH_CLOSING_PRICE"]].copy()
        df["CH_TIMESTAMP"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.set_index("CH_TIMESTAMP", inplace=True)
        return df["CH_CLOSING_PRICE"]
    else:
        return pd.Series(dtype=float)

def get_price_history(tickers, days=HIST_DAYS):
    """Fetch historical prices for all tickers from NSE."""
    all_data = {}
    for ticker in tickers:
        prices = get_nse_history(ticker, days)
        if not prices.empty:
            all_data[ticker] = prices
    if not all_data:
        raise ValueError("❌ No valid price data fetched for any ticker from NSE. Please check tickers.")
    return pd.DataFrame(all_data).dropna(how="all")

def calculate_parametric_var(portfolio_returns, total_value, confidence_level=0.95):
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    z_score = norm.ppf(confidence_level)
    return -(mean_return - z_score * std_dev) * total_value

def calculate_historical_var(portfolio_returns, total_value, confidence_level=0.95):
    return -np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * total_value

def calculate_monte_carlo_var(portfolio_returns, total_value, confidence_level=0.95, num_simulations=MC_SIMULATIONS):
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    simulated_returns = np.random.normal(mean_return, std_dev, num_simulations)
    return -np.percentile(simulated_returns, 100 * (1 - confidence_level)) * total_value

def run_var_calculations(conf_levels=[0.95, 0.99]):
    """Run all VaR calculations and return results DataFrame."""
    holdings_df, weights, total_value = load_holdings()
    price_history = get_price_history(list(weights.index))
    returns = price_history.pct_change().dropna()
    portfolio_returns = returns @ weights

    results = []
    for cl in conf_levels:
        results.append([f"Parametric {int(cl*100)}%", calculate_parametric_var(portfolio_returns, total_value, cl)])
        results.append([f"Historical {int(cl*100)}%", calculate_historical_var(portfolio_returns, total_value, cl)])
        results.append([f"Monte Carlo {int(cl*100)}%", calculate_monte_carlo_var(portfolio_returns, total_value, cl)])

    return pd.DataFrame(results, columns=["Method", "VaR (₹)"])
