import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf

# ===== CONFIG =====
HOLDINGS_FILE = "holdings.xlsx"   # Your holdings file
HOLDINGS_SHEET = "holdings"       # Sheet name in Excel
MC_SIMULATIONS = 100000           # Monte Carlo simulations
HIST_PERIOD = "1y"                # Historical data period from Yahoo
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

def get_price_history(tickers, period=HIST_PERIOD):
    """Download historical adjusted close prices."""
    price_history = yf.download(tickers, period=period, interval="1d")["Adj Close"]
    return price_history.dropna(how="all")

def calculate_parametric_var(portfolio_returns, total_value, confidence_level=0.95):
    """Parametric VaR using mean & std of portfolio returns."""
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    z_score = norm.ppf(confidence_level)
    return -(mean_return - z_score * std_dev) * total_value

def calculate_historical_var(portfolio_returns, total_value, confidence_level=0.95):
    """Historical VaR from portfolio return distribution."""
    return -np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * total_value

def calculate_monte_carlo_var(portfolio_returns, total_value, confidence_level=0.95, num_simulations=MC_SIMULATIONS):
    """Monte Carlo VaR using simulated returns."""
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
