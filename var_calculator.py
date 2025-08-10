import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from nsepython import equity_history
import yfinance as yf

# ===== CONFIG =====
HOLDINGS_FILE = "holdings.xlsx"
HOLDINGS_SHEET = "holdings"
MC_SIMULATIONS = 100000
HIST_DAYS = 365
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
    if today.weekday() >= 5:  # weekend fix
        today -= timedelta(days=today.weekday() - 4)

    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    try:
        df = equity_history(symbol=ticker, start=start_date, end=end_date)
        if not df.empty:
            df = df[["CH_TIMESTAMP", "CH_CLOSING_PRICE"]].copy()
            df["CH_TIMESTAMP"] = pd.to_datetime(df["CH_TIMESTAMP"])
            df.set_index("CH_TIMESTAMP", inplace=True)
            return df["CH_CLOSING_PRICE"]
    except Exception:
        pass

    return pd.Series(dtype=float)

def get_yf_history(ticker, days=HIST_DAYS):
    """Fetch historical prices from Yahoo Finance."""
    try:
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = yf.download(ticker + ".NS", start=start, progress=False)
        if not df.empty:
            return df["Close"]
    except Exception:
        pass
    return pd.Series(dtype=float)

def normalize_ticker(ticker):
    """NSE-friendly format."""
    ticker = ticker.replace("&", "%26")  # M&M fix
    return ticker

def get_price_history(tickers, days=HIST_DAYS):
    """Fetch historical prices for all tickers, NSE first then Yahoo fallback."""
    all_data = {}
    failed_tickers = []

    for raw_ticker in tickers:
        ticker = normalize_ticker(raw_ticker)
        prices = get_nse_history(ticker, days)

        if prices.empty:
            prices = get_yf_history(raw_ticker, days)

        if not prices.empty:
            all_data[raw_ticker] = prices
        else:
            failed_tickers.append(raw_ticker)

    if failed_tickers:
        print(f"⚠️ No data for tickers: {', '.join(failed_tickers)}")

    if not all_data:
        raise ValueError("❌ No valid price data fetched for any ticker from NSE or Yahoo.")

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
    holdings_df, weights, total_value = load_holdings()
    price_history = get_price_history(list(weights.index))
    returns = price_history.pct_change().dropna()
    portfolio_returns = returns @ weights.loc[returns.columns]

    results = []
    for cl in conf_levels:
        results.append([f"Parametric {int(cl*100)}%", calculate_parametric_var(portfolio_returns, total_value, cl)])
        results.append([f"Historical {int(cl*100)}%", calculate_historical_var(portfolio_returns, total_value, cl)])
        results.append([f"Monte Carlo {int(cl*100)}%", calculate_monte_carlo_var(portfolio_returns, total_value, cl)])

    return pd.DataFrame(results, columns=["Method", "VaR (₹)"])
