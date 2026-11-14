


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time


def get_stock_data(ticker, period="5y"):
    """
    Fetch historical stock price data via yfinance API
    :param ticker: stock ticker symbol string
    :param period: data look-back period
    :return: DataFrame with Open, High, Low, Close, Volume columns
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data.empty:
        raise ValueError(f"Failed to retrieve price data for ticker {ticker}. Invalid symbol or API rate-limit triggered.")
    return data


def calculate_moving_averages(data):
    """Calculate 50-day and 200-day simple moving average"""
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["MA_200"] = data["Close"].rolling(window=200).mean()
    return data


def implement_strategy(data):
    """
    Golden-Cross trading strategy inherited from COMP4145 Lab2
    Buy: MA50 crosses above MA200
    Sell trigger (whichever comes first):
        1. Price rises to 15% profit target
        2. Max holding period reaches 60 calendar days
    :param data: stock price DataFrame
    :return: DataFrame containing each completed trade record
    """
    positions = []
    in_position = False
    buy_date = None
    buy_price = None

    data = calculate_moving_averages(data)
    data = data.dropna(subset=["MA_50", "MA_200"])
    if data.empty:
        return pd.DataFrame(positions)

    for idx, row in data.iterrows():
        ma50 = row["MA_50"]
        ma200 = row["MA_200"]
        close_price = row["Close"]

        if not in_position:
            prev_idx = idx - timedelta(days=1)
            if prev_idx in data.index:
                prev_row = data.loc[prev_idx]
                if ma50 > ma200 and prev_row["MA_50"] <= prev_row["MA_200"]:
                    in_position = True
                    buy_date = idx
                    buy_price = close_price
        else:
            hold_days = (idx - buy_date).days
            sell_take_profit = close_price >= 1.15 * buy_price
            sell_time_limit = hold_days >= 60

            if sell_take_profit or sell_time_limit:
                sell_reason = "Target profit 15% reached" if sell_take_profit else "Maximum holding period (60 days)"
                profit_pct = ((close_price - buy_price) / buy_price) * 100
                positions.append({
                    "BuyDate": buy_date,
                    "BuyPrice": round(buy_price, 2),
                    "SellDate": idx,
                    "SellPrice": round(close_price, 2),
                    "HoldingDays": hold_days,
                    "ProfitPct": round(profit_pct, 2),
                    "SellReason": sell_reason
                })
                in_position = False
                buy_date = None
                buy_price = None
    return pd.DataFrame(positions)


def analyze_results(pos_df, return_dict=True):
    """
    Calculate trade-level back-test statistics
    :param pos_df: trade records DataFrame
    :param return_dict: return dict if True, else return original dataframe
    :return: statistics dictionary
    """
    if pos_df.empty:
        res = {
            "TotalTrades": 0,
            "WinningTrades": 0,
            "LosingTrades": 0,
            "WinRate": 0.0,
            "AvgProfitPct": 0.0
        }
        return res if return_dict else pos_df

    total_trades = len(pos_df)
    win_trades = len(pos_df[pos_df["ProfitPct"] > 0])
    lose_trades = total_trades - win_trades
    win_rate = round((win_trades / total_trades) * 100, 2)
    avg_profit = round(pos_df["ProfitPct"].mean(), 2)

    res = {
        "TotalTrades": total_trades,
        "WinningTrades": win_trades,
        "LosingTrades": lose_trades,
        "WinRate": win_rate,
        "AvgProfitPct": avg_profit
    }
    return res if return_dict else pos_df


def calculate_risk_metrics(returns_series, risk_free_rate=0.02):
    """
    Compute financial risk indicators: max drawdown, annualized volatility, Sharpe ratio and risk level
    :param returns_series: daily percentage return pandas Series
    :param risk_free_rate: annual risk-free rate for Sharpe ratio calculation, default 2%
    :return: dictionary of risk metrics
    """
    if returns_series is None or returns_series.empty:
        return {
            "max_drawdown": None,
            "annual_volatility": None,
            "sharpe_ratio": None,
            "risk_level": "N/A"
        }

    cumulative_return = (1 + returns_series).cumprod()
    running_peak = cumulative_return.cummax()
    drawdown_series = (cumulative_return - running_peak) / running_peak
    max_drawdown = drawdown_series.min()

    annual_vol = returns_series.std() * np.sqrt(252)
    if annual_vol == 0:
        sharpe = 0
    else:
        sharpe = (returns_series.mean() * 252 - risk_free_rate) / annual_vol

    if sharpe > 1 and max_drawdown > -0.15:
        risk_level = "Low Risk"
    elif sharpe > 0.5 and max_drawdown > -0.30:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "max_drawdown": round(max_drawdown, 4),
        "annual_volatility": round(annual_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "risk_level": risk_level
    }


def portfolio_backtest(ticker_list, max_limit=10, sleep_sec=0.7):
    """
    Batch multi-stock portfolio back-test
    :param ticker_list: list of stock ticker strings
    :param max_limit: maximum number of stocks to prevent excessive API request
    :param sleep_sec: delay seconds between each stock request for rate-limit protection
    :return: combined result DataFrame for all tickers
    """
    result_collection = []
    ticker_list = list(set([t.strip().upper() for t in ticker_list if t.strip()]))[:max_limit]

    for ticker in ticker_list:
        try:
            stock_data = get_stock_data(ticker, period="5y")
            trade_df = implement_strategy(stock_data)
            stat_result = analyze_results(trade_df)
            daily_ret = stock_data["Close"].pct_change().dropna()
            risk_result = calculate_risk_metrics(daily_ret)
            row_item = {"Ticker": ticker, **stat_result, **risk_result}
            result_collection.append(row_item)
            time.sleep(sleep_sec)
        except Exception as err:
            result_collection.append({
                "Ticker": ticker,
                "TotalTrades": None,
                "WinningTrades": None,
                "LosingTrades": None,
                "WinRate": None,
                "AvgProfitPct": None,
                "max_drawdown": None,
                "annual_volatility": None,
                "sharpe_ratio": None,
                "risk_level": "Error",
                "error_msg": str(err)
            })
    return pd.DataFrame(result_collection)


if __name__ == "__main__":
    sample_data = get_stock_data("MSFT")
    sample_trades = implement_strategy(sample_data)
    print(analyze_results(sample_trades))