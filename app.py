import streamlit as st
import matplotlib.pyplot as plt
import requests
import pandas as pd
from trading_strategy import (
    get_stock_data, implement_strategy, analyze_results,
    calculate_risk_metrics, portfolio_backtest
)

st.set_page_config(page_title="PD2 Enhanced Golden-Cross Strategy Prototype", layout="wide")

for session_key in ["stock_df", "pos_df", "portfolio_result"]:
    if session_key not in st.session_state:
        st.session_state[session_key] = None

menu_selection = st.sidebar.radio(
    "📋 Navigation Menu",
    [
        "📈 Single Stock — Price & Moving-Average Chart",
        "📊 Single Stock — Back-Test Statistics",
        "📜 Single Stock — Detailed Trade Records",
        "💼 Multi-Stock Portfolio Back-Test",
        "🔍 Risk Metrics & HKBU GenAI Analysis Report"
    ]
)

st.sidebar.markdown("---")
ticker_input = st.sidebar.text_input("Single stock ticker symbol", value="MSFT")
run_single_btn = st.sidebar.button("▶ Run Single-Stock Strategy Back-Test")

if run_single_btn:
    with st.spinner("Downloading price data and executing back-test ..."):
        try:
            ticker_clean = ticker_input.strip().upper()
            stock_df = get_stock_data(ticker_clean, period="5y")
            pos_df = implement_strategy(stock_df)
            st.session_state["stock_df"] = stock_df
            st.session_state["pos_df"] = pos_df
            st.success(f"✅ Back-test completed for {ticker_clean}")
        except Exception as error:
            st.error(f"❌ Runtime error: {str(error)}")
            st.session_state["stock_df"] = None
            st.session_state["pos_df"] = None


if menu_selection == "📈 Single Stock — Price & Moving-Average Chart":
    st.header("Stock Price Chart with Moving-Averages and Trade Signals")
    df = st.session_state["stock_df"]
    pos = st.session_state["pos_df"]
    if df is None:
        st.info("ℹ Please enter ticker symbol and click `Run Single-Stock Strategy Back-Test` from sidebar panel.")
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df.index, df["Close"], label="Closing Price", c="#0044bb", linewidth=1.2)
        ax.plot(df.index, df["MA_50"], label="MA-50 (Short-term)", c="#ff9900", linewidth=1.5)
        ax.plot(df.index, df["MA_200"], label="MA-200 (Long-term)", c="#008822", linewidth=1.5)
        if not pos.empty:
            ax.scatter(pos["BuyDate"], pos["BuyPrice"], marker="o", c="red", s=45, label="Buy Signal")
            ax.scatter(pos["SellDate"], pos["SellPrice"], marker="o", c="#992299", s=45, label="Sell Signal")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

elif menu_selection == "📊 Single Stock — Back-Test Statistics":
    st.header("Back-Test Quantitative Statistics Overview")
    pos = st.session_state["pos_df"]
    if pos is None:
        st.info("ℹ Please run single-stock strategy first in sidebar.")
    elif pos.empty:
        st.warning("⚠ No completed trade signals are generated within this back-test period.")
    else:
        stat_dict = analyze_results(pos)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", stat_dict["TotalTrades"])
        c2.metric("Winning Trades", stat_dict["WinningTrades"])
        c3.metric("Losing Trades", stat_dict["LosingTrades"])
        c4.metric("Win Rate (%)", stat_dict["WinRate"])
        c5.metric("Average Profit (%)", stat_dict["AvgProfitPct"])

elif menu_selection == "📜 Single Stock — Detailed Trade Records":
    st.header("Complete Trade History Table")
    pos = st.session_state["pos_df"]
    if pos is None:
        st.info("ℹ Please execute single-stock back-test first.")
    elif pos.empty:
        st.warning("⚠ No finished trade records available.")
    else:
        st.dataframe(pos, use_container_width=True)

elif menu_selection == "💼 Multi-Stock Portfolio Back-Test":
    st.header("Batch Multi-Stock Portfolio Back-Test (Maximum 10 tickers)")
    multi_ticker_text = st.text_area("Input tickers separated by comma", value="MSFT,AAPL,GOOG,AMZN")
    run_portfolio_btn = st.button("▶ Start Portfolio Back-Test")
    if run_portfolio_btn:
        ticker_list_input = [item.strip() for item in multi_ticker_text.split(",")]
        with st.spinner("Fetching multi-stock data, rate-limit sleep applied ..."):
            portfolio_result_df = portfolio_backtest(ticker_list_input)
            st.session_state["portfolio_result"] = portfolio_result_df
    port_df = st.session_state["portfolio_result"]
    if port_df is not None:
        st.dataframe(port_df, use_container_width=True)
        valid_result_df = port_df[port_df["error_msg"].isna()]
        if not valid_result_df.empty:
            st.subheader("📊 Portfolio-level Summary")
            st.write(f"Valid successfully tested ticker count: {len(valid_result_df)}")
            st.write(f"Average win rate across portfolio: {valid_result_df['WinRate'].mean():.2f} %")
            st.write(f"Average profit percentage across portfolio: {valid_result_df['AvgProfitPct'].mean():.2f} %")

elif menu_selection == "🔍 Risk Metrics & HKBU GenAI Analysis Report":
    st.header("Risk Metrics and AI-Generated Natural-Language Analysis Report")
    st.markdown("> Powered by HKBU GenAI Platform (model: gpt-6-luna). API-key will NOT be stored in application.")
    user_api_key = st.text_input("Enter your personal HKBU GenAI API-Key", type="password")

    df_single = st.session_state["stock_df"]
    pos_single = st.session_state["pos_df"]
    if df_single is None:
        st.info("ℹ Please run single-stock back-test from sidebar before generating AI analysis report.")
    else:
        daily_returns = df_single["Close"].pct_change().dropna()
        risk_output = calculate_risk_metrics(daily_returns)
        st.subheader("📋 Calculated Risk Indicators")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Max Drawdown", risk_output["max_drawdown"])
        rc2.metric("Annualized Volatility", risk_output["annual_volatility"])
        rc3.metric("Sharpe Ratio", risk_output["sharpe_ratio"])
        rc4.metric("Risk Level", risk_output["risk_level"])

        def call_hkbu_genai(api_key, prompt_content):
            endpoint_url = "https://genai.hkbu.edu.hk/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-6-luna",
                "messages": [{"role": "user", "content": prompt_content}],
                "temperature": 0.4
            }
            try:
                resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=35)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    return f"API request failed. HTTP status-code: {resp.status_code}, detail: {resp.text}"
            except Exception as err:
                return f"Network or connection error: {str(err)}"

        generate_ai_report_btn = st.button("🤖 Generate AI Investment Analysis Report")
        if generate_ai_report_btn:
            if not user_api_key.strip():
                st.error("⚠ Please fill your HKBU GenAI API-Key. Generate key from genai.hkbu.edu.hk website.")
            else:
                stat_info = analyze_results(pos_single)
                full_prompt_text = f"""
请用通俗易懂的中文为没有金融专业背景的普通学习者解读本次股票回测结果，
避免堆砌晦涩专业术语。务必加上风险提示：本回测仅课程教学原型，不构成任何真实投资建议。

回测统计数据：
总交易次数: {stat_info['TotalTrades']}
盈利交易数: {stat_info['WinningTrades']}
亏损交易数: {stat_info['LosingTrades']}
策略胜率: {stat_info['WinRate']}%
平均收益率: {stat_info['AvgProfitPct']}%
最大回撤: {risk_output['max_drawdown']}
年化波动率: {risk_output['annual_volatility']}
夏普比率: {risk_output['sharpe_ratio']}
风险等级分类: {risk_output['risk_level']}
"""
                with st.spinner("Calling HKBU GenAI service to generate report ..."):
                    ai_response_text = call_hkbu_genai(user_api_key.strip(), full_prompt_text)
                    st.subheader("🤖 AI-Generated Analysis Report")
                    st.write(ai_response_text)

st.sidebar.markdown("---")
st.sidebar.caption("PD2 Prototype | COMP4145 | Educational demo only, not real-investment advice.")