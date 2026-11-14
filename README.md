# COMP4145 PD2 Enhanced Golden-Cross Strategy Prototype
Extended from COMP4145 Lab2 baseline golden-cross back-test prototype, built with Python and Streamlit web UI.

## Core Strategy Rule (Unchanged from Lab2)
- Buy signal: 50-day moving average crosses above 200-day moving average (Golden Cross)
- Sell condition (whichever triggers first):
  1. 15% profit target achieved
  2. Maximum holding period of 60 calendar days

## New PD2 Implemented Features
1. Multi-stock portfolio batch back-test (max 10 tickers, rate-limit protection sleep delay)
2. Risk metrics calculation: max drawdown, annualized volatility, Sharpe ratio and automatic risk-level classification
3. HKBU GenAI (gpt-6-luna) integration for natural-language back-test report. API-key is input from web UI and never hard-coded.

## Project Structure
COMP4145_PD2/
├─ app.py               # Streamlit web frontend, UI layout and interaction
├─ trading_strategy.py  # Back-end core logic: data fetch, strategy simulation, risk calculation
├─ requirements.txt     # Dependency list for Streamlit Cloud deployment
└─ .gitignore           # Exclude cache, virtual environment and secret files

## Local Run
1. Create python virtual environment
2. Install dependencies:
pip install -r requirements.txt
3. Launch web application:
streamlit run app.py

## Deployment
This project can be directly deployed on Streamlit Cloud by linking GitHub repository.

## Disclaimer
This prototype is purely for COMP4145 course educational demonstration only. It does NOT constitute real-world investment advice.