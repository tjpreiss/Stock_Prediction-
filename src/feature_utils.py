import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
import os
import sys


def extract_features():
    """
    Downloads recent AMZN price data along with FX and macro index features.
    Used by the Streamlit app to provide a live feature baseline.
    Returns a DataFrame of log-differenced features (no target column).
    """
    return_period = 5

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    stk_tickers = ['AMZN']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    # Target: future log return (not used directly in notebook, but kept for consistency)
    Y = np.log(stk_data.loc[:, ('Adj Close', 'AMZN')]).diff(return_period).shift(-return_period)
    Y.name = 'AMZN_Future'

    # Feature: AMZN log return
    X1 = np.log(stk_data.loc[:, ('Adj Close', 'AMZN')]).diff(return_period).to_frame()
    X1.columns = ['AMZN']

    # FX log returns
    X2 = np.log(ccy_data).diff(return_period)

    # Macro index log returns
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    dataset.index.name = 'Date'

    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    # Drop the target column — return features only
    features = features.iloc[:, 1:]
    return features


def extract_features_pair():
    """
    Downloads recent price data for AMZN and its closest correlated pair stock (GOOG).
    Used for pair-trading feature construction in the Streamlit app.
    Returns a DataFrame with columns [AMZN, GOOG].
    """
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    # GOOG is historically one of the most cointegrated stocks with AMZN
    stk_tickers = ['AMZN', 'GOOG']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    Y = stk_data.loc[:, ('Adj Close', 'AMZN')]
    Y.name = 'AMZN'

    X = stk_data.loc[:, ('Adj Close', 'GOOG')]
    X.name = 'GOOG'

    dataset = pd.concat([Y, X], axis=1).dropna()
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    return features


def get_bitcoin_historical_prices(days=60):
    """
    Fetches Bitcoin daily price history from CoinGecko.
    Imported by the notebook for completeness but not called in the AMZN workflow.
    Returns a DataFrame with index=Date and column='Close Price (USD)'.
    """
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
        df = df[['Date', 'Close Price (USD)']].set_index('Date')
    except Exception:
        # Return empty DataFrame if API is unavailable
        df = pd.DataFrame(columns=['Close Price (USD)'])
        df.index.name = 'Date'

    return df


def get_year(col):
    return pd.to_numeric(col.iloc[:, 0].str[-4:], errors='coerce').to_frame()


def get_emp_num(col):
    s = col.iloc[:, 0].str.replace('10+ years', '10', regex=False).str.replace('< 1 year', '0', regex=False)
    return pd.to_numeric(s.str.split().str[0], errors='coerce').to_frame()


def get_term_num(col):
    return pd.to_numeric(col.iloc[:, 0].str.replace(' months', '', regex=False), errors='coerce').to_frame()
