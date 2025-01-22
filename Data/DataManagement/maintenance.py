import pandas as pd
import yfinance as yf
from datetime import datetime
import os
from typing import List
import time
from functools import lru_cache
import numpy as np
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / '.git').exists():
            return parent
        if parent.name == 'Financial-Trading-Algorithm-progress-updated':
            return parent
    raise FileNotFoundError("Could not find project root directory")

# Set up project paths
PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / 'Data'
TICKERS_PATH = DATA_PATH / 'Info' / 'Tickers'
LOGS_PATH = DATA_PATH / 'Info' / 'Logs'
STOCK_DATA_PATH = DATA_PATH / 'Stock-Data'

def filter_tickers():
    filtered_path = TICKERS_PATH / "tickers-filtered.csv"
    
    try:
        if (filtered_path.stat().st_size > 0):
            filtered_df = pd.read_csv(filtered_path)
            return filtered_df['Ticker'].tolist()
        else:
            tickers_df = pd.read_csv(TICKERS_PATH / "tickers.csv", usecols=['Name'])
            filtered_df = pd.DataFrame({'Ticker': tickers_df['Name'].values})
            filtered_df.to_csv(filtered_path, index=False)
            return filtered_df['Ticker'].tolist()
    except FileNotFoundError:
        tickers_df = pd.read_csv(TICKERS_PATH / "tickers.csv", usecols=['Name'])
        filtered_df = pd.DataFrame({'Ticker': tickers_df['Name'].values})
        filtered_df.to_csv(filtered_path, index=False)
        return filtered_df['Ticker'].tolist()

def create_log():
    log_file = LOGS_PATH / "stock-price-last-updated.txt"
    log_file.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def read_log():
    log_file = LOGS_PATH / "stock-price-last-updated.txt"
    try:
        last_update = datetime.strptime(log_file.read_text().strip(), "%Y-%m-%d %H:%M:%S")
        return last_update
    except (FileNotFoundError, ValueError):
        return None

@lru_cache(maxsize=1000)
def get_cached_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)

def download_batch(tickers: List[str]) -> List[str]:
    results = []
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    data = yf.download(tickers, period="max", auto_adjust=True, group_by='ticker', session=session)
    
    for ticker in tickers:
        try:
            if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
                ticker_data = data[ticker].copy()
            else:
                ticker_data = data[ticker]
            
            if not ticker_data.empty:
                necessary_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                ticker_data = ticker_data[necessary_columns]
                ticker_data = ticker_data.dropna()
                ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
                ticker_data = ticker_data.sort_index()
                
                for col in ticker_data.select_dtypes(include=[np.float64]).columns:
                    ticker_data[col] = ticker_data[col].astype(np.float32)
                
                if not ticker_data.empty:
                    ticker_data.to_csv(STOCK_DATA_PATH / f"{ticker}.csv", mode='w', header=True)
                    results.append(f"Successfully downloaded clean data for {ticker}")
                else:
                    results.append(f"No valid data available for {ticker} after cleaning")
            else:
                results.append(f"No data available for {ticker}")
                
        except Exception as e:
            results.append(f"Error processing {ticker}: {str(e)}")
            
    return results

def update_stocks(max_workers: int = 10, batch_size: int = 50):
    STOCK_DATA_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    tickers_df = pd.read_csv(TICKERS_PATH / "tickers-filtered.csv", usecols=['Ticker'])
    tickers_list = tickers_df['Ticker'].tolist()
    
    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i + batch_size]
        results = download_batch(batch)
        for result in results:
            print(result)
        time.sleep(1)
    
    create_log()

def main():
    filtered_path = TICKERS_PATH / "tickers-filtered.csv"
    try:
        if filtered_path.stat().st_size == 0:
            filter_tickers()
    except FileNotFoundError:
        filter_tickers()

    last_update = read_log()
    current_time = datetime.now()
    
    if last_update is None or last_update.date() != current_time.date():
        update_stocks()

def update_data():
    main() 

if __name__ == "__main__":
    update_data()


"""ISSUE
INFO:root:Starting data update process
[**********************92%*******************    ]  46 of 50 completedWARNING:urllib3.connectionpool:Connection pool is full, discarding connection: query2.finance.yahoo.com. Connection pool size: 10
[**********************94%********************   ]  47 of 50 completedWARNING:urllib3.connectionpool:Connection pool is full, discarding connection: query2.finance.yahoo.com. Connection pool size: 10
[*********************100%***********************]  50 of 50 completed
Successfully downloaded clean data for UVIX
"""
