import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import concurrent.futures
from typing import Optional, List
import time
from functools import lru_cache
import numpy as np

# Define paths
data_directory = "/Users/simon/Financial-Trading-Algorithm/Data"
tickers_list = os.path.join(data_directory, 'Info', 'Tickers')
logs_path = os.path.join(data_directory, 'Info', 'Logs')
save_directory = os.path.join(data_directory, 'Stock-Data')


def filter_tickers():
    # Read only necessary columns
    tickers_df = pd.read_csv(f"{tickers_list}/tickers.csv", usecols=['Name'])
    
    # More efficient DataFrame creation
    filtered_df = pd.DataFrame({'Ticker': tickers_df['Name'].values})
    
    filtered_df.to_csv(f"{tickers_list}/tickers-filtered.csv", index=False)

def create_log():
    # Create log file with current timestamp
    with open(f"{logs_path}/stock-price-last-updated.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def read_log():
    try:
        with open(f"{logs_path}/stock-price-last-updated.txt", "r") as f:
            last_update = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
            return last_update
    except (FileNotFoundError, ValueError):
        return None

@lru_cache(maxsize=1000)
def get_cached_ticker(ticker: str) -> yf.Ticker:
    """Cache Ticker objects to prevent repeated API calls."""
    return yf.Ticker(ticker)

def download_batch(tickers: List[str]) -> List[str]:
    """Download data for a batch of stock tickers."""
    results = []
    # Download data for multiple tickers at once
    data = yf.download(tickers, period="max", auto_adjust=True, group_by='ticker')
    
    for ticker in tickers:
        try:
            if isinstance(data, pd.DataFrame):
                ticker_data = data
            else:
                ticker_data = data[ticker]
            
            if not ticker_data.empty:
                # Optimize memory usage by converting to float32
                for col in ticker_data.select_dtypes(include=[np.float64]).columns:
                    ticker_data[col] = ticker_data[col].astype(np.float32)
                
                ticker_data.to_csv(f"{save_directory}/{ticker}.csv")
                results.append(f"Successfully downloaded data for {ticker}")
            else:
                results.append(f"No data available for {ticker}")
                
        except Exception as e:
            results.append(f"Error downloading {ticker}: {str(e)}")
            
    return results

def update_stocks(max_workers: int = 10, batch_size: int = 50):
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    # Read filtered tickers efficiently
    tickers_df = pd.read_csv(f"{tickers_list}/tickers-filtered.csv", usecols=['Ticker'])
    tickers_list = tickers_df['Ticker'].tolist()
    
    # Process tickers in batches
    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i + batch_size]
        results = download_batch(batch)
        for result in results:
            print(result)
        # Add delay between batches to prevent rate limiting
        time.sleep(1)
    
    create_log()

def main():
    # Check if tickers-filtered.csv is empty or doesn't exist
    try:
        filtered_size = os.path.getsize(f"{tickers_list}/tickers-filtered.csv")
        if filtered_size == 0:
            filter_tickers()
    except FileNotFoundError:
        filter_tickers()

    # Check if log exists and is current
    last_update = read_log()
    current_time = datetime.now()
    
    # If log doesn't exist or is not from today
    if last_update is None or last_update.date() != current_time.date():
        update_stocks()

if __name__ == "__main__":
    main()
