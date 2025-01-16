import pandas as pd
import yfinance as yf
from datetime import datetime
import os
from typing import List
import time
from functools import lru_cache
import numpy as np

# Define paths
data_directory = "/Users/simon/Financial-Trading-Algorithm/Data"
tickers_dir = os.path.join(data_directory, 'Info', 'Tickers')
logs_path = os.path.join(data_directory, 'Info', 'Logs')
save_directory = os.path.join(data_directory, 'Stock-Data')


def filter_tickers():
    filtered_path = f"{tickers_dir}/tickers-filtered.csv"
    
    # Check if file exists and is not empty
    try:
        if os.path.getsize(filtered_path) > 0:
            # Read existing tickers from filtered file
            filtered_df = pd.read_csv(filtered_path)
            return filtered_df['Ticker'].tolist()
        else:
            # If empty, create new filtered list
            tickers_df = pd.read_csv(f"{tickers_dir}/tickers.csv", usecols=['Name'])
            filtered_df = pd.DataFrame({'Ticker': tickers_df['Name'].values})
            filtered_df.to_csv(filtered_path, index=False)
            return filtered_df['Ticker'].tolist()
    except FileNotFoundError:
        # If file doesn't exist, create it
        tickers_df = pd.read_csv(f"{tickers_dir}/tickers.csv", usecols=['Name'])
        filtered_df = pd.DataFrame({'Ticker': tickers_df['Name'].values})
        filtered_df.to_csv(filtered_path, index=False)
        return filtered_df['Ticker'].tolist()

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
    results = []
    save_directory = "/Users/simon/Financial-Trading-Algorithm/Data/Stock-Data"  # Define this appropriately

    data = yf.download(tickers, period="max", auto_adjust=True, group_by='ticker')
    
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
                    ticker_data.to_csv(f"{save_directory}/{ticker}.csv", mode='w', header=True)
                    results.append(f"Successfully downloaded clean data for {ticker}")
                else:
                    results.append(f"No valid data available for {ticker} after cleaning")
            else:
                results.append(f"No data available for {ticker}")
                
        except Exception as e:
            results.append(f"Error processing {ticker}: {str(e)}")
            
    return results


def update_stocks(max_workers: int = 10, batch_size: int = 50):
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    # Read filtered tickers efficiently
    tickers_df = pd.read_csv(f"{tickers_dir}/tickers-filtered.csv", usecols=['Ticker'])
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
        filtered_size = os.path.getsize(f"{tickers_dir}/tickers-filtered.csv")
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
    
    # For debugging purposes
    # update_stocks()

def update_data():
    """
    Reroute the function call
    - For readability on the other codes on exterior function call
    """
    main() 


if __name__ == "__main__":
    update_data()
