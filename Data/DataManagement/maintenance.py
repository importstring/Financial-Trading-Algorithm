"""
Stock Market Data Manager

This module provides functionality to manage and update S&P 500 stock data.
It includes components for fetching tickers, downloading stock data, and managing data files.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import time
from functools import lru_cache
import numpy as np
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Optional

# Configuration
class PathConfig:
    """Manages all path configurations for the application."""
    
    @staticmethod
    def get_project_root() -> Path:
        """Find and return the project root directory."""
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / '.git').exists():
                return parent
            if parent.name == 'Financial-Trading-Algorithm-progress-updated':
                return parent
        raise FileNotFoundError("Could not find project root directory")

    def __init__(self):
        self.PROJECT_ROOT = self.get_project_root()
        self.DATA_PATH = self.PROJECT_ROOT / 'Data'
        self.TICKERS_PATH = self.DATA_PATH / 'Info' / 'Tickers'
        self.LOGS_PATH = self.DATA_PATH / 'Info' / 'Logs'
        self.STOCK_DATA_PATH = self.DATA_PATH / 'Stock-Data'
        
        # Ensure all directories exist
        for path in [self.DATA_PATH, self.TICKERS_PATH, self.LOGS_PATH, self.STOCK_DATA_PATH]:
            path.mkdir(parents=True, exist_ok=True)

class SP500Tickers:
    """Manages S&P 500 ticker data retrieval and storage."""
    
    def __init__(self, paths: PathConfig):
        self.url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.tickers = None
        self.path = paths.TICKERS_PATH / "tickers.csv"

    def get_tickers(self) -> pd.DataFrame:
        """Fetch current S&P 500 tickers from Wikipedia."""
        tables = pd.read_html(self.url)
        df = tables[0][['Symbol', 'Security']]
        df.columns = ['Ticker', 'Name']
        self.tickers = df
        return df

    def save_data(self) -> None:
        """Save the S&P 500 tickers to CSV."""
        if self.tickers is None:
            self.get_tickers()
        self.tickers.to_csv(self.path, index=False)

    def update(self) -> None:
        """Update and save tickers data."""
        self.get_tickers()
        self.save_data()

class StockDataManager:
    """Manages stock data downloading and processing."""
    
    def __init__(self, paths: PathConfig):
        self.paths = paths
        self.session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """Create a session with retry logic."""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    @lru_cache(maxsize=1000)
    def get_cached_ticker(self, ticker: str) -> yf.Ticker:
        """Get cached ticker object."""
        return yf.Ticker(ticker)

    def download_batch(self, tickers: List[str]) -> List[str]:
        """Download and process a batch of stock data."""
        results = []
        data = yf.download(tickers, period="max", auto_adjust=True, group_by='ticker', session=self.session)
        
        for ticker in tickers:
            try:
                ticker_data = self._process_ticker_data(data, ticker)
                if ticker_data is not None:
                    self._save_ticker_data(ticker_data, ticker)
                    results.append(f"Successfully downloaded clean data for {ticker}")
                else:
                    results.append(f"No valid data available for {ticker}")
            except Exception as e:
                results.append(f"Error processing {ticker}: {str(e)}")
        
        return results

    def _process_ticker_data(self, data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Process raw ticker data into clean format."""
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            ticker_data = data[ticker].copy()
        else:
            ticker_data = data[ticker]

        if ticker_data.empty:
            return None

        necessary_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        ticker_data = ticker_data[necessary_columns]
        ticker_data = ticker_data.dropna()
        ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
        ticker_data = ticker_data.sort_index()

        # Convert float64 to float32 for efficiency
        for col in ticker_data.select_dtypes(include=[np.float64]).columns:
            ticker_data[col] = ticker_data[col].astype(np.float32)

        return ticker_data if not ticker_data.empty else None

    def _save_ticker_data(self, data: pd.DataFrame, ticker: str) -> None:
        """Save processed ticker data to CSV."""
        data.to_csv(self.paths.STOCK_DATA_PATH / f"{ticker}.csv", mode='w', header=True)

class DataUpdateManager:
    """Manages the overall data update process."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.sp500 = SP500Tickers(self.paths)
        self.stock_manager = StockDataManager(self.paths)

    def create_log(self) -> None:
        """Create update log with current timestamp."""
        log_file = self.paths.LOGS_PATH / "stock-price-last-updated.txt"
        log_file.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def read_log(self) -> Optional[datetime]:
        """Read last update timestamp."""
        log_file = self.paths.LOGS_PATH / "stock-price-last-updated.txt"
        try:
            return datetime.strptime(log_file.read_text().strip(), "%Y-%m-%d %H:%M:%S")
        except (FileNotFoundError, ValueError):
            return None

    def filter_tickers(self) -> List[str]:
        """Get and filter S&P 500 tickers."""
        filtered_path = self.paths.TICKERS_PATH / "tickers-filtered.csv"
        
        try:
            if filtered_path.exists() and filtered_path.stat().st_size > 0:
                return pd.read_csv(filtered_path)['Ticker'].tolist()
        except FileNotFoundError:
            pass

        self.sp500.update()
        tickers_df = pd.read_csv(self.paths.TICKERS_PATH / "tickers.csv")
        filtered_df = pd.DataFrame({'Ticker': tickers_df['Ticker'].values})
        filtered_df.to_csv(filtered_path, index=False)
        return filtered_df['Ticker'].tolist()

    def cleanup_stock_data(self) -> None:
        """Remove stock data for tickers no longer in S&P 500."""
        current_tickers = self.sp500.get_tickers()['Ticker'].tolist()
        
        if not self.paths.STOCK_DATA_PATH.exists():
            return
            
        for file_path in self.paths.STOCK_DATA_PATH.glob('*.csv'):
            ticker = file_path.stem
            if ticker not in current_tickers:
                try:
                    file_path.unlink()
                    print(f"Removed data for {ticker} - no longer in S&P 500")
                except Exception as e:
                    print(f"Error removing {ticker}: {str(e)}")

    def update_stocks(self, max_workers: int = 10, batch_size: int = 10) -> None:
        """Update stock data for all tickers."""
        tickers_list = pd.read_csv(self.paths.TICKERS_PATH / "tickers-filtered.csv")['Ticker'].tolist()
        
        for i in range(0, len(tickers_list), batch_size):
            batch = tickers_list[i:i + batch_size]
            results = self.stock_manager.download_batch(batch)
            for result in results:
                print(result)
            time.sleep(2)
        
        self.create_log()

    def update(self) -> None:
        """Main update process."""
        # Ensure we have current tickers
        self.filter_tickers()

        # Check if update is needed
        last_update = self.read_log()
        current_time = datetime.now()
        
        if last_update is None or last_update.date() != current_time.date():
            self.cleanup_stock_data()
            self.update_stocks()

def main():
    """Main entry point."""
    updater = DataUpdateManager()
    updater.update()

if __name__ == "__main__":
    main()

