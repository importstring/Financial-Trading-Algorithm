import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import shutil
import requests
import logging
from typing import List, Optional, Union
from EODHD import StockDataFetcher
from .parquet_handler import ParquetHandler
import time

class SavePath:
    """Manages file paths for S&P 500 data storage"""
    
    def __init__(self):
        self.path = ""
        self.name = ""
        self.parquet_handler = None

    def initialize_ticker_list(self):
        # Use pathlib for cross-platform compatibility
        self.path = Path(__file__).parent.parent / "Stock-Data"
        self.name = "S&P 500 Tickers"
        self.parquet_handler = ParquetHandler(self.path)

    def initialize_data_path(self):
        # Use pathlib for cross-platform compatibility
        self.path = Path(__file__).parent.parent / "Stock-Data"
        self.name = "S&P 500 Data"
        self.parquet_handler = ParquetHandler(self.path)

    def save_data(self, data: Union[list, pd.DataFrame]) -> str:
        """
        Save data to file with appropriate format based on data type.
        
        Args:
            data: Data to save (list or DataFrame)
            
        Returns:
            str: Path where the data was saved
            
        Raises:
            TypeError: If data type is not supported
        """
        self.path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, list):
            file_path = self.path / f"{self.name}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f)
            return str(file_path)
        
        elif isinstance(data, pd.DataFrame):
            return str(self.parquet_handler.save_dataframe(
                data,
                self.name,
                compression='snappy'
            ))

class Smp500Tickers:
    """Manages S&P 500 ticker data and stock information fetching"""
    
    def __init__(self):
        self.url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.tickers: Optional[List[str]] = None
        self.ticker_path = SavePath()
        self.ticker_path.initialize_ticker_list()
        self.data_path = SavePath()
        self.data_path.initialize_data_path()
        self.api_key = "67c48dde837ed4.18473716"
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance"""
        logger = logging.getLogger("smp500_manager")
        logger.setLevel(logging.INFO)
        
        # Add file handler
        log_path = Path(self.data_path.path) / "smp500_manager.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        return logger

    def get_tickers(self) -> List[str]:
        """
        Retrieve current S&P 500 tickers from Wikipedia.
        
        Returns:
            List[str]: List of ticker symbols
            
        Raises:
            ValueError: If ticker data cannot be retrieved
        """
        try:
            tables = pd.read_html(self.url)
            if not tables:
                raise ValueError("No tables found on S&P 500 Wikipedia page")
                
            ticker_table = tables[0]
            if 'Symbol' not in ticker_table.columns:
                raise ValueError("Symbol column not found in S&P 500 table")
                
            self.tickers = ticker_table['Symbol'].tolist()
            self.logger.info(f"Successfully retrieved {len(self.tickers)} S&P 500 tickers")
            return self.tickers
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve S&P 500 tickers: {e}")
            # Return cached tickers if available
            if hasattr(self, 'tickers') and self.tickers:
                self.logger.warning("Using cached tickers as fallback")
                return self.tickers
            raise

    def save_data(self) -> str:
        """
        Save stock data to parquet format.
        
        Returns:
            str: Path where the data was saved
        """
        if not hasattr(self, 'data') or self.data is None:
            self.get_data()
        
        self.logger.info("Saving stock data...")
        return self.data_path.save_data(self.data)
    
    def update_ticker_list(self) -> str:
        """
        Update and save the S&P 500 ticker list.
        
        Returns:
            str: Path where the ticker list was saved
        """
        start_time = time.time()
        self.logger.info("Updating S&P 500 ticker list...")
        
        tickers = self.get_tickers()
        file_path = self.ticker_path.save_data(tickers)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Updated ticker list saved to {file_path} in {elapsed:.2f} seconds")
        return file_path

    def update_stock_data(self, force_refresh: bool = False) -> None:
        """
        Update stock data for all S&P 500 companies.
        
        Args:
            force_refresh: Force refresh all data regardless of cache
        """
        start_time = time.time()
        self.logger.info("Starting S&P 500 stock data update...")
        
        try:
            # Update ticker list first
            self.update_ticker_list()
            
            if force_refresh:
                self.clear_stock_directory()
            
            # Initialize data fetcher with performance monitoring
            data_fetcher = StockDataFetcher(self.api_key, Path(self.data_path.path))
            
            # Fetch data with caching
            data_fetcher.get_stock_data(self.tickers, force_refresh=force_refresh)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed stock data update in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to update stock data: {e}")
            raise

    def clear_stock_directory(self) -> None:
        """Safely clear all files in the data directory while preserving structure"""
        directory = self.data_path.path
        self.logger.info(f"Clearing directory: {directory}")
        
        try:
            for item in directory.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                        self.logger.debug(f"Deleted file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self.logger.debug(f"Deleted directory: {item}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {item}: {e}")
                    
            self.logger.info("Successfully cleared stock directory")
            
        except Exception as e:
            self.logger.error(f"Failed to clear stock directory: {e}")
            raise
    
    def main(self, force_refresh: bool = False) -> None:
        """
        Main execution flow for S&P 500 data management.
        
        Args:
            force_refresh: Force refresh all data regardless of cache
        """
        start_time = time.time()
        self.logger.info("Starting S&P 500 data management process...")
        
        try:
            self.update_stock_data(force_refresh=force_refresh)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed all tasks in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Main process failed: {e}")
            raise

def main():
    """Entry point for S&P 500 data management"""
    smp500 = Smp500Tickers()
    smp500.main(force_refresh=False)  # Set to True to force refresh all data

if __name__ == "__main__":
    main()