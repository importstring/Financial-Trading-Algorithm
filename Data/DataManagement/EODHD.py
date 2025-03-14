import logging
import time
import random
import os
import threading
import requests
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .parquet_handler import ParquetHandler

class RateLimiter:
    """Token bucket algorithm implementation for API rate limiting"""
    
    def __init__(self, rate: float, capacity: int = 100):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.RLock()
        
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            float: Time waited in seconds
        """
        with self.lock:
            self._refill()
            wait_time = 0
            
            if tokens > self.tokens:
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(wait_time)
                self._refill()
                
            self.tokens -= tokens
            return wait_time
            
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

class EODHDClient:
    """Handles communication with EODHD API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        # Configure rate limiter for 10 requests/second with burst capacity of 100
        self.rate_limiter = RateLimiter(rate=10, capacity=100)
        
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get fundamental data for a ticker"""
        self.rate_limiter.acquire()
        url = f'https://eodhd.com/api/fundamentals/{ticker}.US'
        params = {
            'api_token': self.api_key,
            'fmt': 'json'
        }
        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

class StockDataFetcher:
    """Handles fetching and storing stock fundamental data from EODHD API."""
    
    def __init__(self, api_key: str, data_path: Path):
        """
        Initialize the stock data fetcher.
        
        Args:
            api_key: EODHD API key
            data_path: Directory path to store the data
        """
        self.api_key = api_key
        self.data_path = data_path
        self.logger = self._setup_logger()
        self.client = EODHDClient(api_key)
        self.parquet_handler = ParquetHandler(data_path)
        
        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger("stock_fetcher")
        logger.setLevel(logging.INFO)
        
        # Add file handler for persistent logging
        log_path = self.data_path / "stock_fetcher.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Add console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        return logger
    
    def get_stock_data(self, tickers: List[str], force_refresh: bool = False) -> None:
        """
        Fetch stock data with intelligent caching and concurrent processing.
        
        Args:
            tickers: List of ticker symbols
            force_refresh: Force refresh even if cached data exists
        """
        to_fetch = self._filter_cached_tickers(tickers, force_refresh)
        
        if not to_fetch:
            self.logger.info("All tickers have recent data, nothing to fetch")
            return
            
        self.logger.info(f"Fetching {len(to_fetch)} out of {len(tickers)} tickers")
        self._fetch_tickers_concurrently(to_fetch)
    
    def _filter_cached_tickers(self, tickers: List[str], force_refresh: bool) -> List[str]:
        """Filter tickers based on cache status"""
        to_fetch = []
        
        for ticker in tickers:
            try:
                metadata = self.parquet_handler.get_parquet_metadata(ticker)
                age_hours = (time.time() - metadata['last_modified']) / 3600
                
                if force_refresh or age_hours > 24:
                    to_fetch.append(ticker)
                else:
                    self.logger.info(f"Using cached data for {ticker} (age: {age_hours:.1f} hours)")
            except FileNotFoundError:
                to_fetch.append(ticker)
            except Exception as e:
                self.logger.warning(f"Error checking cache for {ticker}: {e}")
                to_fetch.append(ticker)
        
        return to_fetch
    
    def _fetch_tickers_concurrently(self, tickers: List[str], batch_size: Optional[int] = None) -> None:
        """Process tickers concurrently with automatic batch size optimization"""
        if batch_size is None:
            # Optimize batch size based on system capabilities
            batch_size = min(os.cpu_count() * 5 or 10, 50)  # Cap at 50 to respect API limits
        
        self.logger.info(f"Using batch size of {batch_size}")
        
        # Process in batches to manage memory
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(self._fetch_ticker_data_with_retry, ticker): ticker 
                    for ticker in batch
                }
                
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        self.logger.info(f"Processed {ticker}: {'Success' if result else 'Failed'}")
                    except Exception as e:
                        self.logger.error(f"Exception processing {ticker}: {e}")
    
    def _fetch_ticker_data_with_retry(self, ticker: str, max_retries: int = 3) -> bool:
        """Fetch ticker data with exponential backoff retry"""
        retry_count = 0
        base_wait_time = 2  # seconds
        
        while retry_count <= max_retries:
            try:
                start_time = time.time()
                data = self.client.get_fundamentals(ticker)
                
                # Transform data
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df['symbol'] = ticker
                
                # Save using ParquetHandler with optimized settings
                self.parquet_handler.save_dataframe(
                    df,
                    ticker,
                    compression='snappy',
                    partition_cols=['symbol']
                )
                
                elapsed = time.time() - start_time
                self.logger.info(f"Successfully processed {ticker} in {elapsed:.2f} seconds")
                return True
                
            except Exception as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    self.logger.error(f"Final failure for {ticker} after {max_retries} retries: {e}")
                    return False
                
                wait_time = base_wait_time * (2 ** (retry_count - 1))  # Exponential backoff
                jitter = random.uniform(0, 0.5 * wait_time)  # Add jitter
                total_wait = wait_time + jitter
                
                self.logger.warning(
                    f"Retry {retry_count}/{max_retries} for {ticker} after {total_wait:.2f}s. Error: {e}"
                )
                time.sleep(total_wait)
        
        return False
