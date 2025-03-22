import logging
import time
import os
import threading
import requests
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta, time as dt_time
import pytz
from .parquet_handler import ParquetHandler

class EODHDHistoricalClient:
    """Handles efficient historical and daily data loading from EODHD API"""
    
    def __init__(self, api_key: str, data_path: Path):
        """
        Initialize the EODHD client for efficient historical and daily data loading.
        
        Args:
            api_key: EODHD API key
            data_path: Directory path to store the data
        """
        self.api_key = api_key
        self.data_path = data_path
        self.session = requests.Session()
        self.logger = self._setup_logger()
        self.parquet_handler = ParquetHandler(data_path)
        
        # Configure rate limiter (adjust according to your API plan)
        self.rate_limiter = RateLimiter(rate=10, capacity=100)
        
        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Track the last update date for each symbol
        self.last_update_file = self.data_path / "last_updates.csv"
        self.last_updates = self._load_last_updates()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger("eod_historical_client")
        logger.setLevel(logging.INFO)
        
        # Add file handler for persistent logging
        log_path = self.data_path / "eod_client.log"
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
    
    def _load_last_updates(self) -> Dict[str, datetime]:
        """Load the last update dates for each symbol"""
        try:
            if self.last_update_file.exists():
                df = pd.read_csv(self.last_update_file)
                return {row['symbol']: pd.to_datetime(row['last_update']) 
                        for _, row in df.iterrows()}
            return {}
        except Exception as e:
            self.logger.error(f"Error loading last updates: {e}")
            return {}
    
    def _save_last_updates(self) -> None:
        """Save the last update dates for each symbol"""
        try:
            df = pd.DataFrame([
                {'symbol': symbol, 'last_update': date.strftime('%Y-%m-%d')}
                for symbol, date in self.last_updates.items()
            ])
            df.to_csv(self.last_update_file, index=False)
        except Exception as e:
            self.logger.error(f"Error saving last updates: {e}")
    
    def get_current_sp500_symbols(self) -> List[str]:
        """Get current S&P 500 constituents"""
        self.rate_limiter.acquire()
        url = 'https://eodhd.com/api/indices/constituents/GSPC.INDX'
        params = {
            'api_token': self.api_key,
            'fmt': 'json'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                return [item['symbol'] for item in data]
            elif 'constituents' in data:
                return [item['symbol'] for item in data['constituents']]
            else:
                self.logger.error(f"Unexpected response format: {data}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 constituents: {e}")
            return []
    
    def load_historical_data(self, symbols: List[str], batch_size: int = 20) -> None:
        """
        Load complete historical data for a list of symbols.
        This should only be done once per symbol.
        
        Args:
            symbols: List of ticker symbols
            batch_size: Batch size for concurrent processing
        """
        self.logger.info(f"Starting historical data load for {len(symbols)} symbols")
        
        # Filter symbols that don't have historical data yet
        symbols_to_load = []
        for symbol in symbols:
            if symbol not in self.last_updates:
                symbols_to_load.append(symbol)
            else:
                self.logger.info(f"Historical data already exists for {symbol}")
        
        if not symbols_to_load:
            self.logger.info("No symbols need historical data loading")
            return
            
        self.logger.info(f"Loading historical data for {len(symbols_to_load)} symbols")
        
        # Process in batches to manage memory and API rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(self._fetch_historical_data, symbol): symbol 
                for symbol in symbols_to_load
            }
            
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    success, last_date = future.result()
                    if success:
                        self.last_updates[symbol] = last_date
                        self.logger.info(f"Successfully loaded historical data for {symbol}")
                    else:
                        self.logger.error(f"Failed to load historical data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Exception loading historical data for {symbol}: {e}")
        
        # Save the updated last update dates
        self._save_last_updates()
    
    def _fetch_historical_data(self, symbol: str) -> Tuple[bool, Optional[datetime]]:
        """
        Fetch complete historical data for a symbol
        
        Returns:
            Tuple[bool, Optional[datetime]]: (success, last_date)
        """
        try:
            self.rate_limiter.acquire()
            
            url = f'https://eodhd.com/api/eod/{symbol}.US'
            params = {
                'api_token': self.api_key,
                'from': '1970-01-01',  # Starting from 1970 to get all historical data
                'fmt': 'json',
                'order': 'a'  # Ascending order
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                self.logger.warning(f"No historical data returned for {symbol}")
                return False, None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            
            # Save to parquet
            self.parquet_handler.save_dataframe(
                df,
                symbol,
                compression='snappy',
                partition_cols=None  # No partitioning needed for historical data
            )
            
            last_date = df['date'].max()
            return True, last_date
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return False, None
    
    def update_daily_data(self, symbols: Optional[List[str]] = None) -> None:
        """
        Update data for symbols with the latest daily data.
        Only get data since the last update date.
        
        Args:
            symbols: Optional list of symbols to update. If None, update all tracked symbols.
        """
        if symbols is None:
            symbols = list(self.last_updates.keys())
            
        if not symbols:
            self.logger.warning("No symbols to update")
            return
            
        self.logger.info(f"Updating daily data for {len(symbols)} symbols")
        
        today = datetime.now().date()
        updated_symbols = []
        
        # Check if market is closed (simple implementation)
        if not self._is_market_closed():
            self.logger.info("Market may still be open, daily data might be incomplete")
        
        for symbol in symbols:
            last_update = self.last_updates.get(symbol, None)
            
            # Skip if updated today
            if last_update and last_update.date() == today:
                self.logger.info(f"Symbol {symbol} already updated today")
                continue
                
            # Calculate from_date (day after last update)
            from_date = None
            if last_update:
                from_date = (last_update + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # No previous data, use historical fetch instead
                self.logger.info(f"No previous data for {symbol}, using historical fetch")
                success, last_date = self._fetch_historical_data(symbol)
                if success:
                    self.last_updates[symbol] = last_date
                    updated_symbols.append(symbol)
                continue
            
            # Fetch and append daily data
            success, last_date = self._fetch_daily_update(symbol, from_date)
            if success:
                self.last_updates[symbol] = last_date
                updated_symbols.append(symbol)
        
        if updated_symbols:
            self.logger.info(f"Successfully updated {len(updated_symbols)} symbols")
            self._save_last_updates()
        else:
            self.logger.info("No symbols were updated")
    
    def _fetch_daily_update(self, symbol: str, from_date: str) -> Tuple[bool, Optional[datetime]]:
        """
        Fetch daily updates for a symbol since a specific date
        
        Args:
            symbol: Ticker symbol
            from_date: Start date in YYYY-MM-DD format
            
        Returns:
            Tuple[bool, Optional[datetime]]: (success, last_date)
        """
        try:
            self.rate_limiter.acquire()
            
            url = f'https://eodhd.com/api/eod/{symbol}.US'
            params = {
                'api_token': self.api_key,
                'from': from_date,
                'fmt': 'json',
                'order': 'a'  # Ascending order
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                self.logger.info(f"No new data for {symbol} since {from_date}")
                return True, self.last_updates.get(symbol)  # No new data is not an error
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            
            # Get existing data
            try:
                existing_df = self.parquet_handler.load_dataframe(symbol)
                
                # Filter out any overlapping dates (avoid duplicates)
                new_dates = set(df['date'].dt.date)
                existing_dates = set(existing_df['date'].dt.date)
                duplicate_dates = new_dates.intersection(existing_dates)
                
                if duplicate_dates:
                    self.logger.warning(f"Found {len(duplicate_dates)} duplicate dates for {symbol}")
                    df = df[~df['date'].dt.date.isin(duplicate_dates)]
                
                if df.empty:
                    self.logger.info(f"No new unique data for {symbol}")
                    return True, self.last_updates.get(symbol)
                
                # Append new data to existing data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # Save combined data
                self.parquet_handler.save_dataframe(
                    combined_df,
                    symbol,
                    compression='snappy',
                    partition_cols=None
                )
                
                last_date = combined_df['date'].max()
                
            except FileNotFoundError:
                # No existing file, save new data directly
                self.logger.warning(f"No existing file found for {symbol}, creating new file")
                self.parquet_handler.save_dataframe(
                    df,
                    symbol,
                    compression='snappy',
                    partition_cols=None
                )
                
                last_date = df['date'].max()
            
            return True, last_date
            
        except Exception as e:
            self.logger.error(f"Error fetching daily update for {symbol}: {e}")
            return False, None
    
    def _is_market_closed(self) -> bool:
        """
        Check if the US market is closed
        
        Returns:
            bool: True if market is closed, False otherwise
        """
        try:
            # Get today's date in US Eastern time
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            
            # Check if weekend
            if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                self.logger.info("Market is closed (weekend)")
                return True
            
            # Check if holiday (simplified, would need a proper calendar for production)
            # Major US holidays: New Year's, MLK Jr. Day, Presidents Day, Good Friday, 
            # Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas
            
            # Check if market hours (9:30 AM - 4:00 PM ET)
            market_open = dt_time(9, 30, 0)
            market_close = dt_time(16, 0, 0)
            
            if now.time() < market_open:
                self.logger.info(f"Market is closed (before {market_open})")
                return True
            elif now.time() >= market_close:
                self.logger.info(f"Market is closed (after {market_close})")
                return True
                
            # Market is likely open
            self.logger.info("Market appears to be open")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            # If we can't determine, assume market is closed to be safe
            return True
    
    def bulk_update_daily_data(self) -> None:
        """
        Update all symbols using the bulk EOD API for efficiency.
        This is more cost-effective for daily updates of many symbols.
        """
        try:
            self.logger.info("Starting bulk daily update")
            
            # Check if market is closed
            if not self._is_market_closed():
                self.logger.warning("Market may still be open, daily data might be incomplete")
            
            self.rate_limiter.acquire(tokens=5)  # Bulk API costs more tokens
            
            url = 'https://eodhd.com/api/eod-bulk-last-day/US'
            params = {
                'api_token': self.api_key,
                'fmt': 'json'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            bulk_data = response.json()
            
            if not bulk_data:
                self.logger.warning("No data returned from bulk API")
                return
            
            # Get current S&P 500 symbols
            sp500_symbols = self.get_current_sp500_symbols()
            sp500_set = set(sp500_symbols)
            
            # Filter for S&P 500 symbols
            filtered_data = []
            for item in bulk_data:
                ticker = item.get('code', '').replace('.US', '')
                if ticker in sp500_set:
                    item['symbol'] = ticker  # Add symbol field
                    filtered_data.append(item)
            
            bulk_df = pd.DataFrame(filtered_data)
            
            if bulk_df.empty:
                self.logger.warning("No S&P 500 data in bulk response")
                return
            
            # Convert date column
            bulk_df['date'] = pd.to_datetime(bulk_df['date'])
            today = datetime.now().date()
            
            # Update each symbol
            updated_count = 0
            for symbol in sp500_symbols:
                if symbol not in self.last_updates:
                    self.logger.info(f"No historical data for {symbol}, skipping daily update")
                    continue
                
                symbol_data = bulk_df[bulk_df['symbol'] == symbol]
                
                if symbol_data.empty:
                    self.logger.warning(f"No bulk data for {symbol}")
                    continue
                
                try:
                    # Load existing data
                    existing_df = self.parquet_handler.load_dataframe(symbol)
                    
                    # Append new data, avoiding duplicates
                    existing_dates = set(existing_df['date'].dt.date)
                    new_data = symbol_data[~symbol_data['date'].dt.date.isin(existing_dates)]
                    
                    if new_data.empty:
                        self.logger.info(f"No new data for {symbol}")
                        continue
                    
                    # Combine and save
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                    self.parquet_handler.save_dataframe(
                        combined_df,
                        symbol,
                        compression='snappy',
                        partition_cols=None
                    )
                    
                    # Update last_updates
                    last_date = combined_df['date'].max()
                    self.last_updates[symbol] = last_date
                    updated_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error updating {symbol} with bulk data: {e}")
            
            self.logger.info(f"Bulk update completed, updated {updated_count} symbols")
            
            if updated_count > 0:
                self._save_last_updates()
                
        except Exception as e:
            self.logger.error(f"Error in bulk update: {e}")


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


# Example usage
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Get API key from environment variable
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        print("Please set the EODHD_API_KEY environment variable")
        exit(1)
    
    # Create client
    data_path = Path("./data")
    client = EODHDHistoricalClient(api_key, data_path)
    
    # Get S&P 500 symbols
    symbols = client.get_current_sp500_symbols()
    print(f"Found {len(symbols)} S&P 500 symbols")
    
    # First-time historical load (only needed once per symbol)
    # Uncomment to run initial historical load
    # client.load_historical_data(symbols, batch_size=20)
    
    # Daily update (run this daily after market close)
    client.bulk_update_daily_data() 