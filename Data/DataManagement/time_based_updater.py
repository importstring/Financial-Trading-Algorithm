import os
import logging
import json
import pytz
from pathlib import Path
from datetime import datetime, time
from typing import Tuple, Optional

# Import from DataManagement package
from .historical_daily_loader import EODHDHistoricalClient

logger = logging.getLogger("time_based_updater")

class TimeBasedUpdater:
    """
    Utility class for handling time-based stock data updates.
    This provides a clean interface to implement the market hour
    awareness logic that can be imported by other modules.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the updater with data path
        
        Args:
            data_path: Path to data directory. If None, uses environment variable or default.
        """
        if data_path is None:
            data_path = Path(os.getenv('DATA_PATH', './data'))
        
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize EODHD client if API key is available
        api_key = os.environ.get("EODHD_API_KEY")
        self.client = None
        if api_key:
            self.client = EODHDHistoricalClient(api_key, self.data_path)
    
    def is_market_closed(self) -> Tuple[bool, str]:
        """
        Check if US market is closed (after 4pm ET on weekdays)
        
        Returns: 
            tuple: (is_closed, reason)
        """
        # Get current time in ET
        et_zone = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_zone)
        
        # Check if weekend
        if now_et.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return True, "Weekend"
        
        # Check if before market open (9:30am ET) or after market close (4:00pm ET)
        market_open = time(9, 30, 0)
        market_close = time(16, 0, 0)
        
        if now_et.time() < market_open:
            return True, "Before market open"
        elif now_et.time() >= market_close:
            return True, "After market close"
        
        # Market is open
        return False, "Market is open"

    def was_updated_today(self) -> bool:
        """
        Check if data was already updated today
        
        Returns:
            bool: True if updated today, False otherwise
        """
        last_update_file = self.data_path / "last_update.json"
        
        if not last_update_file.exists():
            return False
        
        try:
            with open(last_update_file, 'r') as f:
                last_update = json.load(f)
            
            last_update_date = datetime.fromisoformat(last_update.get('date', '2000-01-01'))
            today = datetime.now().date()
            
            return last_update_date.date() == today
        except Exception as e:
            logger.error(f"Error checking last update: {e}")
            return False

    def record_update(self) -> None:
        """
        Record that data was updated today
        """
        last_update_file = self.data_path / "last_update.json"
        
        try:
            with open(last_update_file, 'w') as f:
                json.dump({
                    'date': datetime.now().isoformat(),
                    'status': 'success'
                }, f)
        except Exception as e:
            logger.error(f"Error recording update: {e}")

    def should_update_data(self, ignore_time: bool = False, force: bool = False) -> Tuple[bool, str]:
        """
        Check if data should be updated based on market hours and previous updates
        
        Args:
            ignore_time: Whether to ignore market hours check
            force: Whether to force update even if already updated today
            
        Returns:
            tuple: (should_update, reason)
        """
        # Always need an API key and client
        if not os.environ.get("EODHD_API_KEY"):
            return False, "No EODHD API key found in environment"
            
        if not self.client:
            api_key = os.environ.get("EODHD_API_KEY")
            self.client = EODHDHistoricalClient(api_key, self.data_path)
        
        # Check market hours unless ignoring time
        if not ignore_time:
            market_closed, reason = self.is_market_closed()
            if not market_closed:
                return False, f"Market is still open ({reason})"
        
        # Check if already updated today unless forcing
        if not force:
            if self.was_updated_today():
                return False, "Data has already been updated today"
        
        # All checks passed, data should be updated
        return True, "Update needed"

    def update_data(self, ignore_time: bool = False, force: bool = False) -> Tuple[bool, str]:
        """
        Update stock data if appropriate based on market hours and previous updates
        
        Args:
            ignore_time: Whether to ignore market hours check
            force: Whether to force update even if already updated today
            
        Returns:
            tuple: (success, message)
        """
        should_update, reason = self.should_update_data(ignore_time, force)
        
        if not should_update:
            logger.info(f"Stock data update skipped: {reason}")
            return False, reason
        
        try:
            logger.info("Starting stock data update")
            self.client.bulk_update_daily_data()
            self.record_update()
            logger.info("Stock data update completed successfully")
            return True, "Update successful"
        except Exception as e:
            error_msg = f"Error updating stock data: {e}"
            logger.error(error_msg)
            return False, error_msg 