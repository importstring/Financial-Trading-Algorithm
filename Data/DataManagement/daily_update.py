#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, time
import logging
import pytz
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from DataManagement.historical_daily_loader import EODHDHistoricalClient

def setup_logger(log_path):
    """Set up logger for daily update script"""
    logger = logging.getLogger("daily_update")
    logger.setLevel(logging.INFO)
    
    # Add file handler
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

def is_market_closed():
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

def was_updated_today(data_path):
    """
    Check if data was already updated today
    
    Args:
        data_path: Path to data directory
        
    Returns:
        bool: True if updated today, False otherwise
    """
    last_update_file = data_path / "last_update.json"
    
    if not last_update_file.exists():
        return False
    
    try:
        with open(last_update_file, 'r') as f:
            last_update = json.load(f)
        
        last_update_date = datetime.fromisoformat(last_update.get('date', '2000-01-01'))
        today = datetime.now().date()
        
        return last_update_date.date() == today
    except Exception as e:
        print(f"Error checking last update: {e}")
        return False

def record_update(data_path):
    """
    Record that data was updated today
    
    Args:
        data_path: Path to data directory
    """
    last_update_file = data_path / "last_update.json"
    
    try:
        with open(last_update_file, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'status': 'success'
            }, f)
    except Exception as e:
        print(f"Error recording update: {e}")

def main():
    parser = argparse.ArgumentParser(description='Update S&P 500 stock data from EODHD')
    parser.add_argument('--data-path', type=str, default=None, 
                      help='Path to store data (default: ./data)')
    parser.add_argument('--force-update', action='store_true', 
                      help='Force update all tickers')
    parser.add_argument('--first-time', action='store_true', 
                      help='Run first-time historical data load')
    parser.add_argument('--api-key', type=str, default=None,
                      help='EODHD API key (default: from environment variable)')
    parser.add_argument('--ignore-time', action='store_true',
                      help='Ignore time check (update regardless of time)')
    
    args = parser.parse_args()
    
    # Set data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Default to a data directory in the project root
        data_path = Path(__file__).parent.parent / "data"
    
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    log_path = data_path / "update_logs" / f"update_{datetime.now().strftime('%Y%m%d')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)
    
    # Check if market is closed and data hasn't been updated today
    market_closed, reason = is_market_closed()
    already_updated = was_updated_today(data_path)
    
    # First-time loads don't need to check if market is closed
    if not args.first_time and not args.force_update and not args.ignore_time:
        if not market_closed:
            logger.warning(f"Market is still open. Updates should only run after market close (4pm ET). Use --ignore-time to override.")
            return 1
        
        if already_updated:
            logger.info(f"Data has already been updated today. Use --force-update to update again.")
            return 0
    
    # Get API key
    api_key = args.api_key or os.environ.get("EODHD_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set EODHD_API_KEY environment variable or use --api-key")
        return 1
    
    # Create client
    client = EODHDHistoricalClient(api_key, data_path)
    logger.info("Starting update process")
    
    # Get S&P 500 symbols
    symbols = client.get_current_sp500_symbols()
    if not symbols:
        logger.error("Failed to retrieve S&P 500 symbols")
        return 1
    
    logger.info(f"Found {len(symbols)} S&P 500 symbols")
    
    # First-time historical load if requested
    if args.first_time:
        logger.info("Starting first-time historical data load")
        client.load_historical_data(symbols, batch_size=20)
        logger.info("Completed first-time historical data load")
    
    # Daily update
    logger.info("Starting daily update")
    if args.force_update:
        logger.info("Forcing update of all symbols")
        client.update_daily_data(symbols)
    else:
        logger.info("Using bulk API for efficiency")
        client.bulk_update_daily_data()
    
    # Record that data was updated today
    record_update(data_path)
    
    logger.info("Daily update completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 