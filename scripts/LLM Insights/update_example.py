#!/usr/bin/env python3
"""
Example script demonstrating how to correctly run data updates
with the StockData class using the update_controller.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("update_example.log")
    ]
)
logger = logging.getLogger("update_example")

# Import modules
from insights import StockData
from update_controller import run_update, get_update_status

def main():
    """Main function demonstrating how to run updates correctly"""
    
    parser = argparse.ArgumentParser(description='Run EODHD data updates correctly')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if already updated today')
    parser.add_argument('--ignore-time', action='store_true',
                       help='Ignore market hours check')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check status, don\'t perform update')
    
    args = parser.parse_args()
    
    # Check if API key is set
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        logger.error("EODHD_API_KEY environment variable not set")
        print("Error: EODHD_API_KEY environment variable not set")
        return 1
    
    # Create StockData instance
    logger.info("Initializing StockData...")
    stock_data = StockData()
    
    # Get current update status
    status = get_update_status(stock_data)
    
    print("\n=== Current Status ===")
    if 'market_closed' in status:
        print(f"Market status: {'CLOSED' if status['market_closed'] else 'OPEN'}")
        print(f"Reason: {status.get('market_status_reason', 'Unknown')}")
    
    if 'updated_today' in status:
        print(f"Updated today: {'Yes' if status['updated_today'] else 'No'}")
    
    # If check-only, exit here
    if args.check_only:
        return 0
    
    # Run the update
    print("\n=== Running Update ===")
    print(f"Force update: {'Yes' if args.force else 'No'}")
    print(f"Ignore time check: {'Yes' if args.ignore_time else 'No'}")
    
    success, message = run_update(
        stock_data,
        force=args.force,
        ignore_time=args.ignore_time
    )
    
    print(f"\nUpdate result: {'SUCCESS' if success else 'FAILED/SKIPPED'}")
    print(f"Message: {message}")
    
    return 0 if success else 2

if __name__ == "__main__":
    sys.exit(main()) 