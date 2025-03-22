#!/usr/bin/env python3
"""
Example script demonstrating how to use the TimeBasedUpdater
to enforce time-based update rules.
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the TimeBasedUpdater
from Data.DataManagement.time_based_updater import TimeBasedUpdater

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("time_based_update.log")
    ]
)
logger = logging.getLogger("time_based_example")

def main():
    """Main function demonstrating TimeBasedUpdater usage"""
    
    parser = argparse.ArgumentParser(description='Demonstrate time-based update logic')
    parser.add_argument('--data-path', type=str, default=None, 
                       help='Path to data directory')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if already updated today')
    parser.add_argument('--ignore-time', action='store_true',
                       help='Ignore market hours check')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if update is needed, don\'t actually update')
    
    args = parser.parse_args()
    
    # Set data path
    data_path = None
    if args.data_path:
        data_path = Path(args.data_path)
    
    # Create the updater
    updater = TimeBasedUpdater(data_path)
    
    # Check market status
    market_closed, reason = updater.is_market_closed()
    if market_closed:
        print(f"Market is currently CLOSED: {reason}")
    else:
        print(f"Market is currently OPEN: {reason}")
    
    # Check if data was updated today
    already_updated = updater.was_updated_today()
    if already_updated:
        print("Data was already updated today")
    else:
        print("Data has not been updated today")
    
    # Check if update should be performed
    should_update, reason = updater.should_update_data(
        ignore_time=args.ignore_time,
        force=args.force
    )
    
    if should_update:
        print(f"Update is needed: {reason}")
        
        if not args.check_only:
            print("Performing update...")
            success, message = updater.update_data(
                ignore_time=args.ignore_time,
                force=args.force
            )
            print(f"Update {'successful' if success else 'failed'}: {message}")
    else:
        print(f"Update is NOT needed: {reason}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 