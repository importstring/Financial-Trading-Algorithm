"""
Update Controller for Financial Trading Algorithm

This module provides utility functions to ensure that data updates
use the correct functions with appropriate parameters.
"""

import logging
import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('update_controller')

def run_update(stock_data_instance, force: bool = False, ignore_time: bool = False) -> Tuple[bool, str]:
    """
    Run the stock data update with the correct function and parameters.
    
    This function ensures that the correct update method is called based on
    the available functionality in the StockData class.
    
    Args:
        stock_data_instance: The StockData instance to update
        force: Whether to force update even if already updated today
        ignore_time: Whether to ignore market hours check
        
    Returns:
        Tuple[bool, str]: Success status and message
    """
    logger.info("Update controller: Initiating stock data update")
    
    # Check if stock_data_instance has a TimeBasedUpdater
    if hasattr(stock_data_instance, 'updater'):
        logger.info("Using TimeBasedUpdater for update")
        try:
            success, message = stock_data_instance.updater.update_data(
                ignore_time=ignore_time,
                force=force
            )
            logger.info(f"Update via TimeBasedUpdater: {message}")
            return success, message
        except Exception as e:
            error_msg = f"Error using TimeBasedUpdater: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    # Otherwise, use the built-in maintain_stock_data method
    # with parameter overrides if available
    else:
        logger.info("Using maintain_stock_data for update")
        try:
            # Check if maintain_stock_data can accept parameters
            import inspect
            sig = inspect.signature(stock_data_instance.maintain_stock_data)
            
            if 'force' in sig.parameters and 'ignore_time' in sig.parameters:
                # Method accepts parameters
                result = stock_data_instance.maintain_stock_data(
                    force=force,
                    ignore_time=ignore_time
                )
                success = result is True
            else:
                # Original method without parameters
                # We'll need to manually override the checks
                if ignore_time or force:
                    logger.warning("Parameter overrides requested but not supported by maintain_stock_data")
                    
                    # Try to directly override internal checks
                    if ignore_time and hasattr(stock_data_instance, 'is_market_closed'):
                        # Monkey patch the is_market_closed method temporarily
                        original_is_market_closed = stock_data_instance.is_market_closed
                        stock_data_instance.is_market_closed = lambda: (True, "Time check overridden")
                        
                    if force and hasattr(stock_data_instance, 'was_updated_today'):
                        # Monkey patch the was_updated_today method temporarily
                        original_was_updated_today = stock_data_instance.was_updated_today
                        stock_data_instance.was_updated_today = lambda: False
                    
                    # Run the update
                    result = stock_data_instance.maintain_stock_data()
                    
                    # Restore original methods
                    if ignore_time and hasattr(stock_data_instance, 'is_market_closed'):
                        stock_data_instance.is_market_closed = original_is_market_closed
                    
                    if force and hasattr(stock_data_instance, 'was_updated_today'):
                        stock_data_instance.was_updated_today = original_was_updated_today
                else:
                    # Just call the method normally
                    result = stock_data_instance.maintain_stock_data()
                
                success = result is True
                
            message = "Update successful" if success else "Update skipped or failed"
            logger.info(f"Update via maintain_stock_data: {message}")
            return success, message
            
        except Exception as e:
            error_msg = f"Error using maintain_stock_data: {e}"
            logger.error(error_msg)
            return False, error_msg

def get_update_status(stock_data_instance) -> Dict[str, Any]:
    """
    Get the current status of market and updates.
    
    Args:
        stock_data_instance: The StockData instance
        
    Returns:
        Dict: Status information
    """
    status = {}
    
    # Check if market is closed
    if hasattr(stock_data_instance, 'is_market_closed'):
        market_closed, reason = stock_data_instance.is_market_closed()
        status['market_closed'] = market_closed
        status['market_status_reason'] = reason
    elif hasattr(stock_data_instance, 'updater') and hasattr(stock_data_instance.updater, 'is_market_closed'):
        market_closed, reason = stock_data_instance.updater.is_market_closed()
        status['market_closed'] = market_closed
        status['market_status_reason'] = reason
    
    # Check if updated today
    if hasattr(stock_data_instance, 'was_updated_today'):
        status['updated_today'] = stock_data_instance.was_updated_today()
    elif hasattr(stock_data_instance, 'updater') and hasattr(stock_data_instance.updater, 'was_updated_today'):
        status['updated_today'] = stock_data_instance.updater.was_updated_today()
    
    return status 