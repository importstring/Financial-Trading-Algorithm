# Stock Data Update Guidelines

This document explains how to correctly run data updates in the Financial Trading Algorithm system.

## Time-Based Update Implementation

The system implements time-based update logic that ensures:

1. Data updates only run after market close (4pm ET on weekdays)
2. Data is only updated once per day
3. API calls are only made when necessary

## Update Controller

The `update_controller.py` module provides utilities to ensure the correct update methods are called:

```python
from update_controller import run_update, get_update_status
```

### Key Functions

1. **run_update(stock_data_instance, force=False, ignore_time=False)**

   - Runs data updates using the correct method
   - Handles different implementations (TimeBasedUpdater or built-in methods)
   - Returns a tuple of (success, message)

2. **get_update_status(stock_data_instance)**
   - Gets the current market status and update history
   - Returns a dictionary with status information

## Correct Usage

### Preferred Method: Using StockData.update_data()

```python
from insights import StockData

# Create StockData instance
stock_data = StockData()

# Update data with the new method (recommended)
success, message = stock_data.update_data(
    force=False,      # Whether to force update even if already updated
    ignore_time=False  # Whether to ignore the market hours check
)
print(f"Update result: {success}, Message: {message}")
```

### Alternative: Direct Method Call

```python
from insights import StockData

# Create StockData instance
stock_data = StockData()

# Run update with correct parameters
result = stock_data.maintain_stock_data(
    force=False,      # Whether to force update even if already updated
    ignore_time=False  # Whether to ignore the market hours check
)
```

### Using Update Controller Directly

```python
from insights import StockData
from update_controller import run_update, get_update_status

# Create StockData instance
stock_data = StockData()

# Check status
status = get_update_status(stock_data)
print(f"Market closed: {status.get('market_closed')}")
print(f"Updated today: {status.get('updated_today')}")

# Run update with correct parameters
success, message = run_update(
    stock_data,
    force=False,       # Force update even if already updated today
    ignore_time=False  # Ignore market hours check
)
print(f"Update result: {success}, Message: {message}")
```

## Command-Line Example

You can use the included example script:

```bash
# Check current status without updating
python update_example.py --check-only

# Normal update (respects market hours and update history)
python update_example.py

# Force update regardless of previous updates
python update_example.py --force

# Ignore market hours check
python update_example.py --ignore-time

# Force update and ignore time
python update_example.py --force --ignore-time
```

## Parameters Explained

1. **force**

   - `True`: Update even if data was already updated today
   - `False`: Only update if not already updated today

2. **ignore_time**
   - `True`: Update regardless of market hours (open or closed)
   - `False`: Only update if market is closed (after 4pm ET on weekdays)

## Integration with Automated Processes

When integrating with automated workflows:

```python
from insights import StockData

def workflow_function():
    # Create instance
    stock_data = StockData()

    # Run update with appropriate parameters using the built-in method
    success, message = stock_data.update_data(
        force=False,
        ignore_time=True  # Consider the workflow's requirements
    )

    if success:
        # Continue with processing using the updated data
        data = stock_data.get_stock_data(['AAPL', 'MSFT', 'GOOG'])
        # Process data...
    else:
        # Handle case where update wasn't needed or failed
        print(f"Update not performed: {message}")
```

## Best Practices

1. **Use update_data()**: Always use the `stock_data.update_data()` method for updates
2. **Check Results**: Always check the return value to confirm if update was performed
3. **Handle Skips**: Design your code to handle cases where updates are skipped
4. **Respect Rate Limits**: Avoid forcing updates unnecessarily to respect API rate limits
5. **Refresh Data**: The `update_data()` method automatically refreshes in-memory data if successful
