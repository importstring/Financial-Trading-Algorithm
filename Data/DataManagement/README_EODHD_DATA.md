# EODHD S&P 500 Data Pipeline

This system implements an efficient approach for retrieving S&P 500 historical and daily stock data from EODHD's API. It follows a cost-effective strategy that performs a one-time historical data load and then appends daily updates.

## Key Features

- **Historical One-Time Loading**: Full historical data is retrieved only once per symbol to avoid costly repeated historical API calls
- **Incremental Daily Updates**: Only new data since the last update is retrieved and appended
- **Bulk Updates**: Uses EODHD's bulk API for cost-effective daily updates
- **Data Integrity**: Handles duplicate detection and ensures data consistency
- **Tracking**: Maintains records of last update dates for each symbol
- **Fault Tolerance**: Includes robust error handling and retry mechanisms
- **Time-Based Updates**: Only runs updates after market close (4pm ET) and once per day

## Components

### `historical_daily_loader.py`

The core module containing the `EODHDHistoricalClient` class which handles:

- Getting current S&P 500 constituents
- Loading complete historical data
- Updating with new daily data
- Tracking update dates
- Rate limiting API requests
- Market hour awareness

### `daily_update.py`

A command-line script designed to be run on a daily schedule (e.g., via cron job) that:

- Updates all tracked symbols with latest data
- Can perform first-time historical load when needed
- Logs all operations for monitoring
- Respects market hours and prevents duplicate updates

### `stock_insights.py`

A utility for analyzing the stock data that:

- Calculates technical indicators
- Generates analysis charts
- Compares multiple stocks
- Includes market hour awareness for data updates

### `time_based_updater.py`

A reusable utility module that provides a clean interface for time-based update logic:

- Can be imported by any other component that needs market awareness
- Checks if market is closed (after 4pm ET on weekdays)
- Tracks when data was last updated to prevent duplicate updates
- Provides a simple API to enforce update rules consistently

### `time_based_update_example.py`

An example script demonstrating how to use the TimeBasedUpdater:

- Shows how to check market status and update history
- Provides command-line flags for forcing updates or ignoring time
- Useful for testing or running manual updates when needed

## Usage

### First-Time Setup (Day 1)

Run the initial historical data load to retrieve complete historical data for all S&P 500 symbols:

```bash
# Set your API key
export EODHD_API_KEY="your_api_key"

# Run the first-time historical load
python -m Data.DataManagement.daily_update --first-time
```

This operation:

1. Gets current S&P 500 constituents
2. Downloads full historical data for each symbol
3. Saves the data in optimized Parquet format
4. Records the last update date for each symbol

### Daily Updates (Day N+)

Set up a daily cron job to run after market close:

```bash
# Example cron job (runs at 4:30 PM ET every weekday)
30 16 * * 1-5 cd /path/to/Financial-Trading-Algorithm && python -m Data.DataManagement.daily_update >> /path/to/log/daily_update.log 2>&1
```

Alternatively, use the provided setup script:

```bash
# Run the setup script to create a cron job
bash setup_cron.sh
```

Each daily update:

1. Checks if the market is closed (after 4pm ET)
2. Verifies data hasn't already been updated today
3. Uses the bulk API to efficiently retrieve the latest day's data
4. Filters for S&P 500 symbols
5. Appends new data to existing files
6. Updates the last update date records
7. Records that today's update was completed

### Using the TimeBasedUpdater in Other Code

If you're building a component that needs to be aware of market hours or check if data should be updated, use the TimeBasedUpdater:

```python
from Data.DataManagement.time_based_updater import TimeBasedUpdater

# Create the updater
updater = TimeBasedUpdater()

# Check market status
is_closed, reason = updater.is_market_closed()
if is_closed:
    print(f"Market is closed: {reason}")

# Check if we should update
should_update, reason = updater.should_update_data()
if should_update:
    # Perform the update
    success, message = updater.update_data()
    print(f"Update {'successful' if success else 'failed'}: {message}")
else:
    print(f"No update needed: {reason}")
```

### Command-Line Options

```
usage: daily_update.py [-h] [--data-path DATA_PATH] [--force-update] [--first-time] [--api-key API_KEY] [--ignore-time]

Update S&P 500 stock data from EODHD

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to store data (default: ./data)
  --force-update        Force update all tickers
  --first-time          Run first-time historical data load
  --api-key API_KEY     EODHD API key (default: from environment variable)
  --ignore-time         Ignore time check (update regardless of time)
```

## Cost Optimization

This approach minimizes API costs by:

1. **One-time historical download**: Historical data is retrieved only once per symbol
2. **Incremental updates**: Only new data points are retrieved after the initial load
3. **Bulk API usage**: Daily updates use the more cost-effective bulk endpoint
4. **Duplicate avoidance**: The system prevents redundant API calls for already updated symbols
5. **Time-based updates**: Only runs updates when the market is closed and data is finalized

## Data Storage

- Data is stored in Parquet format for efficient storage and query performance
- Each symbol has its own file for direct access
- Last update dates are tracked in a CSV file for update management
- A record of the most recent update is maintained to prevent duplicate updates

## Setting Up a Cron Job

For Linux/macOS:

1. Use the provided script (recommended):

   ```
   bash setup_cron.sh
   ```

2. Or manually set up a cron job:

   ```
   crontab -e

   # Add the line:
   30 16 * * 1-5 cd /path/to/Financial-Trading-Algorithm && python -m Data.DataManagement.daily_update
   ```

The cron job is set to run at 4:30 PM ET on weekdays, ensuring:

- Market is closed (4:00 PM ET)
- All final data for the day is available
- Updates happen automatically without manual intervention

## Integrating with Other Components

To integrate the time-based update logic with other components:

1. Import the TimeBasedUpdater in your component:

   ```python
   from Data.DataManagement.time_based_updater import TimeBasedUpdater
   ```

2. Initialize it with your data path:

   ```python
   updater = TimeBasedUpdater(data_path)
   ```

3. Use its methods to enforce consistent update rules:
   ```python
   should_update, reason = updater.should_update_data()
   if should_update:
       updater.update_data()
   ```

This approach ensures that all components respect the same market hour awareness and update tracking logic.

## Monitoring and Logging

Logs are stored in the `data/update_logs` directory with daily log files containing detailed information about:

- API requests and responses
- Data updates and changes
- Errors and retries
- Performance metrics
- Market status checks
- Update verification
