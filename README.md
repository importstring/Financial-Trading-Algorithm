# Financial Trading Algorithm

A sophisticated trading algorithm implementation with efficient data handling using PyArrow and portfolio optimization.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:

```bash
git clone https://github.com/importstring/Financial-Trading-Algorithm.git
cd Financial-Trading-Algorithm
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Additional Setup for Windows Users

PyArrow requires a timezone database on Windows. You have two options:

1. Automatic setup (recommended):

```python
import pyarrow as pa
pa.util.download_tzdata_on_windows()
```

2. Manual setup:
   - Download the IANA timezone database
   - Extract it to `%USERPROFILE%\Downloads\tzdata`
   - Or set a custom path:

```python
import pyarrow as pa
pa.set_timezone_db_path("your_custom_path")
```

## Usage

1. Data Management:

```python
from Data.DataManagement.parquet_handler import ParquetHandler

# Initialize handler
handler = ParquetHandler("path/to/data")

# Save DataFrame
handler.save_dataframe(df, "filename")

# Read DataFrame
df = handler.read_dataframe("filename")
```

2. Portfolio Optimization:

```python
from portfolio_optimization import PortfolioOptimizer, DataManager

# Initialize components
data_manager = DataManager()
optimizer = PortfolioOptimizer()

# Get stock data and optimize
stock_data = data_manager.get_stock_data()
returns = data_manager.calculate_returns(stock_data)
```

## File Structure

- `Data/DataManagement/`: Data handling and storage utilities
  - `parquet_handler.py`: Efficient data storage using PyArrow
  - `EODHD.py`: EOD Historical Data integration
  - `smp500.py`: S&P 500 data management
- `portfolio_optimization.py`: Portfolio optimization algorithms
- `requirements.txt`: Project dependencies

## Notes

- Make sure to have adequate disk space for storing Parquet files
- For Windows users: Ensure timezone data is properly set up before running
- The algorithm uses monthly data by default for optimization
