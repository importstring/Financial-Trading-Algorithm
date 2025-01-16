# Imports
import os
from typing import List
import pandas as pd


# Paths
base_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(base_directory, 'data')
ticker_directory = os.path.join(data_directory, 'tickers')
scripts_directory = os.path.join(base_directory, 'scripts')
environment_directory = os.path.join(scripts_directory, 'environment')

# Initialize Variables
dynamic_load_allocation = {
    'Efficient Frontier': 0.40,
    'LLM Insights': 0.10,
    'Dynamic Hedging': 0.095,
    'Long Term Diversification': 0.5,
}

# Load & Update Data
pass

# Effecient Frontier
efficient_frontier_path = os.path.join(scripts_directory, 'Efficient Frontier')
if not os.path.exists(efficient_frontier_path):
    raise "Efficent Frontier not found at path " + efficient_frontier_path

def download_batch(tickers: List[str], save_directory: str) -> List[str]:
    """Download data for a batch of stock tickers."""
    results = []
    # ...existing code...
    for ticker in tickers:
        try:
            # ...existing code...
            ticker_data = get_ticker_data(ticker)
            # Ensure the DataFrame has a "Date" column
            if 'Date' not in ticker_data.columns:
                ticker_data.reset_index(inplace=True)
            # Write the DataFrame to a CSV file with unique headers
            ticker_data.to_csv(f"{save_directory}/{ticker}.csv", mode='w', header=True, index=False)
            if not ticker_data.empty:
                # ...existing code...
                results.append(f"Successfully downloaded data for {ticker}")
            else:
                results.append(f"No data available for {ticker}")
                
        except Exception as e:
            results.append(f"Error downloading {ticker}: {str(e)}")
            
    return results

# TODO: FINISH OUTLINE BY TOMORROW




