# ============ Imports ============ #
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# ============ Functions ============ #
def get_data(stock_data_dir):
    """
    Get the data from the contents of the Stock Data Directory
    Input: Stock Data Directory --> str
    Output: Stock Data --> Dictionary {Ticker: Data}
    """
    stock_data = {}
    # Helper function to load a single CSV file
    def load_file(file, stock_data_dir):
        # Only process CSV files
        if file.endswith('.csv'):
            # Extract ticker name from filename
            ticker = file.split('.')[0]
            # Read the CSV file
            file_path = os.path.join(stock_data_dir, file)
            data = pd.read_csv(file_path)
            # Handle missing values before calculating returns
            data['Close'] = data['Close'].ffill().bfill()  # Updated line
            data['Return'] = data['Close'].pct_change(fill_method=None)
            data = data.dropna()
            return ticker, data
        return None

    # Get list of all CSV files in directory
    csv_files = [
        file for file in os.listdir(stock_data_dir) 
        if file.endswith('.csv')
    ]

    # Set up parallel processing
    with ThreadPoolExecutor() as executor:
        # Create partial function with fixed stock_data_dir
        load_func = partial(load_file, stock_data_dir=stock_data_dir)
        # Load all files in parallel
        results = executor.map(load_func, csv_files)

    # Filter out None results and convert to dictionary
    stock_data = dict(filter(None, results))
    return stock_data

def calculate_returns(stock_data):
    """
    Calculate the returns from the stock data
    Input: Stock Data --> Dictionary {Ticker: Data}
    Output: Stock Returns --> Dictionary {Ticker: Returns}
    """
    stock_returns = {}
    for ticker, data in stock_data.items():
        # Calculate daily returns
        data['Return'] = data['Close'].pct_change()
        # Drop first row with NaN
        data = data.dropna()
        # Store returns in dictionary
        stock_returns[ticker] = data['Return']
    return stock_returns

def calculate_covariance_matrix(stock_returns):
    """
    Calculate the covariance matrix from the stock returns
    Input: Stock Returns --> Dictionary {Ticker: Returns}
    Output: Covariance Matrix --> DataFrame
    """
    # Concatenate returns into single DataFrame
    returns_df = pd.concat(stock_returns, axis=1)
    # Calculate covariance matrix
    covariance_matrix = returns_df.cov()
    return covariance_matrix

def calculate_portfolio_return(weights, stock_returns):
    """
    Calculate the mean return of a portfolio given the weights and stock returns
    Input: Weights --> List, Stock Returns --> Dictionary {Ticker: Returns}
    Output: Portfolio Return --> float
    """
    # Calculate weighted returns for each stock
    weighted_returns = {
        ticker: stock_returns[ticker] * weight
        for ticker, weight in zip(stock_returns.keys(), weights)
    }
    # Calculate total return of portfolio
    portfolio_return_series = pd.DataFrame(weighted_returns).sum(axis=1)
    return portfolio_return_series.mean()  # Return mean instead of Series

def calculate_portfolio_variance(weights, covariance_matrix):
    """
    Calculate the variance of a portfolio given the weights and covariance matrix
    Input: Weights --> List, Covariance Matrix --> DataFrame
    Output: Portfolio Variance --> float
    """
    # Calculate weighted covariance for each stock pair
    weighted_covariance = covariance_matrix.mul(weights, axis=0).mul(weights, axis=1)
    # Calculate total variance of portfolio
    portfolio_variance = weighted_covariance.values.sum()
    return portfolio_variance

def calculate_portfolio_metrics(weights, stock_returns, covariance_matrix):
    """
    Calculate the return and variance of a portfolio given the weights, stock returns and covariance matrix
    Both return values will now be float scalars
    Input: Weights --> List, Stock Returns --> Dictionary {Ticker: Returns}, Covariance Matrix --> DataFrame
    Output: Portfolio Return --> float, Portfolio Variance --> float
    """
    # Calculate portfolio return
    portfolio_return = calculate_portfolio_return(weights, stock_returns)
    # Calculate portfolio variance
    portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
    return portfolio_return, portfolio_variance

def calculate_efficient_frontier(stock_returns, covariance_matrix, num_points=100):
    """
    Calculate the efficient frontier given the stock returns and covariance matrix
    Input: Stock Returns --> Dictionary {Ticker: Returns}, Covariance Matrix --> DataFrame, Num Points --> int
    Output: Efficient Frontier --> DataFrame
    """
    # Generate random weights for portfolios
    weights = np.random.random(size=(num_points, len(stock_returns)))
    weights = weights / weights.sum(axis=1, keepdims=True)
    # Calculate portfolio metrics for each set of weights
    metrics = np.array([
        calculate_portfolio_metrics(weight, stock_returns, covariance_matrix)
        for weight in weights
    ])
    # Create DataFrame for efficient frontier
    efficient_frontier = pd.DataFrame({
        'Return': metrics[:, 0],
        'Variance': metrics[:, 1]
    })
    return efficient_frontier

def main(stock_data_dir):
    # Get stock data
    stock_data = get_data(stock_data_dir)
    # Calculate stock returns
    stock_returns = calculate_returns(stock_data)
    # Calculate covariance matrix
    covariance_matrix = calculate_covariance_matrix(stock_returns)
    # Calculate efficient frontier
    efficient_frontier = calculate_efficient_frontier(stock_returns, covariance_matrix)
    return efficient_frontier

# ============ Main ============ #
if __name__ == '__main__':
    stock_data_dir = '/Users/simon/Financial-Trading-Algorithm/Data/Stock-Data'  # Path to the Stock Data Directory
    efficient_frontier = main(stock_data_dir)
    print(efficient_frontier.head())