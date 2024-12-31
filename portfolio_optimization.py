import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import pandas as pd

class PortfolioOptimizer:
    def portfolio_stats(self, weights, returns, cov_matrix):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return portfolio_return, portfolio_std

    def optimize_portfolio(self, returns, cov_matrix, target_return):
        num_assets = len(returns.columns)
        args = (returns, cov_matrix)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_stats(x, returns, cov_matrix)[0] - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        initial_weights = np.array([1.0/num_assets] * num_assets)
        result = minimize(
            lambda x: self.portfolio_stats(x, returns, cov_matrix)[1],
            initial_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x

    def generate_efficient_frontier(self, returns, cov_matrix, target_returns):
        return [self.optimize_portfolio(returns, cov_matrix, target) 
                for target in target_returns]

    def get_optimal_portfolio(self, portfolios, risk_free_rate=0.02):
        
        # Calculate Sharpe ratio for each portfolio
        sharpe_ratios = []

        # Calculate Sharpe ratio for each portfolio
        for weights in portfolios:
            returns, std = self.portfolio_stats(weights, self.returns, self.cov_matrix)
            sharpe_ratio = (returns - risk_free_rate) / std
            sharpe_ratios.append(sharpe_ratio)
        
        # Find portfolio with maximum Sharpe ratio
        optimal_idx = np.argmax(sharpe_ratios)
        return portfolios[optimal_idx]

#class TradingExecutor:
#    def __init__(self):
#        self.current_positions = {}  # Track current portfolio positions
#
#    def rebalance_portfolio(self, target_weights):
#        # Implement portfolio rebalancing logic
#        # Calculate required trades to achieve target weights
#        # Execute trades using your broker's API
#        pass
#
#    def execute_trade(self, symbol, quantity, side):
#        # Implement actual trade execution
#        # Connect to your broker's API here
#        pass
#
#    def set_stop_loss(self, symbol, price):
#        # Implement stop loss orders
#        pass
#
#    def monitor_performance(self):
#        # Implement performance monitoring
#        pass



class DataManager:
    def __init__(self):
        self.tickers = self._get_sp500_tickers()

    def _get_sp500_tickers(self):
        # Get S&P 500 tickers using pandas_datareader
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = table['Symbol'].tolist()
            # Clean tickers to ensure compatibility with yfinance
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            # Return a small subset of major S&P 500 stocks as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    def get_stock_data(self, period="20y", interval="1mo"):
        data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(period=period, interval=interval)['Adj Close']
        return pd.DataFrame(data)

    def calculate_returns(self, prices):
        return prices.pct_change().dropna()
    
def plot_efficient_frontier(self, returns, cov_matrix, risk_free_rate=0.02):
    import matplotlib.pyplot as plt
    
    # Generate range of target returns
    min_return = returns.mean().min() * 252
    max_return = returns.mean().max() * 252
    target_returns = np.linspace(min_return, max_return, 100)
    
    # Generate efficient frontier portfolios
    portfolios = self.generate_efficient_frontier(returns, cov_matrix, target_returns)
    
    # Calculate returns and risks for all portfolios
    returns_array = []
    risks_array = []
    for weights in portfolios:
        ret, risk = self.portfolio_stats(weights, returns, cov_matrix)
        returns_array.append(ret)
        risks_array.append(risk)
        
    # Find optimal portfolio
    optimal_portfolio = self.get_optimal_portfolio(portfolios, risk_free_rate)
    opt_return, opt_risk = self.portfolio_stats(optimal_portfolio, returns, cov_matrix)
    
    # Plot efficient frontier
    plt.figure(figsize=(10, 6))
    plt.plot(risks_array, returns_array, 'b-', label='Efficient Frontier')
    plt.scatter(opt_risk, opt_return, color='red', marker='*', s=200, label='Optimal Portfolio')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Initialize components
    data_manager = DataManager()
    optimizer = PortfolioOptimizer()
    #executor = TradingExecutor()

    # Get stock data
    stock_data = data_manager.get_stock_data()
    returns = data_manager.calculate_returns(stock_data)
    cov_matrix = returns.cov()

    # Generate efficient frontier
    target_returns = np.linspace(returns.mean().min(), returns.mean().max(), 100)
    efficient_portfolios = optimizer.generate_efficient_frontier(returns, cov_matrix, target_returns)

    # Execute trades based on optimal portfolio
    optimal_weights = optimizer.get_optimal_portfolio(efficient_portfolios)
    #executor.rebalance_portfolio(optimal_weights)
    
    plot_efficient_frontier(returns, cov_matrix)