import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, data_path, lookback_period=252):
        self.data_path = data_path
        self.lookback_period = lookback_period
        self.prices_data = None
        self.tickers = []

    def load_and_prepare_data(self):
        try:
            self.prices_data = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
            self.prices_data = self.prices_data.astype(float)
            self.tickers = list(self.prices_data.columns)
            logger.info(f"Loaded {len(self.tickers)} stocks")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def filter_similar_stocks(self, n_clusters=50):
        returns = self.prices_data.pct_change().dropna()
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_returns.T)
        
        representative_stocks = []
        for i in range(n_clusters):
            cluster_stocks = np.array(self.tickers)[clusters == i]
            if len(cluster_stocks) > 0:
                cluster_returns = returns[cluster_stocks]
                sharpe_ratios = cluster_returns.mean() / cluster_returns.std()
                representative_stocks.append(sharpe_ratios.idxmax())
        
        self.prices_data = self.prices_data[representative_stocks]
        self.tickers = representative_stocks
        logger.info(f"Filtered to {len(self.tickers)} representative stocks")

    def optimize_portfolio(self):
        mu = expected_returns.mean_historical_return(self.prices_data)
        S = risk_models.sample_cov(self.prices_data)
        
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        return cleaned_weights

    def run_optimization(self):
        self.load_and_prepare_data()
        self.filter_similar_stocks()
        optimized_weights = self.optimize_portfolio()
        
        logger.info("Optimization completed successfully")
        return optimized_weights

if __name__ == "__main__":
    data_path = "stock_data.csv"  # Replace with your actual data path
    optimizer = PortfolioOptimizer(data_path)
    
    try:
        optimized_weights = optimizer.run_optimization()
        print("Optimized Portfolio Weights:")
        for ticker, weight in optimized_weights.items():
            print(f"{ticker}: {weight:.4f}")
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
