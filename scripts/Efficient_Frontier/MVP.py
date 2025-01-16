# Standard library imports
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import (
    EfficientFrontier,
    expected_returns,
    risk_models,
    plotting,
    objective_functions
)

# Configure path for local imports
PROJECT_ROOT = "/Users/simon/Financial-Trading-Algorithm"
sys.path.append(os.path.join(PROJECT_ROOT, "Data/Data-Management"))

try:
    from main import update_data
except ImportError as e:
    logging.error(f"Failed to import update_data: {e}")
    raise


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='efficient_frontier.log'
)

CHECKMARK = "✓"
XMARK = "✗"

class PortfolioOptimizer:
    def __init__(self, data_path: str, lookback_period: int = 252):
        self.data_path = Path(data_path)
        self.lookback_period = lookback_period
        self.min_stocks = 1  # Modified for testing with single stock
        self.prices_data = None
        self.tickers = []
        self.test_number = 1
        self.returns = None
        self.volatility = None
        self.sharpe_ratios = None
        self.test_results = {}  # Store all test results
        
    def _test_output(self, condition: bool, test_name: str, error_details: dict = None) -> None:
        """Helper method to print test results"""
        if condition:
            print(f"Test {self.test_number} - {test_name}: Completed Successfully {CHECKMARK}")
        else:
            print(f"Test {self.test_number} - {test_name}: Failed {XMARK}")
            if error_details:
                print("Error Details:")
                for key, value in error_details.items():
                    print(f"  {key}: {value}")
        self.test_number += 1
        
    def _run_validation_tests(self, data: pd.DataFrame, test_name: str) -> Dict:
        """Run standard validation tests on DataFrame"""
        tests = {
            "No Infinity Values": ~np.isinf(data).any().any(),
            "No Zero Values": (data != 0).any().any(),
            "Positive Values": (data >= 0).all().all(),
            "Not Empty": len(data) > 0,
            "Has Columns": len(data.columns) > 0,
            "No Duplicate Indices": ~data.index.duplicated().any(),
            "Index is Sorted": data.index.is_monotonic_increasing,
            "Data Types Consistent": all(dtype.kind in 'fiu' for dtype in data.dtypes)
        }
        
        error_details = {
            "Shape": data.shape,
            "Data Types": data.dtypes.to_dict(),
            "Missing Values": data.isnull().sum().to_dict(),
            "Infinity Count": np.isinf(data).sum().sum(),
            "Zero Count": (data == 0).sum().sum(),
            "Index Range": f"{data.index.min()} to {data.index.max()}"
        }
        
        all_tests_passed = all(tests.values())
        self._test_output(all_tests_passed, test_name, 
                         error_details if not all_tests_passed else None)
        
        self.test_results[test_name] = tests
        return tests

    def _test_optimization_inputs(self, mu: pd.Series, S: pd.DataFrame) -> Dict:
        """Test optimization inputs"""
        tests = {
            "Mu Not Empty": len(mu) > 0,
            "Covariance Matrix Symmetric": np.allclose(S, S.T),
            "Covariance Matrix PSD": np.all(np.linalg.eigvals(S) > -1e-10),
            "Matching Dimensions": len(mu) == S.shape[0] == S.shape[1],
            "No Missing Values": not (pd.isna(mu).any() or pd.isna(S).any().any())
        }
        
        error_details = {
            "Mu Shape": mu.shape,
            "S Shape": S.shape,
            "Mu Range": f"{mu.min():.4f} to {mu.max():.4f}",
            "S Condition Number": np.linalg.cond(S),
            "S Eigenvalues Range": f"{np.linalg.eigvals(S).min():.4f} to {np.linalg.eigvals(S).max():.4f}"
        }
        
        all_tests_passed = all(tests.values())
        self._test_output(all_tests_passed, "Optimization Inputs", 
                         error_details if not all_tests_passed else None)
        return tests

    def _test_portfolio_weights(self, weights: Dict) -> Dict:
        """Test portfolio weights"""
        weight_values = np.array(list(weights.values()))
        tests = {
            "Weights Sum to 1": np.isclose(sum(weight_values), 1.0, atol=1e-3),
            "No Negative Weights": all(w >= -1e-10 for w in weight_values),
            "No Excessive Weights": all(w <= 1 + 1e-10 for w in weight_values),
            "All Stocks Present": len(weights) == len(self.tickers)
        }
        
        error_details = {
            "Weight Sum": sum(weight_values),
            "Min Weight": min(weight_values),
            "Max Weight": max(weight_values),
            "Non-zero Positions": sum(abs(w) > 1e-5 for w in weight_values)
        }
        
        all_tests_passed = all(tests.values())
        self._test_output(all_tests_passed, "Portfolio Weights", 
                         error_details if not all_tests_passed else None)
        return tests

    def _test_portfolio_metrics(self, portfolios: List[Dict]) -> Dict:
        """Test portfolio performance metrics"""
        returns = [p['return'] for p in portfolios]
        volatilities = [p['volatility'] for p in portfolios]
        sharpes = [p['sharpe'] for p in portfolios]
        
        tests = {
            "Returns in Range": min(returns) > -1 and max(returns) < 1,
            "Positive Volatilities": all(v > 0 for v in volatilities),
            "Reasonable Sharpes": all(-10 < s < 10 for s in sharpes),
            "Increasing Returns": np.corrcoef(range(len(returns)), returns)[0,1] > 0
        }
        
        error_details = {
            "Return Range": f"{min(returns):.4f} to {max(returns):.4f}",
            "Volatility Range": f"{min(volatilities):.4f} to {max(volatilities):.4f}",
            "Sharpe Ratio Range": f"{min(sharpes):.4f} to {max(sharpes):.4f}",
            "Number of Portfolios": len(portfolios)
        }
        
        all_tests_passed = all(tests.values())
        self._test_output(all_tests_passed, "Portfolio Metrics", 
                         error_details if not all_tests_passed else None)
        return tests
        
    def load_and_prepare_data(self) -> None:
        """Load and prepare data from CSV files with enhanced filtering"""
        try:
            logging.info("Starting data loading process")
            all_data = {}
            
            files = list(self.data_path.glob('*.csv'))
            if not files:
                raise ValueError(f"No CSV files found in {self.data_path}")
            
            for file in files:
                try:
                    ticker = file.stem
                    logging.info(f"Loading data for {ticker}")
                    df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
                    if len(df) >= self.lookback_period:
                        all_data[ticker] = df['Close']
                        logging.info(f"Successfully loaded {ticker} with {len(df)} data points")
                except Exception as e:
                    logging.warning(f"Error loading {file}: {str(e)}")
                    continue
            
            if len(all_data) < self.min_stocks:
                raise ValueError(f"Insufficient data: only found {len(all_data)} valid stocks")
            
            logging.info(f"Loaded {len(all_data)} stocks")
            
            # Create DataFrame and handle missing data
            self.prices_data = pd.DataFrame(all_data)
            logging.info(f"Initial prices_data shape: {self.prices_data.shape}")
            
            # Basic cleaning
            self.prices_data = self.prices_data.dropna(axis=1, how='all')
            self.prices_data = self.prices_data.ffill().bfill()
            
            # Light filtering
            self.prices_data = self.prices_data.loc[:, (self.prices_data != 0).any(axis=0)]
            self.prices_data = self.prices_data.loc[:, (self.prices_data.pct_change() != 0).any()]
             
            logging.info(f"Final prices_data shape: {self.prices_data.shape}")
            
            self.tickers = list(self.prices_data.columns)
            logging.info(f"Final tickers: {self.tickers}")
            
            # Calculate metrics
            self.calculate_metrics()
            
        except Exception as e:
            logging.error(f"Data preparation failed: {str(e)}")
            raise
        
        # Validation tests
        test_conditions = {
            "Data Loading": len(self.prices_data) > 0,
            "Sufficient Tickers": len(self.tickers) >= self.min_stocks,
            "No Missing Values": not self.prices_data.isnull().any().any(),
            "Sufficient History": len(self.prices_data) >= self.lookback_period
        }
        
        error_details = {
            "Tickers Loaded": len(self.tickers),
            "Data Points": len(self.prices_data),
            "Missing Values": self.prices_data.isnull().sum().sum()
        }
        
        all_tests_passed = all(test_conditions.values())
        self._test_output(all_tests_passed, "Data Preparation", 
                         error_details if not all_tests_passed else None)

    def filter_stocks(self, correlation_threshold: float = 0.8, liquidity_threshold: float = 100000):
        """Filter out highly correlated and low liquidity stocks"""
        # Run pre-filtering tests
        self._run_validation_tests(self.prices_data, "Pre-Filter Data")
        
        # Calculate correlation matrix
        corr_matrix = self.prices_data.pct_change().corr()
        
        # Filter highly correlated stocks
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    stock_i, stock_j = corr_matrix.columns[i], corr_matrix.columns[j]
                    to_drop.add(stock_j if self.get_sharpe_ratio(stock_i) > self.get_sharpe_ratio(stock_j) else stock_i)
        
        # Filter low liquidity stocks
        avg_volume = self.prices_data.mean()
        low_liquidity = avg_volume[avg_volume < liquidity_threshold].index
        to_drop.update(low_liquidity)
        
        # Remove filtered stocks
        self.prices_data = self.prices_data.drop(columns=list(to_drop))
        self.tickers = list(self.prices_data.columns)
        logging.info(f"Filtered to {len(self.tickers)} stocks")
        
        # Run post-filtering tests
        filtered_tests = self._run_validation_tests(self.prices_data, "Post-Filter Data")
        
        # Additional filtering tests
        filter_tests = {
            "Reduced Correlations": self.prices_data.pct_change().corr().max().max() <= correlation_threshold + 1e-10,
            "Sufficient Stocks Remaining": len(self.tickers) >= self.min_stocks,
            "No Low Liquidity": self.prices_data.mean().min() >= liquidity_threshold
        }
        
        error_details = {
            "Original Stocks": len(self.tickers),
            "Filtered Stocks": len(self.prices_data.columns),
            "Max Correlation": self.prices_data.pct_change().corr().max().max(),
            "Min Liquidity": self.prices_data.mean().min()
        }
        
        self._test_output(all(filter_tests.values()), "Stock Filtering", 
                         error_details if not all(filter_tests.values()) else None)

    def get_sharpe_ratio(self, stock):
        """Calculate Sharpe ratio for a single stock"""
        returns = self.prices_data[stock].pct_change().dropna()
        return returns.mean() / returns.std()

    def calculate_metrics(self):
        """Calculate key metrics for all stocks"""
        self.returns = self.prices_data.pct_change().dropna()
        self.volatility = self.returns.std()
        self.sharpe_ratios = self.returns.mean() / self.volatility

    def select_least_correlated_stocks(self, num_stocks: int = 10) -> List[str]:
        """Select stocks with lowest correlations to each other"""
        logging.info(f"Selecting {num_stocks} least correlated stocks from {len(self.tickers)} total stocks")
        
        # Calculate returns and correlation matrix
        returns = self.prices_data.pct_change().dropna()
        corr_matrix = returns.corr().abs()  # Use absolute correlations
        
        # Start with the stock that has lowest average correlation
        avg_corr = corr_matrix.mean()
        selected_stocks = [avg_corr.idxmin()]
        
        while len(selected_stocks) < num_stocks:
            remaining_stocks = set(self.tickers) - set(selected_stocks)
            if not remaining_stocks:
                break
            
            # Find stock with minimum maximum correlation to already selected stocks
            correlations = corr_matrix.loc[list(remaining_stocks), selected_stocks]
            max_correlations = correlations.max(axis=1)
            least_correlated = max_correlations.idxmin()
            
            selected_stocks.append(least_correlated)
            logging.info(f"Selected stock {len(selected_stocks)}: {least_correlated}")
        
        # Log correlation statistics
        selected_corr = corr_matrix.loc[selected_stocks, selected_stocks]
        logging.info(f"Average correlation among selected stocks: {selected_corr.mean().mean():.4f}")
        logging.info(f"Maximum correlation among selected stocks: {selected_corr.max().max():.4f}")
        
        return selected_stocks

    def generate_efficient_frontier(self, num_portfolios: int = 100) -> Dict:
        """Generate efficient frontier with correlation-based stock selection"""
        try:
            # Select least correlated stocks if we have too many
            if len(self.tickers) > 20:  # Adjust threshold as needed
                selected_tickers = self.select_least_correlated_stocks(num_stocks=20)
                self.prices_data = self.prices_data[selected_tickers]
                self.tickers = selected_tickers
                logging.info(f"Reduced to {len(self.tickers)} least correlated stocks")
            
            # Calculate returns and other metrics
            returns = self.prices_data.pct_change().dropna()
            mu = expected_returns.mean_historical_return(self.prices_data)
            S = risk_models.CovarianceShrinkage(self.prices_data).ledoit_wolf()
            
            # Enhanced validation checks
            assert len(mu) == S.shape[0] == S.shape[1], f"Dimension mismatch: mu: {len(mu)}, S: {S.shape}"
            assert not np.isnan(mu).any() and not np.isnan(S).any(), "NaN values detected in input data"
            assert np.all(np.linalg.eigvals(S) > 0), "Covariance matrix is not positive definite"
            
            # Find feasible volatility range
            ef_min = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
            min_vol_weights = ef_min.min_volatility()
            min_ret, min_vol, _ = ef_min.portfolio_performance()
            
            ef_max = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
            max_ret_weights = ef_max.maximize_sharpe()
            max_ret, max_vol, _ = ef_max.portfolio_performance()
            
            # Generate portfolios
            target_volatilities = np.linspace(min_vol * 1.001, max_vol * 0.999, num_portfolios)
            portfolios = []
            
            for i, target_vol in enumerate(target_volatilities):
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
                    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                    weights = ef.efficient_risk(target_volatility=float(target_vol))
                    cleaned_weights = ef.clean_weights(cutoff=1e-4)
                    ret, vol, sharpe = ef.portfolio_performance()
                    
                    portfolios.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe': sharpe,
                        'weights': cleaned_weights
                    })
                    
                except Exception as e:
                    logging.warning(f"Failed to optimize portfolio {i+1}: {str(e)}")
                    continue
            
            if not portfolios:
                raise ValueError("No valid portfolios generated")
            
            # Create plot
            fig = self._create_plot(portfolios, min_vol, max_vol)
            
            results = {
                'portfolios': portfolios,
                'tickers': self.tickers,
                'plot': fig,
                'optimization_parameters': {
                    'min_volatility': min_vol,
                    'max_volatility': max_vol,
                    'num_assets': len(mu),
                    'correlation_stats': {
                        'mean': returns.corr().mean().mean(),
                        'max': returns.corr().max().max()
                    }
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Efficient frontier generation failed: {str(e)}")
            raise

    def _create_plot(self, portfolios: List[Dict], min_vol: float, max_vol: float) -> plt.Figure:
        """Create efficient frontier plot with enhanced validation"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            vols = [p['volatility'] for p in portfolios]
            rets = [p['return'] for p in portfolios]
            
            # Sort points for better curve visualization
            sort_idx = np.argsort(vols)
            vols = np.array(vols)[sort_idx]
            rets = np.array(rets)[sort_idx]
            
            # Plot frontier curve and points
            ax.plot(vols, rets, 'b-', label='Efficient Frontier')
            ax.scatter(vols, rets, c='red', s=50, alpha=0.6, label='Portfolios')
            
            # Add volatility bounds
            ax.axvline(min_vol, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(max_vol, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_title('Efficient Frontier')
            ax.set_xlabel('Expected Volatility')
            ax.set_ylabel('Expected Return')
            ax.legend()
            ax.grid(True)
            
            return fig
        except Exception as e:
            logging.error(f"Plot creation failed: {str(e)}")
            return None

    def create_efficient_frontier_plot(self, portfolios: List[Dict]) -> plt.Figure:
        """Create efficient frontier plot as fallback if optimization plot fails"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            returns = [p['return'] for p in portfolios]
            volatilities = [p['volatility'] for p in portfolios]
            ax.scatter(volatilities, returns, c='b', s=10)
            ax.set_title('Efficient Frontier')
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Expected Return')
            logging.info("Created fallback efficient frontier plot")
            return fig
        except Exception as e:
            logging.error(f"Failed to create fallback plot: {str(e)}")
            return None

    def plot_efficient_frontier(self, results: Dict) -> None:
        """Plot efficient frontier with enhanced error handling"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = "/Users/simon/Financial-Trading-Algorithm/Data/Info/Logs/Efficient-frontiers"
            os.makedirs(path, exist_ok=True)
            
            plot_file = f'{path}/{timestamp}.png'
            
            # Enhanced validation
            validation = {
                "Has Results": results is not None,
                "Has Portfolios": len(results.get('portfolios', [])) > 0,
                "Has Plot Object": 'plot' in results,
                "Valid Plot": results.get('plot') is not None and isinstance(results.get('plot'), plt.Figure),
                "Has Optimization Data": 'optimization_parameters' in results,
                "Sufficient Success Rate": results.get('optimization_success_rate', 0) > 0.5
            }
            
            error_details = {
                "Results Keys": list(results.keys()),
                "Portfolio Count": len(results.get('portfolios', [])),
                "Plot Present": 'plot' in results,
                "Plot Type": type(results.get('plot')).__name__ if 'plot' in results else None,
                "Success Rate": f"{results.get('optimization_success_rate', 0):.2%}",
                "Optimization Parameters": results.get('optimization_parameters', {})
            }

            if 'plot' in results:
                plt.figure(results['plot'].number)
                plt.savefig(plot_file)
                plt.close()
                validation["Plot Generated"] = os.path.exists(plot_file)
                validation["Plot Size Valid"] = os.path.getsize(plot_file) > 1000  # At least 1KB
            
            all_tests_passed = all(validation.values())
            self._test_output(all_tests_passed, "Plotting", 
                            error_details if not all_tests_passed else None)
            
        except Exception as e:
            error_details = {
                "Error Type": type(e).__name__,
                "Error Message": str(e),
                "Error Location": f"{type(e).__name__} at line {sys.exc_info()[2].tb_lineno}",
                "Results Keys": list(results.keys()) if results else "No results",
                "Path": path,
                "Timestamp": timestamp,
                "Python Version": sys.version,
                "Matplotlib Version": plt.__version__
            }
            self._test_output(False, "Plotting", error_details)
            logging.error(f"Failed to plot efficient frontier: {str(e)}")
            raise

    def _check_disk_space(self, path: str) -> bool:
        """Check if there's sufficient disk space (at least 10MB)"""
        try:
            free_space = self._get_available_space(path)
            return free_space > 10 * 1024 * 1024  # 10MB minimum
        except Exception:
            return True  # Default to True if check fails
            
    def _get_available_space(self, path: str) -> int:
        """Get available space in bytes"""
        try:
            st = os.statvfs(path)
            return st.f_frsize * st.f_bavail
        except Exception:
            return float('inf')  # Return infinite space if check fails

    def save_results(self, results: Dict) -> None:
        """Save optimization results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'efficient_frontier_results_{timestamp}.csv'
        
        data = []
        for p in results['portfolios']:
            row = {
                'return': p['return'],
                'volatility': p['volatility']
            }
            for ticker, weight in zip(results['tickers'], p['weights']):
                row[ticker] = weight
            data.append(row)
        
        pd.DataFrame(data).to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

def main():
    data_path = "/Users/simon/Financial-Trading-Algorithm/Data/Stock-Data"
    update_data()
    optimizer = PortfolioOptimizer(data_path=data_path)
    try:
        optimizer.load_and_prepare_data()
        
        # Modified condition for single stock scenario
        if len(optimizer.tickers) > 1:
            #if len(optimizer.tickers) > 500:
             #   results = optimizer.optimize_in_chunks()
            #else:
                #results = optimizer.generate_efficient_frontier()
            results = optimizer.generate_efficient_frontier()
            
            optimizer.plot_efficient_frontier(results)
            optimizer.save_results(results)
        else:
            logging.info("Single stock detected - skipping optimization")
            print("\nSingle stock analysis completed successfully! 🎉")
            return
            
        print("\nAll processes completed successfully! 🎉")
    except Exception as e:
        print(f"\nProcess failed with error: {str(e)} {XMARK}")
        raise

if __name__ == "__main__":
    main()
