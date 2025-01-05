import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt import black_litterman
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import gc  # Add garbage collection
from sklearn.preprocessing import RobustScaler
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist

# Define standalone functions for multiprocessing compatibility
def apply_risk_model(model_name: str, data: pd.DataFrame, gamma: float = 0.1) -> np.ndarray:
    """Apply risk model with L2 regularization built in"""
    S = None
    if (model_name == 'sample'):
        S = risk_models.sample_cov(data)
    elif (model_name == 'ledoit_wolf'):
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
    elif (model_name == 'semi_covariance'):
        S = risk_models.semicovariance(data)
    elif (model_name == 'exp_cov'):
        S = risk_models.exp_cov(data)
    else:
        raise ValueError(f"Unknown risk model: {model_name}")
    
    # Add L2 regularization directly to covariance matrix
    n_assets = len(data.columns)
    S += gamma * np.eye(n_assets)
    return S

def apply_return_model(model_name: str, data: pd.DataFrame) -> pd.Series:
    """Apply return model with additional error checking"""
    try:
        if (model_name == 'mean'):
            return expected_returns.mean_historical_return(data)
        elif (model_name == 'ema'):
            # Add scaling to prevent numerical issues
            returns = expected_returns.ema_historical_return(data, span=200)
            return returns * 252  # Annualize returns
    except Exception as e:
        logging.error(f"Return model {model_name} failed: {str(e)}")
        raise
    raise ValueError(f"Unknown return model: {model_name}")

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate and clean returns data"""
    # Calculate returns using pct_change
    returns = data.pct_change()
    
    # Remove outliers (values beyond 3 standard deviations)
    returns_std = returns.std()
    returns_mean = returns.mean()
    returns = returns.clip(
        lower=returns_mean - 3*returns_std,
        upper=returns_mean + 3*returns_std,
    )
    
    # Handle missing values
    returns = returns.fillna(0)  # Replace NaN with 0
    
    # Ensure no infinite values
    returns = returns.replace([np.inf, -np.inf], 0)
    
    return returns

def auto_adjust_target_volatility(data: pd.DataFrame, initial_target: float = 0.2, 
                                max_attempts: int = 300, increment: float = 0.05) -> float:
    """Dynamically adjust target volatility until optimization succeeds"""
    target_vol = initial_target
    n = 0
    for attempt in range(max_attempts):
        n += 1
        try:
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
            ef.efficient_risk(target_volatility=target_vol)
            del ef
            return target_vol
        except ValueError:
            target_vol += increment
            target_vol *= n 
            logging.debug(f"Increased target volatility to {target_vol:.2f} (attempt {attempt + 1})")
    
    raise ValueError(f"Could not find suitable target volatility after {max_attempts} attempts")

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data by filling missing values and removing infinite values"""
    data = data.fillna(method='ffill').fillna(method='bfill')
    returns = np.log(data / data.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0)
    return returns

def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate logarithmic returns with better error handling"""
    epsilon = 1e-8
    return np.log((data + epsilon) / (data.shift(1) + epsilon)).fillna(0)

def ensure_positive_semidefinite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Ensure covariance matrix is positive semidefinite"""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def validate_optimization_inputs(mu: np.ndarray, S: np.ndarray) -> bool:
    """Validate optimization inputs for numerical stability"""
    if np.isnan(mu).any() or np.isnan(S).any():
        logging.warning("NaN values detected in optimization inputs")
        return False
    if np.isinf(mu).any() or np.isinf(S).any():
        logging.warning("Infinite values detected in optimization inputs")
        return False
    return True

def get_quasi_diag(link):
    """Extract quasi-diagonal from linkage matrix with correct length"""
    n = len(link)
    return pd.Series(link[:(n-1), 2], index=[link[i, 0] for i in range(n-1)])

def get_recursive_bisection(cov, sort_ix):
    """Perform recursive bisection for HRP"""
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_left, c_right = c_items[i], c_items[i+1]
            c_left_var = get_cluster_var(cov, c_left)
            c_right_var = get_cluster_var(cov, c_right)
            alpha = 1 - c_left_var / (c_left_var + c_right_var)
            w[c_left] *= alpha
            w[c_right] *= 1 - alpha
    return w

def get_cluster_var(cov, c_items):
    """Calculate cluster variance"""
    return np.sqrt(np.sum(cov.loc[c_items, c_items].values))

def hrp(returns: pd.DataFrame) -> pd.Series:
    """Calculate HRP portfolio weights with proper index alignment"""
    cov = returns.cov()
    dist = pd.DataFrame(squareform(pdist(returns.T)), columns=returns.columns, index=returns.columns)
    link = linkage(squareform(dist), 'single')
    sort_ix = get_quasi_diag(link)
    sort_ix = returns.columns[sort_ix].tolist()  # Convert to column names
    weights = pd.Series(1, index=sort_ix)
    clustered_alphas = get_recursive_bisection(cov, sort_ix)
    return weights * clustered_alphas

def scale_data(data: pd.DataFrame, lower_percentile=1, upper_percentile=99) -> pd.DataFrame:
    """Scale data with empty data handling"""
    if data.empty:
        return data
    if data.size == 0:
        return data
        
    lower = np.percentile(data.values[~np.isnan(data.values)], lower_percentile)
    upper = np.percentile(data.values[~np.isnan(data.values)], upper_percentile)
    return data.clip(lower=lower, upper=upper)

class EnhancedPortfolioOptimizer:
    def __init__(self, data_path: str, lookback_period: int = 252):
        self.data_path = Path(data_path)
        self.lookback_period = lookback_period
        self.n_bootstraps = 100  # Reduced from 1000
        self.n_monte_carlo = 10000  # Reduced from 100000
        self.risk_model_names = ['sample', 'ledoit_wolf', 'semi_covariance', 'exp_cov']
        self.return_model_names = ['mean', 'ema']
        self.solver_options = {
            'solver': 'ECOS',  # Changed from SCS to ECOS for better stability
            'verbose': False,
            'max_iters': 5000,
            'eps': 1e-8
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data from CSV files with improved cleaning"""
        all_data = {}  # TO make this easier
        try:
            # Use Path.glob for file iteration  # TO make this easier
            for file in self.data_path.glob('*.csv'):  # TO make this easier
                try:
                    ticker = file.stem  # Extract filename without extension  # TO make this easier
                    df = pd.read_csv(file, parse_dates=['Date'])  # Parse dates automatically  # TO make this easier
                    df.set_index('Date', inplace=True)  # Set Date as index  # TO make this easier
                    # Only include files with Close column and sufficient data  # TO make this easier
                    if 'Close' in df.columns and len(df) >= self.lookback_period:  # TO make this easier
                        all_data[ticker] = df['Close']  # Store just the Close prices  # TO make this easier
                except Exception as e:
                    logging.warning(f"Error loading {file}: {str(e)}")  # Log individual file errors  # TO make this easier
                    continue
            
            if not all_data:  # Check if any data was loaded  # TO make this easier
                raise ValueError("No valid data files found")  # TO make this easier
                
            # Combine all data and handle missing values  # TO make this easier
            combined_data = pd.DataFrame(all_data).ffill().bfill()  # Forward and backward fill  # TO make this easier
            return clean_data(combined_data)  # Clean the data using the helper function  # TO make this easier
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")  # Log any overall errors  # TO make this easier
            raise

    def bootstrap_returns(self, data: pd.DataFrame) -> np.ndarray:
        """Generate bootstrapped returns with improved handling"""
        try:
            # Remove any NaN values before bootstrapping
            clean_data = data.dropna(how='any')
            if len(clean_data) < self.lookback_period:
                raise ValueError("Insufficient data after cleaning")
                
            bootstrap_samples = []
            for _ in range(self.n_bootstraps):
                sample = clean_data.sample(n=len(clean_data), replace=True)
                if not sample.isnull().any().any():  # Verify no NaNs
                    bootstrap_samples.append(sample)
            
            return np.array(bootstrap_samples)
        except Exception as e:
            logging.error(f"Bootstrap failed: {str(e)}")
            return np.array([])

    def monte_carlo_portfolios(self, mu: pd.Series, S: pd.DataFrame) -> List[Dict]:
        """Generate large number of random portfolios"""
        n_assets = len(mu)
        portfolios = []
        
        for _ in range(self.n_monte_carlo):
            weights = np.random.dirichlet(np.ones(n_assets))
            portfolio_return = np.sum(weights * mu)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
            portfolios.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'weights': weights
            })
        
        return portfolios

    def _single_optimization(self, data: pd.DataFrame, return_model: str, risk_model: str) -> Dict:
        """Single optimization run with improved error handling and numerical stability"""
        try:
            logging.info(f"Starting optimization with {return_model}_{risk_model}")
            
            # Scale inputs for numerical stability
            scaler = RobustScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Calculate log returns for better numerical stability
            log_returns = calculate_log_returns(scaled_data)
            
            mu = apply_return_model(return_model, log_returns)
            S = apply_risk_model(risk_model, log_returns, gamma=0.1)
            
            if not validate_optimization_inputs(mu, S):
                return None
            
            # Ensure positive semidefinite covariance matrix
            S = ensure_positive_semidefinite(S)
            
            # Create efficient frontier with ECOS solver
            ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1), solver='ECOS')
            
            # Optimize without passing solver parameter
            ef.max_sharpe(risk_free_rate=0.02)
            weights = ef.clean_weights(cutoff=1e-4)
            performance = ef.portfolio_performance(risk_free_rate=0.02)
            
            # Clear memory
            del ef
            gc.collect()
            
            logging.info(f"Completed optimization for {return_model}_{risk_model}")
            return {
                'model': f"{return_model}_{risk_model}",
                'weights': weights,
                'performance': performance,
                'mu': mu,
                'cov': S
            }
        except Exception as e:
            logging.error(f"Optimization failed for {return_model}_{risk_model}: {str(e)}")
            return None
        finally:
            gc.collect()

    def optimize_portfolio(self, data: pd.DataFrame) -> Dict:
        """Enhanced portfolio optimization with multiple models including HRP"""
        results = []
        
        # Clean the data first
        cleaned_returns = clean_data(data)
        
        # Verify data is clean
        if cleaned_returns.isnull().any().any() or np.isinf(cleaned_returns).any().any():
            raise ValueError("Data still contains NaN or infinite values after cleaning")
        
        total_combinations = len(self.return_model_names) * len(self.risk_model_names)
        
        for i, return_model in enumerate(self.return_model_names):
            for j, risk_model in enumerate(self.risk_model_names):
                current = i * len(self.risk_model_names) + j + 1
                logging.info(f"Processing combination {current}/{total_combinations}")
                
                result = self._single_optimization(data, return_model, risk_model)
                if result is not None:
                    results.append(result)
                
                # Clear memory after each optimization
                gc.collect()
        
        # Add HRP optimization
        hrp_result = self.optimize_portfolio_hrp(cleaned_returns)
        if hrp_result is not None:
            results.append(hrp_result)
        
        if not results:
            raise ValueError("No successful optimizations")
            
        combined_results = self._combine_results(results)
        uncertainty = self._calculate_uncertainty(cleaned_returns, combined_results)
        
        return {
            'optimal_portfolios': combined_results,
            'uncertainty_estimates': uncertainty,
            'individual_results': results
        }

    def _combine_results(self, results: List[Dict]) -> Dict:
        """Combine results from different models using ensemble approach"""
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            raise ValueError("No valid optimization results")
            
        all_weights = np.array([list(r['weights'].values()) for r in valid_results])
        all_returns = np.array([r['performance'][0] for r in valid_results])
        all_risks = np.array([r['performance'][1] for r in valid_results])
        
        return {
            'mean_weights': np.mean(all_weights, axis=0),
            'mean_return': np.mean(all_returns),
            'mean_risk': np.mean(all_risks),
            'mean_cov': np.cov(all_weights.T),  # Add covariance matrix
            'weight_uncertainty': np.std(all_weights, axis=0),
            'return_uncertainty': np.std(all_returns),
            'risk_uncertainty': np.std(all_risks)
        }

    def _calculate_uncertainty(self, data: pd.DataFrame, results: Dict) -> Dict:
        """Calculate uncertainty estimates with dynamic volatility targeting"""
        try:
            bootstrap_samples = self.bootstrap_returns(data)
            bootstrap_results = []
            
            for i, sample in enumerate(bootstrap_samples):
                try:
                    if i % 10 == 0:
                        logging.info(f"Processing bootstrap sample {i}/{len(bootstrap_samples)}")
                        gc.collect()
                    
                    sample_df = pd.DataFrame(sample, columns=data.columns)
                    if sample_df.isnull().any().any() or np.isinf(sample_df).any().any():
                        continue
                    
                    # Get volatility target from portfolio results
                    target_vol = results['mean_risk']  # Use mean risk as target volatility
                    
                    mu = expected_returns.mean_historical_return(sample_df)
                    S = risk_models.CovarianceShrinkage(sample_df).ledoit_wolf()
                    
                    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
                    ef.efficient_risk(target_volatility=target_vol)  # Pass target_volatility parameter
                    weights = ef.clean_weights()
                    performance = ef.portfolio_performance()
                    
                    bootstrap_results.append({
                        'weights': weights,
                        'performance': performance,
                        'target_volatility': target_vol
                    })
                    
                    del ef
                    
                except Exception as e:
                    logging.warning(f"Failed to process bootstrap sample {i}: {str(e)}")
                    continue
                
            # Calculate uncertainty metrics
            if bootstrap_results:  # Only calculate if we have results
                weight_arrays = np.array([[w for w in r['weights'].values()] for r in bootstrap_results])
                return_arrays = np.array([r['performance'][0] for r in bootstrap_results])
                risk_arrays = np.array([r['performance'][1] for r in bootstrap_results])
                
                return {
                    'weight_std': np.std(weight_arrays, axis=0),
                    'return_std': np.std(return_arrays),
                    'risk_std': np.std(risk_arrays),
                    'weight_conf': np.percentile(weight_arrays, [5, 95], axis=0),
                    'return_conf': np.percentile(return_arrays, [5, 95]),
                    'risk_conf': np.percentile(risk_arrays, [5, 95])
                }
            else:
                logging.warning("No valid bootstrap results generated")
                return {}
            
        except Exception as e:
            logging.error(f"Uncertainty calculation failed: {str(e)}")
            return {}

    def optimize_portfolio_hrp(self, data: pd.DataFrame) -> Dict:
        """Optimize portfolio using Hierarchical Risk Parity with better error handling"""
        try:
            logging.info("Starting HRP optimization")
            
            returns = calculate_log_returns(data)
            if returns.empty:
                raise ValueError("Empty returns data")
                
            if not returns.empty:
                returns = scale_data(returns)
                
            if returns.isnull().any().any():
                returns = returns.fillna(0)
                
            if len(returns) == 0:
                raise ValueError("No valid return data available")
                
            weights = hrp(returns)
            
            portfolio_return = (returns * weights).sum().mean() * 252
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            
            result = {
                'model': 'HRP',
                'weights': weights.to_dict(),
                'performance': (portfolio_return, portfolio_volatility, sharpe_ratio)
            }
            
            logging.info("HRP optimization completed")
            return result
            
        except Exception as e:
            logging.error(f"HRP optimization failed: {str(e)}")
            raise
