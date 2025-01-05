from enhanced_MVP import EnhancedPortfolioOptimizer, clean_data, calculate_log_returns
import pandas as pd
import logging
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def validate_data(data: pd.DataFrame) -> None:
    """Enhanced data validation with detailed statistics"""
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Data description:\n{data.describe()}")
    logging.info(f"Any infinite values: {np.isinf(data).any().any()}")
    logging.info(f"Any NaN values: {data.isna().any().any()}")
    
    # Add correlation analysis
    corr_matrix = data.corr()
    logging.info(f"Average correlation: {corr_matrix.mean().mean():.4f}")
    logging.info(f"Max correlation: {corr_matrix.max().max():.4f}")
    
    # Add numerical stability checks
    log_returns = calculate_log_returns(data)
    logging.info(f"Log returns statistics:")
    logging.info(f"  Range: {log_returns.min().min():.6f} to {log_returns.max().max():.6f}")
    logging.info(f"  Mean: {log_returns.mean().mean():.6f}")
    logging.info(f"  Std: {log_returns.std().mean():.6f}")
    
    # Check for zeros or negative values
    if (data <= 0).any().any():
        logging.warning("Data contains zero or negative values")
        
    # Check for symmetry in correlation matrix
    if not np.allclose(corr_matrix, corr_matrix.T):
        logging.warning("Correlation matrix is not symmetric")

def robust_scale_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply robust scaling to the data"""
    scaler = RobustScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    return scaled_data

def main():
    start_time = datetime.now()
    logging.info(f"Starting enhanced portfolio optimization at {start_time}")
    
    data_path = "/Users/simon/Financial-Trading-Algorithm/Data/Stock-Data"
    optimizer = EnhancedPortfolioOptimizer(data_path)
    
    try:
        # Load data using the enhanced loader
        data = optimizer.load_data()
        logging.info(f"Loaded data for {len(data.columns)} stocks")
        
        # Enhanced validation
        validate_data(data)
        
        # Apply improved cleaning and transformation
        clean_data_df = clean_data(data)
        log_returns = calculate_log_returns(clean_data_df)
        scaled_data = robust_scale_data(log_returns)
        
        # Log data quality metrics
        logging.info(f"Original data shape: {data.shape}")
        logging.info(f"Clean data shape: {clean_data_df.shape}")
        logging.info(f"Log returns shape: {log_returns.shape}")
        logging.info(f"Data reduction: {(1 - len(clean_data_df)/len(data))*100:.2f}%")
        
        # Log volatility statistics using improved calculation
        volatilities = np.sqrt(np.diag(scaled_data.cov()))
        min_vol, max_vol = volatilities.min(), volatilities.max()
        mean_vol = volatilities.mean()
        median_vol = np.median(volatilities)
        
        logging.info(f"Data volatility statistics:")
        logging.info(f"  Range: {min_vol:.6f} to {max_vol:.6f}")
        logging.info(f"  Mean: {mean_vol:.6f}")
        logging.info(f"  Median: {median_vol:.6f}")
        
        results = optimizer.optimize_portfolio(scaled_data)
        
        # Add HRP-specific logging
        hrp_results = results['individual_results'][0]  # HRP results are first
        if hrp_results['model'] == 'HRP':
            logging.info("HRP Optimization Results:")
            logging.info(f"Return: {hrp_results['performance'][0]:.4f}")
            logging.info(f"Volatility: {hrp_results['performance'][1]:.4f}")
            logging.info(f"Sharpe Ratio: {hrp_results['performance'][2]:.4f}")
        
        # Save detailed results
        output_path = Path("/Users/simon/Financial-Trading-Algorithm/Data/Info/Logs/Enhanced-Results")
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pd.to_pickle(results, output_path / f'enhanced_results_{timestamp}.pkl')
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Optimization completed in {duration}")
        
    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error details: {str(e)}")
        raise

if __name__ == "__main__":
    main()
