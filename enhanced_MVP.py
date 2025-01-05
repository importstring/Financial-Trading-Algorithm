def _calculate_uncertainty(self, data: pd.DataFrame, results: Dict) -> Dict:
    """Calculate uncertainty estimates with improved validation"""
    try:
        # Generate bootstrap samples with validation
        bootstrap_samples = self.bootstrap_returns(data)
        bootstrap_results = []
        
        # Calculate target volatility from the original results
        target_volatility = results['mean_risk']  # Use the mean risk as target
        
        for i, sample in enumerate(bootstrap_samples):
            try:
                if i % 10 == 0:
                    logging.info(f"Processing bootstrap sample {i}/{len(bootstrap_samples)}")
                    gc.collect()
                
                # Create DataFrame and validate
                sample_df = pd.DataFrame(sample, columns=data.columns)
                if sample_df.isnull().any().any() or np.isinf(sample_df).any().any():
                    continue
                
                # Calculate returns and risk for the sample
                mu = expected_returns.mean_historical_return(sample_df)
                S = risk_models.CovarianceShrinkage(sample_df).ledoit_wolf()
                
                ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
                # Pass the target_volatility parameter
                ef.efficient_risk(target_volatility=float(target_volatility))
                weights = ef.clean_weights()
                performance = ef.portfolio_performance()
                
                bootstrap_results.append({
                    'weights': weights,
                    'performance': performance
                })
                
            except Exception as e:
                logging.warning(f"Failed to process bootstrap sample {i}: {str(e)}")
                continue
            
        # ...existing code...
