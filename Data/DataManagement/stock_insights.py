#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta, time
import logging
from typing import List, Dict, Any, Optional, Tuple
import argparse
import json
import pytz

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
from Data.DataManagement.historical_daily_loader import EODHDHistoricalClient
from Data.DataManagement.parquet_handler import ParquetHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stock_insights.log")
    ]
)
logger = logging.getLogger("stock_insights")

class StockInsights:
    """Class for analyzing stock data loaded by the EODHDHistoricalClient"""
    
    def __init__(self, data_path: Path):
        """
        Initialize StockInsights with path to data
        
        Args:
            data_path: Path to the directory containing stock data
        """
        self.data_path = data_path
        self.parquet_handler = ParquetHandler(data_path)
        logger.info("StockInsights initialized with data path: %s", data_path)
    
    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Load data for a specific ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with stock data
        """
        try:
            df = self.parquet_handler.load_dataframe(ticker)
            logger.info(f"Loaded data for {ticker}: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a stock dataframe
        
        Args:
            df: DataFrame with stock data (must contain 'close' column)
            
        Returns:
            DataFrame with technical indicators added
        """
        if df.empty:
            return df
            
        # Make a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Sort by date
        result = result.sort_values('date')
        
        # Calculate moving averages
        result['MA_20'] = result['close'].rolling(window=20).mean()
        result['MA_50'] = result['close'].rolling(window=50).mean()
        result['MA_200'] = result['close'].rolling(window=200).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        result['EMA_12'] = result['close'].ewm(span=12).mean()
        result['EMA_26'] = result['close'].ewm(span=26).mean()
        result['MACD'] = result['EMA_12'] - result['EMA_26']
        result['MACD_Signal'] = result['MACD'].ewm(span=9).mean()
        result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
        
        # Calculate Bollinger Bands
        result['BB_Middle'] = result['close'].rolling(window=20).mean()
        result['BB_Std'] = result['close'].rolling(window=20).std()
        result['BB_Upper'] = result['BB_Middle'] + (result['BB_Std'] * 2)
        result['BB_Lower'] = result['BB_Middle'] - (result['BB_Std'] * 2)
        
        # Calculate ATR (Average True Range)
        result['high_low'] = result['high'] - result['low']
        result['high_close'] = np.abs(result['high'] - result['close'].shift())
        result['low_close'] = np.abs(result['low'] - result['close'].shift())
        result['TR'] = result[['high_low', 'high_close', 'low_close']].max(axis=1)
        result['ATR'] = result['TR'].rolling(window=14).mean()
        
        # Clean up helper columns
        result = result.drop(['high_low', 'high_close', 'low_close', 'TR'], axis=1)
        
        logger.info(f"Calculated technical indicators: {', '.join(col for col in result.columns if col not in df.columns)}")
        return result
    
    def analyze_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with analysis results
        """
        df = self.load_ticker_data(ticker)
        
        if df.empty:
            logger.warning(f"No data available for {ticker}")
            return {"ticker": ticker, "status": "No data"}
        
        # Calculate indicators
        df_with_indicators = self.calculate_technical_indicators(df)
        
        # Get recent data (last 90 days)
        recent_df = df_with_indicators.tail(90).copy()
        
        # Get the most recent values
        latest_data = recent_df.iloc[-1]
        
        # Calculate trend directions (positive = uptrend, negative = downtrend)
        ma_trend = 1 if latest_data['MA_20'] > latest_data['MA_50'] else -1
        macd_trend = 1 if latest_data['MACD'] > latest_data['MACD_Signal'] else -1
        rsi_trend = 1 if latest_data['RSI'] > 50 else -1
        
        # Determine if price is near Bollinger Bands
        bb_position = 0  # Neutral
        if latest_data['close'] > latest_data['BB_Upper']:
            bb_position = 2  # Overbought
        elif latest_data['close'] < latest_data['BB_Lower']:
            bb_position = -2  # Oversold
        elif latest_data['close'] > latest_data['BB_Middle']:
            bb_position = 1  # Above middle
        elif latest_data['close'] < latest_data['BB_Middle']:
            bb_position = -1  # Below middle
        
        # Calculate short-term momentum
        momentum = (latest_data['close'] / recent_df['close'].iloc[-5] - 1) * 100
        
        # Calculate volatility
        volatility = recent_df['close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
        
        # Calculate support and resistance levels
        recent_lows = recent_df['low'].rolling(window=10).min().dropna()
        recent_highs = recent_df['high'].rolling(window=10).max().dropna()
        
        support_levels = []
        resistance_levels = []
        
        if not recent_lows.empty and not recent_highs.empty:
            # Find local minima for support
            for i in range(1, len(recent_lows)-1):
                if recent_lows.iloc[i] < recent_lows.iloc[i-1] and recent_lows.iloc[i] < recent_lows.iloc[i+1]:
                    support_levels.append(recent_lows.iloc[i])
            
            # Find local maxima for resistance
            for i in range(1, len(recent_highs)-1):
                if recent_highs.iloc[i] > recent_highs.iloc[i-1] and recent_highs.iloc[i] > recent_highs.iloc[i+1]:
                    resistance_levels.append(recent_highs.iloc[i])
        
        # Keep only the 3 most recent support and resistance levels
        support_levels = sorted(support_levels)[-3:] if support_levels else []
        resistance_levels = sorted(resistance_levels)[:3] if resistance_levels else []
        
        # Determine overall technical score (-100 to 100)
        technical_score = (ma_trend * 20) + (macd_trend * 30) + (rsi_trend * 20) + (bb_position * 15) + (np.sign(momentum) * 15)
        technical_score = max(-100, min(100, technical_score))  # Clamp to -100 to 100
        
        # Create signal based on technical score
        signal = "Strong Buy" if technical_score > 70 else \
                 "Buy" if technical_score > 30 else \
                 "Hold" if technical_score > -30 else \
                 "Sell" if technical_score > -70 else "Strong Sell"
        
        # Calculate risk level
        risk_level = "High" if volatility > 30 else \
                    "Medium" if volatility > 15 else "Low"
        
        analysis_result = {
            "ticker": ticker,
            "date": latest_data['date'].strftime('%Y-%m-%d'),
            "close": latest_data['close'],
            "technical_indicators": {
                "MA_20": latest_data['MA_20'],
                "MA_50": latest_data['MA_50'],
                "MA_200": latest_data['MA_200'],
                "RSI": latest_data['RSI'],
                "MACD": latest_data['MACD'],
                "MACD_Signal": latest_data['MACD_Signal'],
                "ATR": latest_data['ATR']
            },
            "technical_analysis": {
                "ma_trend": "Bullish" if ma_trend > 0 else "Bearish",
                "macd_trend": "Bullish" if macd_trend > 0 else "Bearish",
                "rsi_level": "Overbought" if latest_data['RSI'] > 70 else \
                             "Oversold" if latest_data['RSI'] < 30 else "Neutral",
                "bollinger_position": ["Oversold", "Below Middle", "Neutral", "Above Middle", "Overbought"][bb_position+2],
                "momentum": f"{momentum:.2f}%",
                "volatility": f"{volatility:.2f}%",
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            },
            "summary": {
                "technical_score": technical_score,
                "signal": signal,
                "risk_level": risk_level
            }
        }
        
        logger.info(f"Analysis completed for {ticker}: Technical Score = {technical_score}, Signal = {signal}")
        return analysis_result
    
    def plot_ticker_analysis(self, ticker: str, save_path: Optional[Path] = None) -> bool:
        """
        Generate technical analysis chart for a ticker
        
        Args:
            ticker: Stock symbol
            save_path: Optional path to save the chart
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = self.load_ticker_data(ticker)
            
            if df.empty:
                logger.warning(f"No data available for {ticker} to plot")
                return False
                
            # Calculate indicators
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Get recent data (last 120 days)
            recent_df = df_with_indicators.tail(120).copy()
            
            # Create the plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                                             gridspec_kw={'height_ratios': [3, 1, 1]})
            fig.suptitle(f'{ticker} Technical Analysis', fontsize=16)
            
            # Main price chart with moving averages and Bollinger Bands
            ax1.plot(recent_df['date'], recent_df['close'], label='Close Price', color='black', linewidth=2)
            ax1.plot(recent_df['date'], recent_df['MA_20'], label='20-day MA', color='blue', linewidth=1)
            ax1.plot(recent_df['date'], recent_df['MA_50'], label='50-day MA', color='orange', linewidth=1)
            ax1.plot(recent_df['date'], recent_df['MA_200'], label='200-day MA', color='red', linewidth=1)
            ax1.plot(recent_df['date'], recent_df['BB_Upper'], '--', label='BB Upper', color='gray', linewidth=0.8)
            ax1.plot(recent_df['date'], recent_df['BB_Lower'], '--', label='BB Lower', color='gray', linewidth=0.8)
            ax1.fill_between(recent_df['date'], recent_df['BB_Lower'], recent_df['BB_Upper'], color='gray', alpha=0.1)
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # MACD plot
            ax2.plot(recent_df['date'], recent_df['MACD'], label='MACD', color='blue', linewidth=1)
            ax2.plot(recent_df['date'], recent_df['MACD_Signal'], label='Signal', color='red', linewidth=1)
            ax2.bar(recent_df['date'], recent_df['MACD_Hist'], label='Histogram', color='green', alpha=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # RSI plot
            ax3.plot(recent_df['date'], recent_df['RSI'], label='RSI', color='purple', linewidth=1)
            ax3.axhline(y=70, color='red', linestyle='--', linewidth=0.8)
            ax3.axhline(y=30, color='green', linestyle='--', linewidth=0.8)
            ax3.fill_between(recent_df['date'], 70, recent_df['RSI'], where=(recent_df['RSI'] >= 70), color='red', alpha=0.3)
            ax3.fill_between(recent_df['date'], 30, recent_df['RSI'], where=(recent_df['RSI'] <= 30), color='green', alpha=0.3)
            ax3.set_ylabel('RSI')
            ax3.set_ylim([0, 100])
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            ax3.set_xlabel('Date')
            
            plt.tight_layout()
            
            if save_path:
                save_file = save_path / f"{ticker}_technical_analysis.png"
                plt.savefig(save_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved technical analysis chart for {ticker} to {save_file}")
                plt.close(fig)
            else:
                plt.show()
                
            return True
            
        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            return False
    
    def compare_tickers(self, tickers: List[str], days: int = 90) -> dict:
        """
        Compare multiple tickers based on performance and technical indicators
        
        Args:
            tickers: List of stock symbols
            days: Number of days to look back
            
        Returns:
            Dictionary with comparison results
        """
        if not tickers:
            logger.warning("No tickers provided for comparison")
            return {}
            
        comparison = {
            "tickers": tickers,
            "period_days": days,
            "date": datetime.now().strftime('%Y-%m-%d'),
            "performance": {},
            "technical_metrics": {},
            "rankings": {}
        }
        
        # Load data for all tickers
        ticker_data = {}
        valid_tickers = []
        
        for ticker in tickers:
            df = self.load_ticker_data(ticker)
            if not df.empty:
                ticker_data[ticker] = df.sort_values('date').tail(days+1)
                valid_tickers.append(ticker)
        
        if not valid_tickers:
            logger.warning("No valid data found for any of the provided tickers")
            return comparison
            
        # Calculate period returns
        performance = {}
        for ticker, df in ticker_data.items():
            if len(df) > 1:
                first_close = df.iloc[0]['close']
                last_close = df.iloc[-1]['close']
                period_return = ((last_close / first_close) - 1) * 100
                performance[ticker] = {
                    "start_price": first_close,
                    "end_price": last_close,
                    "return_pct": period_return
                }
        
        # Calculate technical metrics for each ticker
        technical_metrics = {}
        for ticker, df in ticker_data.items():
            if len(df) > 20:  # Need at least 20 days for some indicators
                df_with_indicators = self.calculate_technical_indicators(df)
                latest = df_with_indicators.iloc[-1]
                
                technical_metrics[ticker] = {
                    "rsi": latest['RSI'],
                    "macd": latest['MACD'],
                    "ma_20_50_cross": 1 if latest['MA_20'] > latest['MA_50'] else -1,
                    "volatility": df['close'].pct_change().std() * np.sqrt(252) * 100
                }
        
        # Rank tickers by different metrics
        rankings = {}
        
        # Rank by return
        if performance:
            return_ranking = sorted(
                [(ticker, data["return_pct"]) for ticker, data in performance.items()], 
                key=lambda x: x[1], 
                reverse=True
            )
            rankings["by_return"] = [t[0] for t in return_ranking]
        
        # Rank by RSI (closest to 50 is best - neither overbought nor oversold)
        if technical_metrics:
            rsi_ranking = sorted(
                [(ticker, abs(data["rsi"] - 50)) for ticker, data in technical_metrics.items()], 
                key=lambda x: x[1]
            )
            rankings["by_rsi_balance"] = [t[0] for t in rsi_ranking]
        
        # Rank by MACD trend strength
        if technical_metrics:
            macd_ranking = sorted(
                [(ticker, data["macd"]) for ticker, data in technical_metrics.items()], 
                key=lambda x: x[1], 
                reverse=True
            )
            rankings["by_macd_strength"] = [t[0] for t in macd_ranking]
        
        # Rank by volatility (lower is better)
        if technical_metrics:
            volatility_ranking = sorted(
                [(ticker, data["volatility"]) for ticker, data in technical_metrics.items()], 
                key=lambda x: x[1]
            )
            rankings["by_stability"] = [t[0] for t in volatility_ranking]
        
        # Update the results
        comparison["performance"] = performance
        comparison["technical_metrics"] = technical_metrics
        comparison["rankings"] = rankings
        
        logger.info(f"Completed comparison of {len(valid_tickers)} tickers over {days} days")
        return comparison

def is_market_closed():
    """
    Check if US market is closed (after 4pm ET on weekdays)
    Returns: 
        tuple: (is_closed, reason)
    """
    # Get current time in ET
    et_zone = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_zone)
    
    # Check if weekend
    if now_et.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return True, "Weekend"
    
    # Check if before market open (9:30am ET) or after market close (4:00pm ET)
    market_open = time(9, 30, 0)
    market_close = time(16, 0, 0)
    
    if now_et.time() < market_open:
        return True, "Before market open"
    elif now_et.time() >= market_close:
        return True, "After market close"
    
    # Market is open
    return False, "Market is open"

def was_updated_today(data_path):
    """
    Check if data was already updated today
    
    Args:
        data_path: Path to data directory
        
    Returns:
        bool: True if updated today, False otherwise
    """
    last_update_file = data_path / "last_update.json"
    
    if not last_update_file.exists():
        return False
    
    try:
        with open(last_update_file, 'r') as f:
            last_update = json.load(f)
        
        last_update_date = datetime.fromisoformat(last_update.get('date', '2000-01-01'))
        today = datetime.now().date()
        
        return last_update_date.date() == today
    except Exception as e:
        logger.error(f"Error checking last update: {e}")
        return False

def record_update(data_path):
    """
    Record that data was updated today
    
    Args:
        data_path: Path to data directory
    """
    last_update_file = data_path / "last_update.json"
    
    try:
        with open(last_update_file, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'status': 'success'
            }, f)
    except Exception as e:
        logger.error(f"Error recording update: {e}")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Analyze stock data loaded from EODHD')
    parser.add_argument('--data-path', type=str, default=None, 
                       help='Path to the data directory')
    parser.add_argument('--analyze', type=str, 
                       help='Analyze a specific ticker')
    parser.add_argument('--plot', type=str,
                       help='Generate technical analysis chart for a ticker')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Compare multiple tickers')
    parser.add_argument('--days', type=int, default=90,
                       help='Number of days to look back for comparison')
    parser.add_argument('--load-history', action='store_true',
                       help='Load historical data for S&P 500 tickers')
    parser.add_argument('--update', action='store_true',
                       help='Update data with latest daily values')
    parser.add_argument('--api-key', type=str, 
                       help='EODHD API key (or set EODHD_API_KEY env var)')
    parser.add_argument('--ignore-time', action='store_true',
                       help='Ignore time check (update regardless of time)')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if already updated today')
    
    args = parser.parse_args()
    
    # Set data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Default to a data directory in the project root
        data_path = Path(__file__).parent.parent.parent / "data"
    
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Get API key if needed for data loading/updating
    api_key = None
    if args.load_history or args.update:
        api_key = args.api_key or os.environ.get("EODHD_API_KEY")
        if not api_key:
            logger.error("API key required for loading/updating data. Set EODHD_API_KEY environment variable or use --api-key")
            return 1
    
    # Check if we should proceed with update
    if args.update and not args.ignore_time:
        market_closed, reason = is_market_closed()
        already_updated = was_updated_today(data_path)
        
        if not market_closed:
            logger.warning(f"Market is still open ({reason}). Updates should only run after market close (4pm ET). Use --ignore-time to override.")
            return 1
        
        if already_updated and not args.force:
            logger.info(f"Data has already been updated today. Use --force to update again.")
            return 0
    
    # Create EODHDHistoricalClient if needed
    client = None
    if args.load_history or args.update:
        client = EODHDHistoricalClient(api_key, data_path)
    
    # Load historical data if requested
    if args.load_history:
        logger.info("Loading historical data for S&P 500 tickers...")
        symbols = client.get_current_sp500_symbols()
        if not symbols:
            logger.error("Failed to retrieve S&P 500 symbols")
            return 1
        
        logger.info(f"Found {len(symbols)} S&P 500 symbols. Starting historical load...")
        client.load_historical_data(symbols, batch_size=20)
        logger.info("Historical data load completed")
    
    # Update data if requested
    if args.update:
        logger.info("Updating data with latest daily values...")
        client.bulk_update_daily_data()
        logger.info("Data update completed")
        record_update(data_path)
    
    # Create StockInsights instance
    insights = StockInsights(data_path)
    
    # Analyze a specific ticker if requested
    if args.analyze:
        logger.info(f"Analyzing ticker: {args.analyze}")
        analysis = insights.analyze_ticker(args.analyze)
        
        # Print analysis in a nice format
        print(f"\n=== Technical Analysis for {args.analyze} ({analysis['date']}) ===")
        print(f"Current Price: ${analysis['close']:.2f}")
        
        print("\nTechnical Indicators:")
        for name, value in analysis['technical_indicators'].items():
            print(f"  {name}: {value:.2f}")
        
        print("\nTechnical Analysis:")
        for name, value in analysis['technical_analysis'].items():
            if name in ['support_levels', 'resistance_levels']:
                print(f"  {name}: {', '.join([f'${level:.2f}' for level in value])}")
            else:
                print(f"  {name}: {value}")
        
        print("\nSummary:")
        print(f"  Technical Score: {analysis['summary']['technical_score']}")
        print(f"  Signal: {analysis['summary']['signal']}")
        print(f"  Risk Level: {analysis['summary']['risk_level']}")
    
    # Plot technical analysis chart if requested
    if args.plot:
        logger.info(f"Generating technical analysis chart for {args.plot}")
        success = insights.plot_ticker_analysis(args.plot)
        if not success:
            logger.error(f"Failed to generate chart for {args.plot}")
    
    # Compare tickers if requested
    if args.compare:
        logger.info(f"Comparing tickers: {args.compare}")
        comparison = insights.compare_tickers(args.compare, args.days)
        
        if comparison and 'performance' in comparison and comparison['performance']:
            print(f"\n=== Ticker Comparison ({args.days} days) ===")
            
            # Performance table
            print("\nPerformance:")
            for ticker, perf in comparison['performance'].items():
                print(f"  {ticker}: ${perf['start_price']:.2f} → ${perf['end_price']:.2f} ({perf['return_pct']:.2f}%)")
            
            # Technical metrics
            if 'technical_metrics' in comparison and comparison['technical_metrics']:
                print("\nTechnical Metrics:")
                for ticker, metrics in comparison['technical_metrics'].items():
                    ma_status = "MA20 > MA50" if metrics['ma_20_50_cross'] > 0 else "MA20 < MA50"
                    print(f"  {ticker}: RSI={metrics['rsi']:.2f}, MACD={metrics['macd']:.4f}, {ma_status}, Volatility={metrics['volatility']:.2f}%")
            
            # Rankings
            if 'rankings' in comparison and comparison['rankings']:
                print("\nRankings:")
                for ranking_type, ranked_tickers in comparison['rankings'].items():
                    print(f"  {ranking_type}: {' > '.join(ranked_tickers)}")
        else:
            logger.warning("No valid comparison data available")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 