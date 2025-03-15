# Financial Trading Algorithm

A production-grade algorithmic trading system that uses large language models for market analysis and trading decisions.

## Features

- Multi-model market analysis using OpenAI, Perplexity, and Ollama
- Secure API key management with encryption
- Connection pooling for optimal API performance
- Comprehensive technical and fundamental analysis
- Sentiment analysis and market psychology insights
- Portfolio optimization and risk management
- Real-time market data processing
- Modular and extensible architecture

## Prerequisites

- Python 3.10 or higher
- Ollama running locally (for local LLM support)
- API keys for OpenAI and Perplexity
- Sufficient system memory (8GB+ recommended)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Financial-Trading-Algorithm.git
cd Financial-Trading-Algorithm
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Create .env file
touch .env

# Add your API keys
echo "API_MASTER_KEY=your_master_key_here" >> .env
echo "OPENAI_API_KEY=your_openai_key_here" >> .env
echo "PERPLEXITY_API_KEY=your_perplexity_key_here" >> .env
```

## Configuration

The system uses a configuration file (`config.ini`) for various settings. Key configurations:

```ini
[API]
openai_model = gpt-4o
perplexity_model = llama-3.1-sonar-large-128k-online
ollama_model = llama3.2:1b

[Trading]
max_position_size = 1000
risk_tolerance = 0.02
min_portfolio_size = 30
max_portfolio_size = 120
```

## Usage

1. Start the trading system:

```bash
python main.py
```

2. Monitor the dashboard:

```bash
python -m financial_trading_algorithm.utils.dashboard
```

3. View trading recommendations:

```bash
python -m financial_trading_algorithm.trading.recommendations
```

## Project Structure

```
financial_trading_algorithm/
  ├── api/                 # API client implementations
  │   ├── openai_client.py
  │   ├── perplexity_client.py
  │   └── ollama_client.py
  ├── data/               # Data handling and storage
  │   ├── stock_data.py
  │   └── parquet_handler.py
  ├── analysis/           # Analysis modules
  │   ├── technical.py
  │   ├── sentiment.py
  │   └── fundamentals.py
  ├── trading/            # Trading logic
  │   ├── agent.py
  │   ├── portfolio.py
  │   └── executor.py
  ├── utils/             # Utility functions
  │   ├── config_manager.py
  │   └── api_key_manager.py
  └── main.py            # Entry point
```

## Security

- API keys are encrypted using Fernet encryption
- Master key required for API key decryption
- Connection pooling prevents resource exhaustion
- Rate limiting and retry mechanisms implemented
- Regular security audits recommended

## Performance Optimization

- Connection pooling for API clients
- Caching for expensive calculations
- Batch processing for API requests
- Asynchronous operations where possible
- Efficient data storage using Parquet format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:

```bash
pytest tests/test_api/
pytest tests/test_trading/
pytest tests/test_analysis/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4o API
- Perplexity for advanced language models
- Ollama for local LLM support
- Contributors and maintainers

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
