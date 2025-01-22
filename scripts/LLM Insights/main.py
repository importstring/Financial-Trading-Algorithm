import re
import openai
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import pandas as pd
import glob
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import ollama
import random
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
import math
from loading import LoadingBar

# Get project root directory
def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / '.git').exists():
            return parent
        if parent.name == 'Financial-Trading-Algorithm-progress-updated':
            return parent
    print("Could not find project root directory")
    raise FileNotFoundError("Could not find project root directory")

# Set up project paths
PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / 'Data'
CONVERSION_PATH = DATA_PATH / 'Conversations'
STOCK_DATA_PATH = DATA_PATH / 'Stock-Data'
DATA_MANAGEMENT_PATH = DATA_PATH / 'DataManagement'
API_KEYS_PATH = PROJECT_ROOT / 'scripts' / 'LLM Insights' / 'API_Keys'
PORTFOLIO_PATH = DATA_PATH / 'Databases' / 'Portfolio'
TICKERS_PATH = DATA_PATH / 'Info' / 'Tickers'

# Add Data Management to system path
sys.path.append(str(PROJECT_ROOT))
from Data.DataManagement.maintenance import update_data

logging.getLogger("httpx").setLevel(logging.WARNING)

class ChatGPT4o:
    def __init__(self):
        self.loading_bar = LoadingBar()
        self.tests = 1
        self.checkmark = "✅"
        self.crossmark = "❌"
        self.api_key_path = API_KEYS_PATH / 'OpenAI.txt'
        self.default_model = "ChatGPT-4o"
        self.default_role = (
            "You are a financial analyst who has talked with millions of people about the stock market. You are here to provide insights about the stock market based on your interactions."
            )
        self.api_key = self.read_api()
        self.sia = self._initialize_sentiment_analyzer()
        self.loading_bar.update('ChatGPT4o.__init__', 'Instance initialization complete', 1, 'completed')
        
    def _initialize_sentiment_analyzer(self):
        """Initialize VADER sentiment analyzer."""
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()

    def analyze_sentiment(self, statement):
        self.loading_bar.update('analyze_sentiment', 'Starting sentiment analysis')
        sentiment_scores = self.sia.polarity_scores(statement)
        compound_score = sentiment_scores['compound']
         
        if compound_score >= 0.05:
            sentiment = "Bullish"
        elif compound_score <= -0.05:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        normalized_score = compound_score * 2
        self.loading_bar.update('analyze_sentiment', 'Sentiment analysis complete', 1, 'completed')
        return sentiment, normalized_score

    def read_api(self):
        self.loading_bar.update('read_api', 'Reading API key')
        try:
            with open(self.api_key_path, 'r') as file:
                self.api_key = file.read().strip()
        except FileNotFoundError:
            self.loading_bar.update('API key file not found')
            raise ValueError(f"API key file not found at {self.api_key_path}")
        
        if not self.api_key:
            self.loading_bar.update('Empty API key')
            raise ValueError("OpenAI API key is empty")
        self.loading_bar.update('API key loaded successfully')
        self.loading_bar.update('read_api', 'API key read complete', 1, 'completed')
        return self.api_key

    
    def query_OpenAI(self, model="Chatgpt-4o", query="", max_tokens=150, temperature=0.5, role=""):
        self.loading_bar.update('query_OpenAI', 'Preparing OpenAI query')
        model = self.default_model if model == "" else model
        role = self.default_role if role == "" else role
        def test_input_validity(self, model, query, max_tokens, temperature):

            def validate_chatgpt_model(self, model_input):
                valid_models = [
                    "gpt-4o", "chatgpt-4o-latest", "gpt-4-0125-preview",
                    "gpt-4-1106-preview", "gpt-4", "gpt-4-0613",
                    "gpt-4-0314", "gpt-4o-mini", "gpt-3.5-turbo"
                ]
                result = True if model_input in valid_models else False
                description = f"{"Valid" if result else "Invalid"} ChatGPT model: {model_input}"
                return result, description
                
            def validate_query(self, query_input):
                result = True if query_input != "" else False
                description = f"{"Valid" if result else "Invalid"} query provided."
                return result, description
                
            def validate_max_tokens(self, max_tokens_input):
                result = True if max_tokens_input > 0 else False
                description = f"{"Valid" if result else "Invalid"} max_tokens: {max_tokens_input}"
                return result, description
            
            def validate_temperature(self, temperature_input):
                result = True if 0 <= temperature_input <= 1 else False
                description = f"{"Valid" if result else "Invalid"} temperature: {temperature_input}"
                return result, description

            tests = [
                validate_chatgpt_model(model),
                validate_query(query),
                validate_max_tokens(max_tokens),
                validate_temperature(temperature)
            ]

            for test in tests:
                print(f"Test {self.tests}: {self.checkmark if test[0] else self.crossmark} {test[1]}")
                self.tests += 1
            
            return all([test[0] for test in tests])

        if not test_input_validity(model, query, max_tokens, temperature):
            return "Invalid input provided. Please check the input parameters."

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            output, status =  response.choices[0].message.content, True
        except Exception as e:
            output, status =  f"An error occurred: {e}", False
        
        self.loading_bar.update('query_OpenAI', 'OpenAI query complete', 1, 'completed')
        return output, status

    def evaluate_response(self, response):
        self.loading_bar.update('evaluate_response', 'Evaluating response')
        """
        Evaluate response from ChatGPT-4o API and return insights with sentiment analysis.
        """
        try:
            if not response:
                return "No response received."
            if "error" in str(response).lower():
                return "The response contains an error."
            
            sentiment, score = self.analyze_sentiment(str(response))
            return {
                "text": str(response),
                "sentiment": sentiment,
                "sentiment_score": score
            }
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            return "Error during evaluation."
        self.loading_bar.update('evaluate_response', 'Response evaluation complete', 1, 'completed')
        

class Perplexity:
    def __init__(self):
        self.loading_bar = LoadingBar()
        self.tests = 1
        self.checkmark = "✅"
        self.crossmark = "❌"
        self.api_key_path = API_KEYS_PATH / 'Perplexity.txt'
        self.default_model = "llama-3.1-sonar-small-128k-online"
        self.default_role = (
            "You are an AI financial analyst providing market insights based on extensive data analysis."
        )
        self.api_key = self.read_api()
        self.sia = self._initialize_sentiment_analyzer()
        self.cache = {}
        self.loading_bar.update('Perplexity.__init__', 'Instance initialization complete', 1, 'completed')
        
    def read_api(self) -> str:
        self.loading_bar.update('read_api', 'Reading Perplexity API key')
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.read().strip()
                if not api_key:
                    raise ValueError("Perplexity API key is empty")
                return api_key
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")
        self.loading_bar.update('read_api', 'Perplexity API key read', 1, 'completed')

    def query_perplexity(self, model: str = "", query: str = "", 
                        max_tokens: int = 150, temperature: float = 0.5, 
                        role: str = "") -> Tuple[str, bool]:
        self.loading_bar.update('query_perplexity', 'Querying Perplexity')
        """
        Query the Perplexity API with validation and caching.
        Returns tuple of (response_text, success_status)
        """
        model = self.default_model if model == "" else model
        role = self.default_role if role == "" else role

        cache_key = f"{query}_{model}_{max_tokens}_{temperature}_{role}"
        if cache_key in self.cache:
            return self.cache[cache_key], True

        if not self._test_input_validity(model, query, max_tokens, temperature):
            return "Invalid input provided. Please check the input parameters.", False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "query": query,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "role": role
            }
            response = requests.post(
                "https://api.perplexity.ai/query",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            output = result.get('response', '')
            self.cache[cache_key] = (output, True)
            return output, True
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logging.error(error_msg)
            return error_msg, False
        self.loading_bar.update('query_perplexity', 'Perplexity query complete', 1, 'completed')

    def _test_input_validity(self, model: str, query: str, 
                           max_tokens: int, temperature: float) -> bool:
        """Validate input parameters for the API request."""
        def validate_perplexity_model(model_input: str) -> Tuple[bool, str]:
            valid_models = [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online", 
                "llama-3.1-sonar-huge-128k-online"
            ] # Other models can be added but I don't have access to them as I am not on their beta program
            result = model_input in valid_models
            description = f"{'Valid' if result else 'Invalid'} Perplexity model: {model_input}"
            return result, description

        tests = [
            validate_perplexity_model(model),
            (bool(query.strip()), "Valid query provided." if query.strip() else "Invalid query."),
            (max_tokens > 0, f"{'Valid' if max_tokens > 0 else 'Invalid'} max_tokens: {max_tokens}"),
            (0 <= temperature <= 1, f"{'Valid' if 0 <= temperature <= 1 else 'Invalid'} temperature: {temperature}")
        ]

        for test in tests:
            print(f"Test {self.tests}: {self.checkmark if test[0] else self.crossmark} {test[1]}")
            self.tests += 1

        return all(test[0] for test in tests)

    def _initialize_sentiment_analyzer(self):
        """Initialize VADER sentiment analyzer."""
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()

    def analyze_sentiment(self, statement):
        self.loading_bar.update('analyze_sentiment', 'Starting sentiment analysis')
        """Analyze sentiment of a statement using VADER."""
        sentiment_scores = self.sia.polarity_scores(statement)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            sentiment = "bullish"
        elif compound_score <= -0.05:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        normalized_score = compound_score * 2
        self.loading_bar.update('analyze_sentiment', 'Sentiment analysis complete', 1, 'completed')
        return sentiment, normalized_score

    def evaluate_response(self, response: Union[str, Dict]) -> Dict:
        self.loading_bar.update('evaluate_response', 'Evaluating Perplexity response')
        """
        Evaluate response from Perplexity API with enhanced error handling.
        """
        try:
            if not response:
                return {"error": "No response received", "sentiment": "neutral", "sentiment_score": 0.0}
            
            text = response if isinstance(response, str) else str(response)
            sentiment, score = self.analyze_sentiment(text)
            self.loading_bar.update('evaluate_response', 'Response evaluation complete', 1, 'completed')
            
            return {
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": score,
                "error": None
            }
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            return {
                "error": f"Error during evaluation: {str(e)}",
                "sentiment": "neutral",
                "sentiment_score": 0.0
            } 


class StockData:
    def __init__(self):
        self.loading_bar = LoadingBar()
        self.data_path = STOCK_DATA_PATH / 'Stock-Data'
        self.maintain_stock_data()
        self.stock_data = {}  # Initialize empty dictionary
        self.read_stock_data()  # Now populate it
        self.tickers = list(self.stock_data.keys())
        self.loading_bar.update('StockData.__init__', 'Stock data initialization complete', 1, 'completed')

    def get_stock_data(self, tickers):
        self.loading_bar.update('get_stock_data', 'Fetching stock data')
        data = {}
        for ticker in tickers:
            data[ticker] = self.stock_data[ticker]
        self.loading_bar.update('get_stock_data', 'Fetched stock data', 1, 'completed')
        return data
    
    def read_stock_data(self):
        self.loading_bar.update('read_stock_data', 'Reading all stock data')
        """
        Data Path --> self.data_path
        Inside the Data path
        - Read all the data
        Format:
        {ticker}.csv
        Inside the CSV:
            Date,Open,High,Low,Close,Volume
            E.g: 2024-11-18,25.75,25.86,25.74,25.81,1209300.0
        """

        stock_data = {}
        stock_data_path = DATA_PATH / 'Stock-Data'
        try:
            # Get all CSV files in the data path using pathlib
            csv_files = list(self.data_path.glob("*.csv"))
            
            for file_path in csv_files:
                # Extract ticker from filename
                ticker = file_path.stem
                
                # Read CSV file
                df = pd.read_csv(file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                
                # Store in dictionary with ticker as key
                stock_data[ticker] = df
            self.loading_bar.update('read_stock_data', 'Stock data read complete', 1, 'completed')
            return stock_data
            
        except Exception as e:
            logging.error(f"Error reading stock data: {e}")
            return {}

    def maintain_stock_data(self) -> Optional[bool]:
        self.loading_bar.update('maintain_stock_data', 'Maintaining stock data')
        """
        Run update_data() from the Data Management module.
        
        Returns:
            Optional[bool]: True if update was successful, False if an error occurred, None if update_data() doesn't return a status.
        """
        try:


            logging.info("Starting data update process")
            result = update_data()
            logging.info("Data update process completed")
            self.loading_bar.update('maintain_stock_data', 'Maintenance complete', 1, 'completed')
            return result if isinstance(result, bool) else None
        
        except ImportError as e:
            logging.error(f"Failed to import update_data: {e}")
            return False
        except Exception as e:
            logging.error(f"An error occurred during data update: {e}")
            return False

class Ollama:
    def __init__(self):
        self.loading_bar = LoadingBar()
        self.model = 'llama3.2:1b'
        self.max_iterations = 10
        self.max_time = 1800  # 30 minutes in seconds
        self.action_handlers = {
            "research": "Research x",
            "insight": "Get insights on x",
            "buy": "Buy y shares of x at $z",
            "sell": "Sell y shares of x at $z",
            "hold": "Hold x",
            "stockdata": "Get stock data for x",
            'reason': "Reason for x"
        }
        self.loading_bar.update('Ollama.__init__', 'Ollama initialization complete', 1, 'completed')

    def reason(self, query):
        self.loading_bar.update('reason', 'Initiating reasoning')
        self.loading_bar.update('reason', 'Initiating chain of thought reasoning')
        
        chain_of_thought = [
            f"Initial query: {query}",
            "Step 1: Analyze the query and break it down into key components",
            "Step 2: Identify relevant financial concepts and data points",
            "Step 3: Consider multiple perspectives and potential outcomes",
            "Step 4: Evaluate risks and opportunities",
            "Step 5: Synthesize insights and form a conclusion"
        ]
        
        for step in chain_of_thought[1:]:
            self.loading_bar.update(f'Reasoning: {step}')
            step_query = f"{step}\nBased on the previous steps and the initial query, what insights can we derive?"
            step_response = self.ollama.query_ollama(step_query)
            chain_of_thought.append(f"Output: {step_response}")
        
        final_reasoning = "\n".join(chain_of_thought)
        self.loading_bar.update('Finalizing chain of thought reasoning')
        self.loading_bar.update('reason', 'Reasoning complete', 1, 'completed')
        return final_reasoning


    def query_ollama(self, prompt: str) -> str:
        self.loading_bar.update('query_ollama', 'Querying Ollama')
        try:
            response = ollama.generate(
                model = 'llama3.2:1b',
                prompt=prompt,
#z                max_tokens=120000,
#                temperature=0.5,
                system="You are a financial agent making decisions based on market analysis."
            )
            self.loading_bar.update('query_ollama', 'Ollama query complete', 1, 'completed')
            return response['response']
        except Exception as e:
            logging.error(f"Error querying Ollama: {e}")
            return "An error occurred during the query."

        
    


    def _parse_response(self, response: str) -> tuple:
        """Parse response to extract action and parameters."""
        try:
            match = re.search(r'ACTION:\s*(\w+)(?:\s*PARAMS:\s*(.+))?', response, re.IGNORECASE)
            if match:
                action = match.group(1).lower()
                params = [p.strip() for p in match.group(2).split(',')] if match.group(2) else [""]
                return action, params
            
            words = response.lower().split()
            if words and words[0] in self.action_handlers:
                return words[0], words[1:] if len(words) > 1 else [""]
            
            return None, []
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return None, []


    def query_ollama_for_plan(self, prompt: str) -> str:
        self.loading_bar.update('query_ollama_for_plan', 'Generating plan with Ollama')
        try:
            self.loading_bar.update('Generating plan, Querying Ollama')
            response = ollama.generate(
                model=self.model,
                prompt=f"{prompt}\n\nIMPORTANT: Your response MUST start with 'ACTION:' followed by one of these actions: {', '.join(self.action_handlers.keys())}. Then, add 'PARAMS:' followed by the necessary parameters separated by commas.",
                system="You are a financial agent making decisions based on market analysis. Always respond with an action and parameters in the specified format."
            )
            self.loading_bar.update('Generating plan', 'Querying Ollama', 1, 'completed')
            self.loading_bar.update('query_ollama_for_plan', 'Plan generation complete', 1, 'completed')
            return response['response']
        except Exception as e:
            logging.error(f"Error querying Ollama: {e}")
            return "ACTION: error PARAMS: An error occurred during the query."


    def make_plan(self, query: str) -> str:
        self.loading_bar.update('make_plan', 'Making plan')
        self.loading_bar.update('Generating plan')
        if not query:
            raise ValueError("Query cannot be empty")

        self.loading_bar.update('Generating plan', 'Initializing start time')
        start_time = time.time()
        self.loading_bar.update('Generating plan', 'Initializing start time', 1, 'completed')
        self.loading_bar.update('Generating plan', 'Initializing plan')
        plan = []
        self.loading_bar.update('Generating plan', 'Initializing plan', 1, 'completed')
        
        def get_plan():
            return "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

        try:
            while time.time() - start_time < self.max_time:
                reasoning_prompt = f"""
                Query: {query}
                Current plan: {get_plan()}
                
                What's the next step in the plan? Be specific and actionable.
                """
                
                response = self.query_ollama_for_plan(reasoning_prompt)
                next_step = response.strip()
                
                if next_step:
                    plan.append(next_step)
                
                if len(plan) >= self.max_iterations:
                    break
            self.loading_bar.update('make_plan', 'Plan made', 1, 'completed')
            return f"Final plan:\n{get_plan()}"

        except Exception as e:
            logging.error(f"Error in make_plan: {e}")
            return f"Error creating plan: {str(e)}"

class Agent:
    def __init__(self, initial_balance=100, risk_tolerance=0.02):
        self.loading_bar.update('Agent.__init__', 'Agent initialization')
        self.balance = initial_balance
        
        # Inialize components
        self.loading_bar = LoadingBar()
        self.chatgpt = ChatGPT4o(self.loading_bar)
        self.perplexity = Perplexity(self.loading_bar)
        self.stock_data = StockData(self.loading_bar)
        self.ollama = Ollama(self.loading_bar)


        # Spinner
        self.console = Console()
        self.tasks = { 
            }
        
        # Get the Stock info from the Stock Data
        self.stock_data.read_stock_data()

        # Initalize starting values
        self.portfolio = {}
        self.new_recommendations = []
        self.history = {}
        self.active_tickers = []
        self.plan = "" 
        self.previous_actions = []
        self.current_actions = {}
        self.tickers = self.stock_data.tickers
        self.decisons = None
        self.environment = None
        self.active_tickers = None

        self.action_inputs = None # Inputs for the current action
        self.actions = {
            "buy": self.buy, 
            "sell": self.sell,
            "hold": self.hold,
            "research": self.research,
            'insight': self.insight,
            "reason": self.reason,
            "stockdata": lambda tickers: self.stock_data.get_stock_data(tickers) if tickers else None
        }

        self.risk_tolerance = risk_tolerance
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.tasks = {
            "Initialization": ["Setting up dependencies", "Reading/Loading data"],
            "Planning": ["Generating plan actions"],
            "PickingTickers": ["Filtering stocks"],
            "Validation": ["Testing outputs"],
            "ResearchAndInsight": ["Gathering research"],
            "ExecuteTrades": ["Executing trades"],
            "Learning": ["Updating model", "Saving progress"]
        }
        self.console = Console()
        self.progress = None
        self.task_progresses = {}
        self.overall_task = None
        self.most_recent_action = "Initializing..."
        self.loading_bar.update('Agent.__init__', 'Initialization complete', 1, 'completed')


    def buy(self, ticker, shares, price, min=0, max=0):
        if ticker.empty():
            return
        self.loading_bar.update(f'Buying {ticker}')
        if min == max == 0:
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max')
            self.loading_bar.update(f'Buying {ticker}', 'Predicting future volume')
            predicted_volume = self.predict_future_volume(ticker)
            self.loading_bar.update(f'Buying {ticker}', 'Predicting future volume', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 0.25)
            self.loading_bar.update(f'Buying {ticker}', 'Predicting future volume', 1, 'completed')
            self.loading_bar.update(f'Buying {ticker}', 'Calculating dynamic lead time')
            dynamic_lead_time = self.calculate_dynamic_lead_time(ticker)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating dynamic lead time', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 0.5)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating dynamic lead time', 1, 'completed')
            self.loading_bar.update(f'Buying {ticker}', 'Calculating adaptive safety stock')
            safety_stock = self.calculate_adaptive_safety_stock(ticker)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating adaptive safety stock', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 0.75)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating adaptive safety stock', 1, 'completed')
            self.loading_bar.update(f'Buying {ticker}', 'Calculating base min')
            base_min = (predicted_volume * dynamic_lead_time) + safety_stock
            self.loading_bar.update(f'Buying {ticker}', 'Calculating base min', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 1)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating base min', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Optimizing reorder quantity')
            reorder_qty = self.optimize_reorder_quantity(ticker, base_min)
            self.loading_bar.update(f'Buying {ticker}', 'Optimizing reorder quantity', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 0.93)
            self.loading_bar.update(f'Buying {ticker}', 'Optimizing reorder quantity', 1, 'completed')
            self.loading_bar.update(f'Buying {ticker}', 'Adjusting levels for sentiment')
            sentiment_adjusted_min, sentiment_adjusted_max = self.adjust_levels_for_sentiment(
                ticker,
                base_min,
                base_min + reorder_qty
            )
            self.loading_bar.update(f'Buying {ticker}', 'Adjusting levels for sentiment', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 0.97)
            self.loading_bar.update(f'Buying {ticker}', 'Adjusting levels for sentiment', 1, 'completed')
            self.loading_bar.update(f'Buying {ticker}', 'Finalizing buy')
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 0.98)
            min, max = round(sentiment_adjusted_min), round(sentiment_adjusted_max)
            self.loading_bar.update(f'Buying {ticker}', 'Finalizing buy', 0.87)
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 1)
            self.loading_bar.update(f'Buying {ticker}', 'Finalizing buy', 1, 'completed')
            self.loading_bar.update(f'Buying {ticker}', 'Calculating min and max', 1, 'completed')
        self.loading_bar.update(f'Buying {ticker}', 'Saving information')
        self.new_recommendations.append(f"buy {shares} {ticker} at {price} with a min of {min} and a max of {max}")
        self.loading_bar.update(f'Buying {ticker}', 'Saving information', 1)
        self.loading_bar.update(f'Buy {ticker}', 'completed', 1, f'Buy action completed: Buy {shares} of {ticker} at {price} with a min of {min} and a max of {max}')

    def adjust_levels_for_sentiment(self, ticker, min_level, max_level):
        sentiment_score = self.sentiment_analyzer.get_sentiment(ticker)
        factor = 1 + (sentiment_score * 0.1)
        return round(min_level * factor), round(max_level * factor)

    def calculate_dynamic_lead_time(self, ticker):
        lead_times = self.stock_data.get_historical_lead_times(ticker)
        volatility = self.stock_data.get_market_volatility()
        return np.mean(lead_times) * (1 + volatility)

    def predict_future_volume(self, ticker):
        data = self.stock_data.get_historical_data(ticker)
        model = self.ml_models.get_volume_prediction_model()
        return model.predict(data)

    def calculate_adaptive_safety_stock(self, ticker):
        vol = self.stock_data.get_volatility(ticker)
        liq = self.stock_data.get_liquidity(ticker)
        return (vol / liq) * self.stock_data.get_average_daily_volume(ticker)

    def optimize_reorder_quantity(self, ticker, min_level):
        carrying_cost = self.stock_data.get_carrying_cost(ticker)
        ordering_cost = self.stock_data.get_ordering_cost(ticker)
        demand_forecast = self.predict_future_volume(ticker)
        return math.sqrt((2 * ordering_cost * demand_forecast) / carrying_cost)
        
    def sell(self, ticker, shares, price, min, max):
        self.loading_bar.update('sell', f'Selling {ticker}')
        # Incorporate sentiment analysis to adjust sell thresholds
        sentiment_score = self.sentiment_analyzer.get_sentiment(ticker)
        sentiment_factor = 1 + (sentiment_score * 0.1)
        adjusted_min = round(min * sentiment_factor)
        adjusted_max = round(max * sentiment_factor)

        # Use predictive models to determine optimal sell timing
        predicted_price = self.ml_models.get_price_prediction_model().predict(
            self.stock_data.get_historical_data(ticker)
        )

        if price >= predicted_price:
            self.new_recommendations.append(
                f'sell {shares} {ticker} at {price} with a min of {adjusted_min} and a max of {adjusted_max}'
            )
        else:
            self.new_recommendations.append(
                f'hold {ticker} - current price below predicted price'
            )
        self.loading_bar.update('sell', 'Sell action complete', 1, 'completed')

    def hold(self, ticker):
        self.loading_bar.update('hold', f'Holding {ticker}')
        # Incorporate market trend analysis to determine if holding is optimal
        market_trend = self.stock_data.get_market_trend(ticker)
        sentiment_score = self.sentiment_analyzer.get_sentiment(ticker)

        if market_trend == 'uptrend' and sentiment_score > 0:
            self.new_recommendations.append(
                f'hold {ticker} - positive trend and sentiment'
            )
        elif market_trend == 'downtrend' or sentiment_score < 0:
            self.new_recommendations.append(
                f'consider selling {ticker} - negative trend or sentiment'
            )
        else:
            self.new_recommendations.append(
                f'hold {ticker} - neutral conditions'
            )
        self.loading_bar.update('hold', 'Hold action complete', 1, 'completed')
        
    
    def research(self):
        self.loading_bar.update('research', 'Researching stock')
        """Get detailed research analysis using Perplexity."""
        if not self.action_inputs:
            return {"error": "No ticker provided", "sentiment": "neutral", "sentiment_score": 0.0}
            
        query = f"Provide detailed research analysis for {self.action_inputs}. Include financial metrics, competitive analysis, and growth prospects."
        response, status = self.perplexity.query_perplexity(query=query)
        if status:
            return self.perplexity.evaluate_response(response)
        self.loading_bar.update('research', 'Stock research complete', 1, 'completed')
        return {"error": "Failed to get research", "sentiment": "neutral", "sentiment_score": 0.0}

    def insight(self):
        self.loading_bar.update('insight', 'Gathering market insights')
        """Get market insights for a specific ticker using ChatGPT."""
        if not self.action_inputs:
            return {"error": "No ticker provided", "sentiment": "neutral", "sentiment_score": 0.0}
            
        query = f"Provide market insights and sentiment analysis for {self.action_inputs}. Focus on recent trends, news, and market sentiment."
        response, status = self.chatgpt.query_OpenAI(query=query)
        if status:
            return response
        self.loading_bar.update('insight', 'Insights gathered', 1, 'completed')
        return {"error": "Failed to get insights", "sentiment": "neutral", "sentiment_score": 0.0}

    def spinner_animation(self, plan_output=None, current_stage=None):
        self.loading_bar.update('spinner_animation', 'Animating spinner')
        if self.progress is None:
            self.initialize_progress()
        
        if current_stage:
            self.progress.update(self.overall_task, status=f"[bold blue]{current_stage}")
        
        if plan_output:
            self.console.print(Panel.fit(f"[bold magenta]Expected Plan:[/bold magenta]\n{plan_output}", border_style="magenta"))
            self.initialize_plan_tasks(plan_output)
        
        if current_stage:
            self.update_plan_progress(current_stage)

        # NEW: show most recent action
        self.console.print(f"[bold cyan]Most Recent Action:[/bold cyan] {self.most_recent_action}")
        self.loading_bar.update('spinner_animation', 'Spinner animation complete', 1, 'completed')

    def initialize_plan_tasks(self, plan_output):
        self.loading_bar.update('initialize_plan_tasks', 'Initializing plan tasks')
        plan_steps = plan_output.split('\n')
        for step in plan_steps:
            task_name = step.strip()
            if task_name:
                self.task_progresses[task_name] = self.progress.add_task(f"[cyan]{task_name}", total=100, status="Pending")
        self.loading_bar.update('initialize_plan_tasks', 'Plan tasks initialized', 1, 'completed')

    def update_plan_progress(self, current_stage):
        self.loading_bar.update('update_plan_progress', 'Updating plan progress')
        for task_name, task_id in self.task_progresses.items():
            if task_name in current_stage:
                self.progress.update(task_id, completed=100, status="[bold green]Completed[/bold green]")
            elif self.progress.tasks[task_id].completed == 0:
                self.progress.update(task_id, advance=50, status="[bold yellow]In Progress[/bold yellow]")
        
        completed_tasks = sum(1 for task in self.progress.tasks if task.completed == 100)
        if len(self.task_progresses) == 0:
            overall_progress = 0
        else: 
            overall_progress = (completed_tasks / len(self.task_progresses)) * 100
        self.progress.update(self.overall_task, completed=overall_progress)
        self.loading_bar.update('update_plan_progress', 'Plan progress updated', 1, 'completed')


    def update_task_progress(self, task, status, advance=0):
        self.loading_bar.update('update_task_progress', f'Updating task: {task}')
        if task not in self.task_progresses:
            self.task_progresses[task] = self.progress.add_task(f"[cyan]{task}", total=100, status="Pending")
        self.progress.update(self.task_progresses[task], advance=advance, status=status)
        self.loading_bar.update('update_task_progress', f'{task} updated', 1, 'completed')

    def initialize_progress(self):
        self.loading_bar.update('initialize_progress', 'Initializing progress tracking')
        self.console.print(Panel.fit("[bold cyan]Financial Agent Task Execution[/bold cyan]", border_style="blue"))
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold blue]{task.fields[status]}"),
            console=self.console,
            expand=True,
        )
        self.progress.start()
        self.overall_task = self.progress.add_task("[yellow]Overall Progress", total=100, status="In Progress")
        self.task_progresses = {}
        self.loading_bar.update('initialize_progress', 'Progress tracking initialized', 1, 'completed')

    def update_most_recent_action(self, action):
        self.loading_bar.update('update_most_recent_action', 'Updating most recent action')
        self.most_recent_action = action
        self.spinner_animation(current_stage=self.most_recent_action)
        self.loading_bar.update('update_most_recent_action', 'Action updated', 1, 'completed')

    def finalize_execution(self):
        self.loading_bar.update('finalize_execution', 'Finalizing execution')
        for task_id in self.task_progresses.values():
            self.progress.update(task_id, completed=100, status="[bold green]Completed[/bold green]")
        self.progress.update(self.overall_task, completed=100, status="[bold green]Completed[/bold green]")
        self.progress.stop()
        self.console.print(Panel.fit("[bold green]All financial tasks completed successfully.[/bold green]", border_style="green"))
        self.loading_bar.update('finalize_execution', 'Execution finalized', 1, 'completed')

    def return_stock_data(self):
        self.loading_bar.update('return_stock_data', 'Returning stock data')
        """
        Input .self
        |--> action_inputs
        |----> tickers
        |--> stock_data
        |----> 
        Output
        |--> Dictionary
        |----> {ticker: DataFrame}
        """
        self.loading_bar.update('return_stock_data', 'Stock data returned', 1, 'completed')
        return self.stock_data.get_stock_data(self.action_inputs)
        
    def perceive_environment(self, stock_data, perplexity_insights, chatgpt_insights):
        self.loading_bar.update('perceive_environment', 'Perceiving environment')
        self.environment = {
            "stock_data": stock_data,
            "perplexity_insights": perplexity_insights,
            "chatgpt_insights": chatgpt_insights,
        }
        self.loading_bar.update('Updating technical indicators')
        self._update_technical_indicators()
        self.loading_bar.update('perceive_environment', 'Environment perceived', 1, 'completed')


    def get_portfolio(self):
        self.loading_bar.update('get_portfolio', 'Loading portfolio data')
        try:
            # Use pathlib for portfolio path handling
            portfolio_files = list(PORTFOLIO_PATH.rglob("portfolio.csv"))
            if portfolio_files:
                # Get the most recent portfolio file
                latest_portfolio = max(portfolio_files, key=lambda p: p.stat().st_mtime)
                return pd.read_csv(latest_portfolio)
            return pd.DataFrame()  # Return empty DataFrame if no portfolio found
        except Exception as e:
            logging.error(f"Error reading portfolio: {e}")
            return pd.DataFrame()
        self.loading_bar.update('get_portfolio', 'Portfolio data loaded', 1, 'completed')

    def load_previous_actions(self):
        self.loading_bar.update('load_previous_actions', 'Loading previous actions')
        try:
            # Use pathlib for portfolio path handling
            actions_files = list(CONVERSION_PATH.rglob("actions.csv"))
            if actions_files:
                # Get the most recent actions file
                latest_actions = max(actions_files, key=lambda p: p.stat().st_mtime)
                self.loading_bar.update('load_previous_actions', 'Previous actions loaded', 1, 'completed')
                return pd.read_csv(latest_actions)
            return pd.DataFrame()  # Return empty DataFrame if no actions found
        except Exception as e:
            logging.error(f"Error reading actions: {e}")
            return pd.DataFrame()


    def plan_actions(self):
        self.loading_bar.update('plan_actions', 'Planning actions')
        self.loading_bar.update("Planning", "Generating plan")
        prompt = (
            f"""
            You are a financial agent that tries to make as much money as possible.
            You have {len(self.tickers)} valid tickers available to trade:
            {self.tickers}
            Here is your current portfolio:
            {self.get_portfolio()}
            Here are our previous interactions:
            {self.load_previous_actions()}
            You can request data for tickers and they will be provided to you.
            Step 1: Get starting information from options = [Chatgpt, Perplexity] 
            Step 1: random.choice(options)
            Step 1: Goal: Tickers that might be intersting to potentially research or invest in.
            To determine tickers do not guess randomly but instead go for an insight --> 'insight' --> Chatgpt
            For input query let the desired output >> 'insight' >> 'tickers'
            If sectors are given, dig deeper and ask for specific tickers with the previous output key information in the new query.
            Step 2: Mark your action as complete and then move to the next action
            Step 3: Once the tickers are loaded, you can now plan your actions
            Here you are to plan out how you would like to proceed with the tickers
            You have these actions:
                "buy": self.buy(), # Buy a stock --> ticker, shares, price --> API/Excute via function
                "sell": self.sell, # Sell a stock --> ticker, shares, price --> API/Excute via function
                "hold": self.hold, # Hold the stock --> ticker --> Database/Excute via function
                "research": self.research(), # Research the stock --> ticker --> 'Perplexity' 
                'insight': self.insight(), # Get insights on the stock --> ticker --> 'ChatGPT'
                "reason": self.reason() # Reiterate and start a new action --> Prompt
                'stockdata': self.get_stock_data() # Get stock data for array --> array of tickers output stock data
            What do you think you will want to do given each action can be done once per turn. Plan out as many turns as you want until your final action is complete.
            You may want to start with research and then move to insights and then to the final action.
            You may wannt to start with insight and then move to insights and then to the final action.
            Don't forget to emplement reasoning in your actions before trading as you can reason for as long as you'd like while the functions [Perplexity, Chatgpt] are costly but still not extremely expensive. Their ROI is high for data value but reasoning you should be able to do yourself.
            Step 4: Mark a set end point and the trades will be excuted ['buy', 'sell', 'hold]
            Step 5: Learn 
            What will I use your plan for?
            You will use your plan to excute it not perfectly but use it as a probabalistic guide to your next action.
            """
        )
        self.plan = self.ollama.make_plan(prompt)
        self.loading_bar.update("Planning", "Plan generated")
        self.loading_bar.update('plan_actions', 'Actions planned', 1, 'completed')

    def decide_action(self):
        self.loading_bar.update('decide_action', 'Deciding action')
        '''
        Phase 0: Plan out actions
            query ollama min 100 max ((60)(60))/5 times. 
            get stock data
        Phase 1: Get insights and research on which tickers might be interesting to trade
            query ollama
            query perplexity
            query chatgpt
            get stock data
        Phase 2: Save
            save information to the class variable for future use
        Phase 3:  Narrow down the amount of trades using the actions
            "research": self.research(), # Research the stock --> ticker --> 'Perplexity' 
            'insight': self.insight(), # Get insights on the stock --> ticker --> 'ChatGPT'
            "reason": self.reason() # Reiterate and start a new action --> Prompt
            'stockdata': self.get_stock_data() # Get stock data for array --> array of tickers output stock data
        Phase 4: Excute
            log the trades for the user to excute when they want.  
        Phase 5: Learn & Justify/Final notes
            Learn from the trades and justify the actions taken.
            Save the information for future use.
            Save notes on what could have gone better
        '''

        self.run_phase()
      
        self.loading_bar.update('decide_action', 'Action decided', 1, 'completed')
        return

    def run_phase(self):
        self.loading_bar.update('run_phase', 'Running phase')
        phases = {
            0: self.plan_actions(),
            1: self.pick_tickers(),
            2: self.save_info(),
            3: self.research(),
            4: self.excute_trades(),
            5: self.learn()
        }

        try:
            phases[self.phase]()
            self.loading_bar.update('run_phase', 'Phase run complete', 1, 'completed')
            return
        except KeyError:
            logging.error(f"Invalid phase: {self.phase}")
            return 
        
    def pick_tickers(self):
        self.loading_bar.update('pick_tickers', 'Picking tickers')
        self.update_most_recent_action("Initiating comprehensive stock selection process")
        
        try:
            initial_prompt = self._generate_initial_prompt()
            output = self.ollama.query_ollama(initial_prompt)
            self.previous_actions.append(output)
            
            selected_stocks = set(self.tickers)
            iteration_count = 0
            max_iterations = (60*60)//5  # Prevent infinite loops
            
            while iteration_count < max_iterations:
                self.loading_bar.update(f'Stock selection iteration {iteration_count + 1}')
                actions = self.get_actions(output)
                
                for action_name, action_func in actions:
                    if action_name == 'reason':
                        refined_output = self.reason(action_func)
                        self.previous_actions.append(refined_output)
                        
                        if self._should_stop_narrowing(refined_output, selected_stocks):
                            return self._finalize_stock_selection(selected_stocks)
                        
                        new_selection = self.extract_selected_stocks(refined_output)
                        if new_selection:
                            selected_stocks = self._evaluate_and_adjust_selection(new_selection, selected_stocks)
                    
                    elif action_name in ['insight', 'research', 'stockdata']:
                        action_result = action_func()
                        self._incorporate_action_result(action_name, action_result, selected_stocks)
                
                if 30 <= len(selected_stocks) <= 120:
                    if self._confirm_optimal_set(selected_stocks):
                        return self._finalize_stock_selection(selected_stocks)
                
                output = self.ollama.query_ollama(self._generate_refinement_prompt(selected_stocks))
                iteration_count += 1
            self.loading_bar.update('pick_tickers', 'Tickers picked', 1, 'completed')
            return self._handle_max_iterations_reached(selected_stocks)
        
        except Exception as e:
            logging.error(f"Error in pick_tickers: {e}")
            self.loading_bar.update('pick_tickers', 'Tickers picked', 1, 'completed')
            return self._emergency_stock_selection()
        

    def _emergency_stock_selection(self):
        logging.warning("Performing emergency stock selection")
        try:
            # Select a diverse set of stocks based on sectors and market cap
            sectors = self._get_stock_sectors()
            market_caps = self._get_stock_market_caps()
            
            emergency_selection = set()
            for sector in set(sectors.values()):
                sector_stocks = [ticker for ticker, s in sectors.items() if s == sector]
                if sector_stocks:
                    emergency_selection.add(random.choice(sector_stocks))
            
            # Ensure we have at least 30 stocks
            while len(emergency_selection) < 30:
                emergency_selection.add(random.choice(list(self.tickers)))
            
            # Limit to 120 stocks if we've exceeded that
            emergency_selection = list(emergency_selection)[:120]
            
            logging.info(f"Emergency selection complete. Selected {len(emergency_selection)} stocks.")
            return self._finalize_stock_selection(emergency_selection)
        except Exception as e:
            logging.error(f"Error in emergency stock selection: {e}")
            return list(self.tickers)[:120]  # Absolute fallback
    def _extract_selected_stocks(self, output):
        # Use regex to find stock tickers in the output
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        return set(re.findall(ticker_pattern, output))

    def _evaluate_and_adjust_selection(self, new_selection, current_selection):
        # Combine new and current selections, prioritizing diversity
        sectors = self._get_stock_sectors()
        market_caps = self._get_stock_market_caps()
        
        combined_selection = current_selection.union(new_selection)
        if len(combined_selection) <= 120:
            return combined_selection
        
        # If we have too many stocks, prioritize based on sector and market cap diversity
        diverse_selection = set()
        for sector in set(sectors.values()):
            sector_stocks = [ticker for ticker in combined_selection if sectors.get(ticker) == sector]
            diverse_selection.update(sector_stocks[:max(2, len(sector_stocks)//2)])
        
        # Fill remaining slots with a mix of large, mid, and small cap stocks
        remaining_slots = 120 - len(diverse_selection)
        if remaining_slots > 0:
            remaining_stocks = list(combined_selection - diverse_selection)
            random.shuffle(remaining_stocks)
            diverse_selection.update(remaining_stocks[:remaining_slots])
        
        return diverse_selection

    def _incorporate_action_result(self, action_name, action_result, selected_stocks):
        if action_name == 'insight' or action_name == 'research':
            # Extract potential stock picks from the insight or research
            mentioned_stocks = self._extract_selected_stocks(action_result)
            selected_stocks.update(mentioned_stocks)
        elif action_name == 'stockdata':
            # Use stock data to filter out stocks with undesirable characteristics
            for ticker, data in action_result.items():
                if self._is_stock_viable(data):
                    selected_stocks.add(ticker)
                elif ticker in selected_stocks:
                    selected_stocks.remove(ticker)

    def _is_stock_viable(self, stock_data):
        # Implement logic to determine if a stock is viable based on its data
        # This is a simplified example; you should expand this based on your criteria
        if len(stock_data) < 30:  # Ensure we have enough historical data
            return False
        
        latest_price = stock_data['Close'].iloc[-1]
        avg_volume = stock_data['Volume'].mean()
        
        if latest_price < 1:  # Avoid penny stocks
            return False
        if avg_volume < 100000:  # Ensure sufficient liquidity
            return False
        
        return True

    def _handle_timeout(self, selected_stocks):
        logging.warning("Stock selection timed out")
        if len(selected_stocks) < 30:
            return self._emergency_stock_selection()
        elif len(selected_stocks) > 120:
            # Prioritize stocks based on sector diversity and market cap
            return list(self._evaluate_and_adjust_selection(selected_stocks, set()))[:120]
        else:
            return self._finalize_stock_selection(selected_stocks)

    def _get_stock_sectors(self):
        # This should be implemented to return a dictionary of {ticker: sector}
        # For this example, we'll return a mock dictionary
        return {ticker: random.choice(['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']) for ticker in self.tickers}

    def _get_stock_market_caps(self):
        # This should be implemented to return a dictionary of {ticker: market_cap}
        # For this example, we'll return a mock dictionary
        return {ticker: random.choice(['Large', 'Mid', 'Small']) for ticker in self.tickers}

    def _generate_refinement_prompt(self, selected_stocks):
        return f"""
        We have currently selected {len(selected_stocks)} stocks.
        Our goal is to have between 30 and 120 high-potential stocks.
        Current selection: {', '.join(selected_stocks)}
        
        Please analyze this selection and suggest refinements:
        1. If we have fewer than 30 stocks, suggest additional stocks to consider.
        2. If we have more than 120 stocks, suggest which ones to remove and why.
        3. Ensure we maintain a diverse portfolio across different sectors and market caps.
        4. Consider current market trends and economic factors in your suggestions.
        
        Provide your analysis and suggestions in a clear, actionable format.
        """

    def _finalize_stock_selection(self, selected_stocks):
        self.active_tickers = list(selected_stocks)
        logging.info(f"Stock selection finalized. Selected {len(self.active_tickers)} stocks.")
        return self.active_tickers


    def _generate_initial_prompt(self):
        return f"""
        You are a financial agent aiming to maximize returns over a 20-year horizon. Your goal is to identify high-risk, high-reward stocks with the greatest long-term potential.
        Available tickers: {self.tickers}
        Current portfolio: {self.get_portfolio()}
        Previous actions: {self.previous_actions}
        
        Follow these steps to narrow down the stock selection:
        1. Categorize stocks into sectors and risk levels.
        2. Identify emerging trends and potentially disruptive technologies.
        3. Analyze each stock's growth potential, competitive advantage, and financial health.
        4. Consider macroeconomic factors and long-term industry outlooks.
        5. Assign a risk-reward score to each stock.
        6. Progressively eliminate lower-scoring stocks while maintaining diversification.
        7. Continue narrowing until you have between 30 and 120 high-potential stocks.
        8. Stop the process when you believe you have an optimal set of high-potential stocks within this range.

        Use these functions as needed:
        'insight' --> ${{ticker}}
        'research' --> ${{ticker}}
        'reason' --> query --> {r'${YOUR QUERY HERE}'}
        'stockdata' --> ${{ticker}}

        Begin your analysis and stock selection process. Provide detailed reasoning for your decisions.
        """

    def _should_stop_narrowing(self, refined_output, selected_stocks):
        if "STOP NARROWING" in refined_output.upper():
            if 30 <= len(selected_stocks) <= 120:
                self.active_tickers = list(selected_stocks)
                return True
        return False

    def _confirm_optimal_set(self, selected_stocks):
        confirmation = self.ollama.reason(f"We have narrowed down to {len(selected_stocks)} stocks. Is this an optimal set of high-potential stocks for our 20-year horizon?")
        return "OPTIMAL SET" in confirmation.upper()

    def _handle_max_iterations_reached(self, selected_stocks):
        self.active_tickers = list(selected_stocks)[:120]  # Take the first 120 if exceeded
        reasoning_prompt = """
        Maximum iterations reached in stock selection process. We need to finalize our selection.
        
        1. Evaluate the current set of selected stocks:
        - Analyze sector distribution
        - Assess risk profile diversity
        - Consider market cap distribution
        
        2. Identify any potential gaps or overexposure in the current selection
        
        3. Propose final adjustments to optimize the portfolio:
        - Stocks to potentially remove
        - Stocks to potentially add
        - Rationale for each adjustment
        
        4. Summarize the strengths and potential weaknesses of the final selection
        
        5. Suggest a strategy for monitoring and potentially rebalancing this selection in the future
        """
        reasoning_output = self.reason(reasoning_prompt)
        logging.info(f"Final stock selection reasoning:\n{reasoning_output}")
        
        # Implement final adjustments based on reasoning output
        self._apply_final_adjustments(reasoning_output)



    def research_and_insight(self):
        self.loading_bar.update('research_and_insight', 'Researching and gathering insights')
        self.update_most_recent_action("Researching and gathering insights")
        def generate_initial_prompt():
            return f"""
            You are a sophisticated financial agent tasked with maximizing returns through comprehensive analysis and data-driven decision-making. Analyze the following high-potential stocks:
            Active tickers: {self.active_tickers}
            Current portfolio: {self.get_portfolio()}

            For each stock, follow this advanced analysis framework:
            1. Fundamental Analysis:
            - Scrutinize financial statements (income statement, balance sheet, cash flow)
            - Evaluate key financial ratios (P/E, P/B, debt-to-equity, ROE, profit margins)
            - Assess company's competitive position, market share, and economic moat
            2. Technical Analysis:
            - Examine price trends, volume patterns, and advanced momentum indicators
            - Identify key support and resistance levels, and potential breakout points
            3. Industry and Market Analysis:
            - Evaluate sector performance, trends, and cyclicality
            - Consider macroeconomic factors and their impact on the industry
            4. Qualitative Factors:
            - Research management quality, corporate governance, and insider trading patterns
            - Assess company's innovation pipeline, R&D spending, and growth strategies
            5. Risk Assessment:
            - Identify and quantify potential risks (market, financial, operational, regulatory)
            - Evaluate ESG factors and their potential impact on long-term performance
            6. Valuation:
            - Determine intrinsic value using multiple methods (DCF, comparables, sum-of-parts)
            - Compare to current market price and assess margin of safety
            7. Investment Decision:
            - Synthesize all information to make a buy, sell, or hold decision
            - Provide a detailed rationale and specify position sizing

            Available functions:
            'insight' --> {r"${ticker}"}
            'research' --> {r"${ticker}"}
            'reason' --> query --> {r'${YOUR QUERY HERE}'}
            'stockdata' --> {r"${ticker}"}

            Use insight and research judiciously due to higher cost. Prioritize reasoning and stockdata for initial analysis.
            Previous plan: {self.plan}

            Begin your analysis. Provide detailed reasoning for each step and action.
            """

        def reason_with_advanced_chain_of_thought(action_func, chain_of_thought, current_ticker):
            financial_theory_hints = [
                "Capital Asset Pricing Model (CAPM) for risk-adjusted returns",
                "Multi-factor models (e.g., Fama-French Five-Factor Model)",
                "Beta and volatility analysis, including conditional and time-varying beta",
                "Dividend Discount Model and Gordon Growth Model for valuation",
                "Discounted Cash Flow (DCF) analysis with Monte Carlo simulations",
                "Relative valuation ratios (P/E, EV/EBITDA, P/B) with sector-specific considerations",
                "Efficient Market Hypothesis and behavioral finance theories",
                "Modern Portfolio Theory and post-modern portfolio theory",
                "Advanced technical indicators (Ichimoku Cloud, Fibonacci retracements)",
                "Fundamental analysis metrics (FCF yield, ROIC, economic value added)",
                "Industry-specific metrics and benchmarks",
                "Options-based analysis (implied volatility, put-call parity)"
            ]
            
            prompt = f"""
            Analyzing: {current_ticker}
            Previous analysis steps: {' -> '.join(chain_of_thought[-5:])}
            
            Consider these advanced financial concepts and methods:
            {', '.join(financial_theory_hints)}
            
            Based on the previous analysis and these concepts:
            1. What is the next critical step in our analysis?
            2. What specific calculations or evaluations should we perform to gain deeper insights?
            3. How does this step contribute to our overall investment thesis and risk management?
            4. Are there any potential biases, hidden risks, or overlooked factors in our current analysis?
            5. How can we validate or challenge our current assumptions using alternative data sources or models?
            6. What contrarian viewpoints should we consider to stress-test our analysis?

            Provide a detailed explanation for your reasoning and next steps, incorporating quantitative and qualitative factors.
            """
            
            response = action_func(prompt)
            chain_of_thought.append(f"{current_ticker}: {response[:100]}...")
            return response

        def apply_sentiment_analysis(refined_output, ticker):
            sentiment, score = self.perplexity.analyze_sentiment(refined_output)
            sentiment_interpretation = f"""
            Sentiment Analysis for {ticker}:
            - Overall Sentiment: {sentiment}
            - Sentiment Score: {score}
            
            Interpretation:
            - Bullish (score > 0.05): Consider this positive sentiment, but verify with fundamental data and assess if it's priced in.
            - Bearish (score < -0.05): Investigate reasons for negative sentiment. Is it a short-term reaction or indicative of long-term issues?
            - Neutral (-0.05 <= score <= 0.05): Market uncertainty detected. Look for potential catalysts and analyze options market for implied volatility.

            Questions to address:
            1. How does this sentiment align with or contradict our fundamental and technical analysis?
            2. Is there a divergence between sentiment and stock performance that we can exploit?
            3. How does this sentiment compare to sector peers, and what might be driving any differences?
            4. Are there any upcoming events or announcements that could shift this sentiment?

            Incorporate this sentiment analysis into our overall investment thesis, considering its reliability and potential impact on short-term price movements.
            """
            return sentiment_interpretation

        def review_decisions(decisions, results):
            review_prompt = f"""
            We need to review our investment decisions for the following stocks:

            {decisions}

            Consider the following factors in your review:
            1. Portfolio Diversification: Assess sector allocation and risk exposure.
            2. Risk-Adjusted Returns: Evaluate expected returns in relation to potential risks.
            3. Market Timing: Consider current market conditions and their impact on our decisions.
            4. Contrarian Opportunities: Identify any potential contrarian plays that may have been overlooked.
            5. Correlation Analysis: Examine how the chosen stocks correlate with each other and the broader market.
            6. Liquidity Considerations: Ensure that our positions are appropriately sized for the stocks' liquidity.
            7. Macro Environment: Re-evaluate decisions in light of current and projected macroeconomic conditions.
            8. Catalyst Identification: Confirm that we've identified potential near-term and long-term catalysts for each position.

            For each stock, provide:
            1. A confirmation or revision of the original decision (BUY/SELL/HOLD).
            2. Any adjustments to position sizing.
            3. A brief explanation of your reasoning, especially if the decision has changed.

            Present your review in a clear, tabular format.
            """
            
            review_output = self.ollama.query_ollama(review_prompt)
            reviewed_decisions = self.extract_decisions(review_output)
            
            return reviewed_decisions

        prompt = generate_initial_prompt()
        output = self.ollama.query_ollama(prompt)
        self.previous_actions.append(output)
        actions = self.get_actions(output)

        results = {}
        decisions = {}
        query_count = 0
        research_or_insight_used = set()
        chain_of_thought = []

        for ticker in self.active_tickers:
            ticker_query_count = 0
            ticker_data_collected = False
            ticker_decision_made = False

            self.loading_bar.update(f"Analyzing {ticker}")

            while not (ticker_data_collected and ticker_decision_made):
                for action_name, action_func in actions:
                    if action_name == 'reason':
                        refined_output = reason_with_advanced_chain_of_thought(action_func, chain_of_thought, ticker)
                        self.previous_actions.append(refined_output)
                        
                        sentiment_interpretation = apply_sentiment_analysis(refined_output, ticker)
                        self.previous_actions.append(sentiment_interpretation)
                        
                        actions = self.get_actions(refined_output)
                        query_count += 1
                        ticker_query_count += 1

                        if ticker_query_count >= 20:
                            decision_prompt = f"""
                            Based on our comprehensive analysis of {ticker}, including fundamental, technical, and sentiment factors:
                            1. Summarize the key findings from our analysis, highlighting both bullish and bearish factors.
                            2. Weigh the pros and cons of investing in this stock, considering risk-adjusted returns.
                            3. Provide a clear DECISION (BUY, SELL, or HOLD) with a confidence level (Low, Medium, High).
                            4. Explain the rationale behind this decision, addressing potential risks, growth opportunities, and catalysts.
                            5. Suggest a position size or adjustment based on the overall portfolio strategy, considering volatility and correlation with other holdings.
                            6. Specify any risk management measures, such as stop-loss levels or hedging strategies.

                            Format your decision as follows:
                            DECISION: [BUY/SELL/HOLD]
                            CONFIDENCE: [Low/Medium/High]
                            RATIONALE: [Your detailed explanation]
                            POSITION: [Suggested position size or adjustment]
                            RISK MANAGEMENT: [Specific risk management measures]
                            """
                            decision_output = action_func(decision_prompt)
                            new_decision = self.extract_decisions(decision_output)
                            if ticker in new_decision:
                                decisions[ticker] = new_decision[ticker]
                                ticker_decision_made = True

                    elif action_name in ['insight', 'research', 'stockdata']:
                        if not ticker_data_collected:
                            result = action_func(ticker)
                            results[ticker] = result
                            ticker_data_collected = True
                            if action_name in ['insight', 'research']:
                                research_or_insight_used.add(ticker)

                    if query_count >= 100 and len(research_or_insight_used) >= len(self.active_tickers) // 3:
                        break

                if query_count >= 100 and len(research_or_insight_used) >= len(self.active_tickers) // 3:
                    break

            if query_count >= 100 and len(research_or_insight_used) >= len(self.active_tickers) // 3:
                break

        if set(decisions.keys()) == set(self.active_tickers):
            reviewed_decisions = review_decisions(decisions, results)
            
            confirmation_prompt = f"""
            We have reviewed and potentially revised our decisions for all {len(self.active_tickers)} stocks in our analysis. Here are the final decisions:

            {reviewed_decisions}

            Please perform the following:
            1. Summarize the overall portfolio strategy based on these final decisions.
            2. Confirm that we have addressed any potential overexposure to specific sectors or risk factors.
            3. Verify that the suggested position sizes align with our risk management guidelines.
            4. Ensure that our portfolio maintains proper diversification and aligns with our investment objectives.
            5. Identify any remaining concerns or areas that may require ongoing monitoring.

            Respond with CONFIRM if you agree with the final decisions and strategy, or REVIEW if you think we should reconsider any aspects.
            """
            confirmation = self.ollama.query_ollama(confirmation_prompt)
            if "CONFIRM" in confirmation.upper():
                return reviewed_decisions
            else:
                # If further review is needed, we can implement an iterative review process
                return review_decisions(reviewed_decisions, results)

        self.loading_bar.update('research_and_insight', 'Research and insights gathered', 1, 'completed')
        return decisions
    
    def get_actions(self, output):
        self.loading_bar.update('get_actions', 'Getting actions')
        """Parse actions from output text."""
        actions_mapping = {
            'insight': self.insight,
            'research': self.research,
            'reason': self.reason,
            'stockdata': self.stock_data.get_stock_data
        }

        actions = []
        for line in output.splitlines():
            for key, action in actions_mapping.items():
                if key in line.lower():
                    param_start = line.find("$") + 1 if "$" in line else -1
                    param_end = line.find("}") if "}" in line else len(line)
                    if param_start >= 0:
                        param = line[param_start:param_end].strip()
                        self.action_inputs = param
                        actions.append((key, action))

        self.loading_bar.update('get_actions', 'Actions retrieved', 1, 'completed')
        return actions

    def stockdata(self, ticker):
        self.loading_bar.update('stockdata', f'Getting data for {ticker}')
        """Retrieve historical stock data for analysis."""
        if ticker in self.stock_data.stock_data:
            data = self.stock_data.get_stock_data([ticker])
            self.loading_bar.update('stockdata', 'Data retrieved', 1, 'completed')
            return {ticker: data[ticker]}
        self.loading_bar.update('stockdata', 'Data retrieval failed', 1, 'completed')
        return {"error": f"No data available for {ticker}"}
    
        # def decide_action(self): #FIXED: HORRIBLE CODE. The one function the AI did
        #     actions = []
        #     for ticker, data in self.environment["stock_data"].items():
        #         sentiment_score = self._analyze_sentiment(ticker)
        #         technical_score = self._analyze_technicals(data)
        #         prediction = self._predict_price(data)
                
        #         overall_score = (sentiment_score + technical_score + prediction) / 3
                
        #         if overall_score > 0.7:  # Strong buy signal
        #             action = self._calculate_buy_amount(ticker, data)
        #             if action:
        #                 actions.append(action)
        #         elif overall_score < 0.3:  # Strong sell signal
        #             action = self._calculate_sell_amount(ticker, data)
        #             if action:
        #                 actions.append(action)
            
        #     return actions if actions else [{"action": "hold"}]

    def _analyze_sentiment(self, ticker):
        self.loading_bar.update('_analyze_sentiment', 'Analyzing sentiment')
        self.loading_bar.update(f'Analyzing sentiment for {ticker}')
        chatgpt_sentiment = self.environment["chatgpt_insights"].get(ticker, "")
        perplexity_sentiment = self.environment["perplexity_insights"].get(ticker, "")
        
        sentiment_score = 0.5  # Neutral by default
        if "bullish" in chatgpt_sentiment:
            sentiment_score += 0.25
        if "bearish" in chatgpt_sentiment:
            sentiment_score -= 0.25
        
        self.loading_bar.update('_analyze_sentiment', 'Sentiment analysis complete', 1, 'completed')
        return max(0, min(1, sentiment_score))

    def _analyze_technicals(self, data):
        self.loading_bar.update('_analyze_technicals', 'Analyzing technical indicators')
        latest = data.iloc[-1]
        score = 0
        
        # SMA crossover
        if latest['SMA_20'] > latest['SMA_50']:
            score += 0.2
        else:
            score -= 0.2
        
        # RSI
        if 30 < latest['RSI'] < 70:
            score += 0.1
        elif latest['RSI'] <= 30:
            score += 0.2  # Oversold
        else:
            score -= 0.2  # Overbought
        
        # MACD
        if latest['MACD'] > latest['Signal']:
            score += 0.2
        else:
            score -= 0.2
        
        self.loading_bar.update('_analyze_technicals', 'Technical analysis complete', 1, 'completed')
        return max(0, min(1, score + 0.5))

    def _predict_price(self, data):
        self.loading_bar.update('_predict_price', 'Predicting future price')
        self.loading_bar.update('Predicting future price')
        X = data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].values
        y = data['Close'].shift(-1).dropna().values
        
        X = self.scaler.fit_transform(X[:-1])
        y = y[:-1]
        
        self.model.fit(X, y)
        
        next_day = self.scaler.transform(data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1))
        prediction = self.model.predict(next_day)[0]
        
        current_price = data['Close'].iloc[-1]
        predicted_return = (prediction - current_price) / current_price
        
        self.loading_bar.update('_predict_price', 'Price prediction complete', 1, 'completed')
        return max(0, min(1, (predicted_return + 0.1) / 0.2))

    def act(self, decisions):
        self.loading_bar.update('act', 'Executing decisions')
        for decision in decisions:
            self.loading_bar.update(f'Executing {decision.get("action")} action for {decision.get("ticker")}')
            if decision.get("action") == "hold":
                print("Action: Hold. No transactions made.")
                ticker = decision["ticker"]
                self.hold(ticker)
                continue
                
            ticker = decision["ticker"]
            price = decision["price"]
            shares = decision["shares"]

            if decision["action"] == "buy":
                cost = shares * price
                if self.balance >= cost:
                    self.balance -= cost
                    self.portfolio[ticker] = self.portfolio.get(ticker, 0) + shares
                    self.history.append(f"Bought {shares} shares of {ticker} at {price}")
                    print(f"Action: Bought {shares} shares of {ticker} at {price}")
                    self.buy(ticker, shares, price)
                else:
                    print(f"Insufficient funds to buy {shares} shares of {ticker}")

            elif decision["action"] == "sell":
                if self.portfolio.get(ticker, 0) >= shares:
                    self.balance += shares * price
                    self.portfolio[ticker] -= shares
                    if self.portfolio[ticker] == 0:
                        del self.portfolio[ticker]
                    self.history.append(f"Sold {shares} shares of {ticker} at {price}")
                    print(f"Action: Sold {shares} shares of {ticker} at {price}")
                else:
                    print(f"Insufficient shares to sell {shares} shares of {ticker}")
        self.loading_bar.update('act', 'Decisions executed', 1, 'completed')

    def learn(self):
        self.loading_bar.update('learn', 'Learning from recent trades')
        self.loading_bar.update('Learning from recent trades')
        recent_trades = self.history[-10:]
        if recent_trades:
            profit_ratio = sum(1 for trade in recent_trades if "Sold" in trade) / len(recent_trades)
            if profit_ratio > 0.6:
                self.risk_tolerance = min(0.05, self.risk_tolerance * 1.1)
            elif profit_ratio < 0.4:
                self.risk_tolerance = max(0.01, self.risk_tolerance * 0.9)
        self.loading_bar.update('learn', 'Learning complete', 1, 'completed')

    def get_portfolio_value(self):
        self.loading_bar.update('get_portfolio_value', 'Calculating portfolio value')
        self.loading_bar.update('Calculating portfolio value')
        total_value = self.balance
        for ticker, shares in self.portfolio.items():
            if ticker in self.environment["stock_data"]:
                current_price = self.environment["stock_data"][ticker]['Close'].iloc[-1]
                total_value += shares * current_price
        self.loading_bar.update('get_portfolio_value', 'Portfolio value calculated', 1, 'completed')
        return total_value

    def get_performance_report(self):
        self.loading_bar.update('get_performance_report', 'Generating performance report')
        self.loading_bar.update('Generating performance report')
        current_value = self.get_portfolio_value()
        initial_value = 100
        total_return = (current_value - initial_value) / initial_value * 100
        
        report = f"Performance Report:\n"
        report += f"Current Portfolio Value: ${current_value:.2f}\n"
        report += f"Total Return: {total_return:.2f}%\n"
        report += f"Current Holdings:\n"
        
        for ticker, shares in self.portfolio.items():
            if ticker in self.environment["stock_data"]:
                current_price = self.environment["stock_data"][ticker]['Close'].iloc[-1]
                position_value = shares * current_price
                report += f"  {ticker}: {shares} shares, Value: ${position_value:.2f}\n"
        
        self.loading_bar.update('get_performance_report', 'Report generated', 1, 'completed')
        return report
    
    def reason(self, query=None):
        self.loading_bar.update('reason', 'Starting chain-of-thought reasoning')
        self.loading_bar.update('Processing reasoning query')
        if query == None:
            query = self.action_inputs
        response = self.ollama.query_ollama(query)
        self.loading_bar.update('reason', 'Reasoning complete', 1, 'completed')
        return response
    
    def test_outputs(self):
        self.loading_bar.update('test_outputs', 'Testing outputs')
        self.loading_bar.update('Testing outputs')
        if self.plan == "":
            raise "Plan not generated."
        if self.active_tickers == []:
            raise "Tickers not selected."
        if self.previous_actions == {}:
            raise "No previous actions."
        if self.current_actions == {}:
            raise "No current actions."
        if self.tickers == []:
            raise "No tickers available."
        self.loading_bar.update('test_outputs', 'Outputs tested', 1, 'completed')

    def execute_trades(self):
        self.loading_bar.update('execute_trades', 'Executing trades')
        self.update_most_recent_action("Executing trades")
        self.act(self.decisons)
        self.spinner_animation(current_stage="ExecuteTrades")
        self.loading_bar.update('execute_trades', 'Trades executed', 1, 'completed')



    def begin(self):
        self.loading_bar.update('begin', 'Starting agent')
        self.loading_bar.initialize(total_tasks=6)  # 6 phases in total
        
        phases = [
            ("Strategic Planning", self.plan_actions),
            ("Comprehensive Stock Selection", self.pick_tickers),
            ("Rigorous Validation", self.test_outputs),
            ("In-depth Research and Insight Gathering", self.research_and_insight),
            ("Precise Trade Execution", self.execute_trades),
            ("Continuous Learning and Adaptation", self.learn)
        ]
        
        self.loading_bar.set_expected_steps([phase[0] for phase in phases])
        
        for phase_name, phase_function in phases:
            self.loading_bar.update(phase_name)
            try:
                phase_result = phase_function()
           #     self._evaluate_phase_result(phase_name, phase_result)
            except Exception as e:
                logging.error(f"Error in {phase_name} phase: {e}")
           #     self._handle_phase_error(phase_name, e)
                raise e
            finally:
                self.loading_bar.update(phase_name)
                self.loading_bar.remove_completed_step(phase_name)
        
        self.loading_bar.stop()
        self._generate_comprehensive_report()
        self.finalize_execution()
        self.loading_bar.update('begin', 'Agent begun', 1, 'completed')



def main():
    logging.basicConfig(level=logging.INFO)

    agent = Agent()
    agent.begin()

if __name__ == "__main__":
    main()
