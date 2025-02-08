import re
import openai
import os
import sys
import time
import logging
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List, Any

# Added missing imports (preserving existing ones)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from pandas import Timestamp

import pandas as pd
from cachetools import LRUCache
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
from loading import DynamicLoadingBar
import json
from openai import OpenAI
import asyncio
import functools
from datetime import datetime, timedelta
from pandas import DataFrame
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
from cryptography.fernet import Fernet
import aiohttp

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
RANKING_PATH = DATA_PATH / 'Rankings'

# Add Data Management to system path
sys.path.append(str(PROJECT_ROOT))
from Data.DataManagement.maintenance import update_data

logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize the global loading bar
loading_bar = DynamicLoadingBar()

class ChatGPT4o:
    def __init__(self):
        self.tests = 1
        self.checkmark = "✅"
        self.crossmark = "❌"
        self.api_key_path = API_KEYS_PATH / 'OpenAI.txt'
        self.default_role = (
            "You are a financial analyst with extensive experience in the stock market. \n You provide insights based on your current knowledge."
        )
        self.client = None  # Will be initialized in read_api
        self.sia = self._initialize_sentiment_analyzer()
        self.read_api()  # Initialize OpenAI client
        loading_bar.dynamic_update("Instance initialization complete", operation="ChatGPT4o.__init__")

    def read_api(self):
        """Initialize OpenAI client with API key."""
        loading_bar.dynamic_update("Reading API key", operation="read_api")
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.read().strip()
                if not api_key:
                    raise ValueError("OpenAI API key is empty")
                self.client = OpenAI(api_key=api_key)  # Initialize client with 2025 pattern
                loading_bar.dynamic_update("API key loaded successfully", operation="read_api")
                return api_key
        except (FileNotFoundError, PermissionError) as e:
            loading_bar.dynamic_update("API key file error", operation="read_api")
            raise ValueError(f"API key file error: {e}")
        except Exception as e:
            loading_bar.dynamic_update("Unexpected error reading API key", operation="read_api")
            raise ValueError(f"Unexpected error reading API key: {e}")

    def query_OpenAI(self, model="", query="", max_tokens=150, temperature=0.5, role=""):
        """Query OpenAI API with 2025 features."""
        loading_bar.dynamic_update("Preparing OpenAI query", operation="query_OpenAI")

        # Validate and set model
        model = self.default_model if not model else model
        supported_models = {
            "gpt-4o": {"max_tokens": 128000},  # 128k context window
            "gpt-4o-realtime-preview": {"max_tokens": 32000},  # For real-time interactions
            "gpt-4": {"max_tokens": 8000},  # Legacy support
            "gpt-3.5-turbo": {"max_tokens": 4000},  # Legacy support
        }
        
        if model not in supported_models:
            raise ValueError(f"Unsupported model: {model}. Available models: {list(supported_models.keys())}")
            
        role = self.default_role if not role else role

        def test_input_validity(query_input):
            """Validate query input."""
            result = bool(query_input.strip())
            description = f"{'Valid' if result else 'Invalid'} query provided"
            print(f"Test {self.tests}: {self.checkmark if result else self.crossmark} {description}")
            self.tests += 1
            return result

        if not test_input_validity(query):
            return "Invalid query provided. Please check the input.", False

        try:
            # Using 2025 OpenAI client pattern
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": query}
                ],
                max_tokens=min(max_tokens, supported_models[model]["max_tokens"]),
                temperature=temperature
            )
            output = response.choices[0].message.content
            status = True
            
        except openai.BadRequestError as e:
            output, status = f"Invalid request: {e}", False
        except openai.AuthenticationError:
            output, status = "Authentication failed. Check your API key.", False
        except openai.RateLimitError:
            output, status = "Rate limit exceeded. Please try again later.", False
        except Exception as e:
            output, status = f"An unexpected error occurred: {e}", False
        
        loading_bar.dynamic_update("OpenAI query complete", operation="query_OpenAI")
        return output, status

    def analyze_sentiment(self, statement):
        """Analyze sentiment with enhanced accuracy."""
        loading_bar.dynamic_update("Starting sentiment analysis", operation="analyze_sentiment")
        
        try:
            # First get VADER sentiment
            sentiment_scores = self.sia.polarity_scores(statement)
            compound_score = sentiment_scores['compound']
            
            # Then get GPT-4o's analysis for better accuracy
            sentiment_prompt = f"""
            Analyze the market sentiment in this statement. Consider technical factors, market psychology, and economic indicators:
            "{statement}"
            
            Rate the sentiment on a scale from -1 (extremely bearish) to +1 (extremely bullish).
            Respond with only the numerical score.
            """
            
            gpt_score, success = self.query_OpenAI(
                model="gpt-4o",
                query=sentiment_prompt,
                max_tokens=10,
                temperature=0.3
            )
            
            if success:
                try:
                    gpt_score = float(gpt_score.strip())
                    # Combine VADER and GPT-4o scores with more weight to GPT-4o
                    final_score = (compound_score + 2 * gpt_score) / 3
                except ValueError:
                    final_score = compound_score
            else:
                final_score = compound_score
            
            # Determine sentiment category
            if final_score >= 0.05:
                sentiment = "Bullish"
            elif final_score <= -0.05:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
            
            # Normalize to 0-1 range for consistency
            normalized_score = (final_score + 1) / 2
            
            loading_bar.dynamic_update("Sentiment analysis complete", operation="analyze_sentiment")
            return sentiment, normalized_score
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            loading_bar.dynamic_update("Error in sentiment analysis", operation="analyze_sentiment")
            return "Neutral", 0.5

    def evaluate_response(self, response):
        """Evaluate response with enhanced analysis."""
        loading_bar.dynamic_update("Evaluating response", operation="evaluate_response")
        
        try:
            if not response:
                return "No response received."
            if "error" in str(response).lower():
                return "The response contains an error."
            
            # Get sentiment analysis
            sentiment, score = self.analyze_sentiment(str(response))
            
            # Extract key insights using GPT-4o
            insight_prompt = f"""
            Analyze this market-related text and extract key insights:
            "{str(response)}"
            
            Focus on:
            1. Technical indicators
            2. Market trends
            3. Risk factors
            4. Action recommendations
            
            Format as JSON with these keys: technical, trends, risks, recommendations
            """
            
            insights, success = self.query_OpenAI(
                model="gpt-4o",
                query=insight_prompt,
                max_tokens=5000,
                temperature=0.3
            )
            
            result = {
                "text": str(response),
                "sentiment": sentiment,
                "sentiment_score": score,
                "insights": insights if success else "Could not extract insights"
            }
            
            loading_bar.dynamic_update("Response evaluation complete", operation="evaluate_response")
            return result
            
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            loading_bar.dynamic_update("Error in response evaluation", operation="evaluate_response")
            return "Error during evaluation."

    def _initialize_sentiment_analyzer(self):
        """Initialize VADER sentiment analyzer with error handling."""
        loading_bar.dynamic_update("Initializing sentiment analyzer", operation="initialize_sentiment")
        try:
            # Download VADER lexicon quietly
            nltk.download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer()
            loading_bar.dynamic_update("Sentiment analyzer initialized", operation="initialize_sentiment")
            return analyzer
        except Exception as e:
            logging.error(f"Error initializing sentiment analyzer: {e}")
            loading_bar.dynamic_update("Error initializing sentiment analyzer", operation="initialize_sentiment")
            raise RuntimeError(f"Failed to initialize sentiment analyzer: {e}")

class Perplexity:
    def __init__(self, config: dict = None):
        """Initialize Perplexity API client with configuration."""
        from pathlib import Path
        from typing import Dict, Tuple, Union, Optional
        from tenacity import retry, stop_after_attempt, wait_exponential
        import hashlib
        from cryptography.fernet import Fernet
        import aiohttp
        
        self.tests = 1
        self.checkmark = "✅"
        self.crossmark = "❌"
        
        # Configuration handling
        self.config = config or {
            "default_model": "pplx-7b-online",
            "max_tokens": 1000,
            "temperature": 0.7,
            "api_key_path": API_KEYS_PATH / 'Perplexity.txt'
        }
        
        self.api_key_path = self.config["api_key_path"]
        self.default_model = self.config["default_model"]
        self.default_role = (
            "You are an AI financial analyst providing market insights based on extensive data analysis."
        )
        
        # Initialize components
        self.api_key = self._encrypt_key(self.read_api())
        self.sia = self._initialize_sentiment_analyzer()  # Add back sentiment analyzer
        self.cache = {}
        loading_bar.dynamic_update("Instance initialization complete", operation="Perplexity.__init__")

    def _encrypt_key(self, key: str) -> bytes:
        """Encrypt API key for secure storage."""
        try:
            cipher_key = Fernet.generate_key()
            cipher = Fernet(cipher_key)
            return cipher.encrypt(key.encode())
        except Exception as e:
            logging.error(f"Error encrypting API key: {e}")
            return key.encode()  # Fallback to encoded but unencrypted

    def _create_cache_key(self, query: str, model: str, max_tokens: int, temperature: float) -> str:
        """Create unique cache key with collision resistance."""
        key_components = f"{query}-{model}-{max_tokens}-{temperature}"
        return hashlib.sha256(key_components.encode()).hexdigest()

    def read_api(self) -> str:
        """Read API key with improved error handling."""
        loading_bar.dynamic_update("Reading Perplexity API key", operation="read_api")
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.read().strip()
                if not api_key:
                    raise ValueError("Perplexity API key is empty")
                return api_key
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")
        finally:
            loading_bar.dynamic_update("Perplexity API key read", operation="read_api")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def query_perplexity(self, model: str = "", query: str = "", 
                        max_tokens: int = 300, temperature: float = 0.7) -> Tuple[str, bool]:
        """Query Perplexity API with retry logic and improved validation."""
        loading_bar.dynamic_update("Querying Perplexity", operation="query_perplexity")
        
        # Input validation and defaults
        model = model or self.default_model
        if not self._validate_content(query):
            return "Content validation failed", False
            
        cache_key = self._create_cache_key(query, model, max_tokens, temperature)
        if cache_key in self.cache:
            return self.cache[cache_key], True

        if not self._validate_inputs(model, query, max_tokens, temperature):
            return "Invalid input parameters", False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key.decode()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.default_role},
                    {"role": "user", "content": query}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            output = result['choices'][0]['message']['content']
            self.cache[cache_key] = (output, True)
            return output, True
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API Error: {str(e)}"
            logging.error(error_msg)
            return error_msg, False
        finally:
            loading_bar.dynamic_update("Perplexity query complete", operation="query_perplexity")

    async def query_async(self, session: aiohttp.ClientSession, query: str) -> Tuple[str, bool]:
        """Async query support for improved scalability."""
        try:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key.decode()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.default_model,
                    "messages": [
                        {"role": "system", "content": self.default_role},
                        {"role": "user", "content": query}
                    ]
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'], True
                return f"Error: {response.status}", False
        except Exception as e:
            return f"Error: {str(e)}", False

    def _validate_inputs(self, model: str, query: str, max_tokens: int, temperature: float) -> bool:
        """Validate all input parameters with updated model list."""
        valid = True
        # Validate model using _validate_model
        model_valid, model_msg = self._validate_model(model)
        if not model_valid:
            print(f"Validation {self.tests}: FAIL - {model_msg}")
            self.tests += 1
            valid = False
        else:
            print(f"Validation {self.tests}: PASS - {model_msg}")
            self.tests += 1

        # Validate query
        if bool(query.strip()):
            print(f"Validation {self.tests}: PASS - Valid query")
            self.tests += 1
        else:
            print(f"Validation {self.tests}: FAIL - Empty query")
            self.tests += 1
            valid = False

        # Check and adjust max_tokens if out of range
        if not (10 <= max_tokens <= 1000000):
            print(f"Validation {self.tests}: FAIL - Max tokens {max_tokens} out of range. Setting to default value 300.")
            max_tokens = 300
            self.tests += 1
        else:
            print(f"Validation {self.tests}: PASS - Max tokens {max_tokens}")
            self.tests += 1

        # Check and adjust temperature if out of range
        if not (0.0 <= temperature <= 1.0):
            print(f"Validation {self.tests}: FAIL - Temperature {temperature} out of range. Setting to default value 0.7.")
            temperature = 0.7
            self.tests += 1
        else:
            print(f"Validation {self.tests}: PASS - Temperature {temperature}")
            self.tests += 1

        return valid

    def _validate_model(self, model: str) -> Tuple[bool, str]:
        """Validate against current Perplexity models."""
        valid_models = [
            "pplx-7b-online",
            "pplx-70b-online", 
            "llama-2-70b-chat",
            "codellama-34b-instruct"
        ]
        valid = model in valid_models
        return (valid, f"{'Valid' if valid else 'Invalid'} model: {model}")

    def _validate_content(self, query: str) -> bool:
        """Validate content for safety and policy compliance."""
        blocked_terms = [
            "malware", "exploit", "breach", "hack", "crack",
            "illegal", "password", "credential"
        ]
        return not any(term in query.lower() for term in blocked_terms)

    def evaluate_response(self, response: Union[str, dict]) -> dict:
        """Evaluate response with improved error handling."""
        try:
            if isinstance(response, dict) and 'error' in response:
                return response
                
            text = response if isinstance(response, str) else response.get('text', '')
            if not text:
                return {"error": "Empty response", "sentiment": "neutral"}
            
            sentiment, score = self.analyze_sentiment(text)
            return {
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": score,
                "error": None
            }
            
        except Exception as e:
            logging.error(f"Evaluation error: {str(e)}")
            return {
                "error": f"Evaluation failed: {str(e)}",
                "sentiment": "neutral",
                "sentiment_score": 0.0
            }

    def analyze_sentiment(self, statement: str) -> Tuple[str, float]:
        """Analyze sentiment with improved accuracy."""
        try:
            scores = self.sia.polarity_scores(statement)
            compound = scores['compound']
            
            if compound >= 0.05:
                return "bullish", compound
            if compound <= -0.05:
                return "bearish", compound
            return "neutral", compound
            
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return "neutral", 0.0

    def _initialize_sentiment_analyzer(self):
        """Initialize VADER sentiment analyzer"""
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()

class StockData:
    def __init__(self):
        self.data_path = STOCK_DATA_PATH
        self.maintain_stock_data()
        self.stock_data = self.read_stock_data()  

        self.tickers = list(self.stock_data.keys())
            
        loading_bar.dynamic_update("Stock data initialization complete", operation="StockData.__init__")

    def get_stock_data(self, tickers):
        loading_bar.dynamic_update("Fetching stock data", operation="get_stock_data")
        data = {}
        for ticker in tickers:
            data[ticker] = self.stock_data[ticker]
        loading_bar.dynamic_update("Fetched stock data", operation="get_stock_data")
        return data
    
    def read_stock_data(self):
        loading_bar.dynamic_update("Reading all stock data", operation="read_stock_data")
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
            loading_bar.dynamic_update("Stock data read complete", operation="read_stock_data")
            return stock_data
            
        except Exception as e:
            logging.error(f"Error reading stock data: {e}")
            return {}

    def maintain_stock_data(self) -> Optional[bool]:
        loading_bar.dynamic_update("Maintaining stock data", operation="maintain_stock_data")
        """
        Run update_data() from the Data Management module.
        
        Returns:
            Optional[bool]: True if update was successful, False if an error occurred, None if update_data() doesn't return a status.
        """
        try:


            logging.info("Starting data update process")
            result = update_data()
            logging.info("Data update process completed")
            loading_bar.dynamic_update("Maintenance complete", operation="maintain_stock_data")
            return result if isinstance(result, bool) else None
        
        except ImportError as e:
            logging.error(f"Failed to import update_data: {e}")
            return False
        except Exception as e:
            logging.error(f"An error occurred during data update: {e}")
            return False

class Ollama:
    def __init__(self):
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
        loading_bar.dynamic_update("Ollama initialization complete", operation="Ollama.__init__")

    def reason(self, query):
        loading_bar.dynamic_update("Initiating reasoning", operation="reason")
        loading_bar.dynamic_update("Initiating chain of thought reasoning", operation="reason")
        
        chain_of_thought = [
            f"Initial query: {query}",
            "Step 1: Analyze the query and break it down into key components",
            "Step 2: Identify relevant financial concepts and data points",
            "Step 3: Consider multiple perspectives and potential outcomes",
            "Step 4: Evaluate risks and opportunities",
            "Step 5: Synthesize insights and form a conclusion"
        ]
        
        for step in chain_of_thought[1:]:
            loading_bar.dynamic_update(f'Reasoning: {step}', operation="reason")
            step_query = f"{step}\nBased on the previous steps and the initial query, what insights can we derive?"
            step_response = self.ollama.query_ollama(step_query)
            chain_of_thought.append(f"Output: {step_response}")
        
        final_reasoning = "\n".join(chain_of_thought)
        loading_bar.dynamic_update("Finalizing chain of thought reasoning", operation="reason")
        loading_bar.dynamic_update("Reasoning complete", operation="reason")
        os.clear()
        print("===================================================================")
        print(final_reasoning)
        return final_reasoning

    def query_ollama(self, prompt: str) -> str:
        loading_bar.dynamic_update("Querying Ollama", operation="query_ollama")
        try:
            response = ollama.chat(
                model='llama3.2:1b',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial agent making decisions based on market analysis.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                stream=False
            )
            loading_bar.dynamic_update("Ollama query complete", operation="query_ollama")
            print("Formatted content:", response['message']['content'])
            return response['message']['content']
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
        loading_bar.dynamic_update("Querying Ollama for plan", operation="query_ollama_for_plan")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial agent. Always respond with ACTION: <action> PARAMS: <parameters>.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                stream=False
            )
            
            content = response['message']['content']
            print(content)
            loading_bar.dynamic_update("Received response from Ollama", operation="query_ollama_for_plan")
            return content

        except Exception as e:
            logging.error(f"Error querying Ollama: {e}")
            return "ACTION: reason PARAMS: Error occurred, need to reassess"

    def make_plan(self, query: str) -> str:
        loading_bar.dynamic_update("Making plan", operation="make_plan")
        if not query:
            raise ValueError("Query cannot be empty")

        loading_bar.dynamic_update("Initializing plan", operation="make_plan")
        plan = []
        iteration_count = 0
        max_iterations = 5  # Reduced max iterations for faster planning
        
        def get_plan():
            return "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

        try:
            while (iteration_count < max_iterations):
                iteration_count += 1
                loading_bar.dynamic_update(f"Plan iteration {iteration_count}/{max_iterations}", operation="make_plan")
                
                reasoning_prompt = f"""
                Query: {query}
                Current plan: {get_plan()}
                
                Provide the next step in the plan using one of these action formats:
                ACTION: insight PARAMS: <ticker>
                ACTION: research PARAMS: <ticker>
                ACTION: reason PARAMS: <query>
                ACTION: stockdata PARAMS: <ticker>
                
                Your response MUST start with 'ACTION:' followed by one of these actions.
                """
                
                response = self.query_ollama_for_plan(reasoning_prompt)
                if not response.strip():
                    continue
                    
                # Validate action format
                if self._validate_action_format(response):
                    plan.append(response.strip())
                    loading_bar.dynamic_update(f"Added valid action: {response.strip()}", operation="make_plan")
                else:
                    # Convert invalid response to reasoning action
                    formatted_action = f"ACTION: reason PARAMS: {response.strip()}"
                    plan.append(formatted_action)
                    loading_bar.dynamic_update(f"Converted to reasoning action", operation="make_plan")
                
                # Check if we have enough valid steps
                if len(plan) >= 3:  # Minimum 3 steps for a complete plan
                    break

            loading_bar.dynamic_update("Plan generation complete", operation="make_plan")
            print(f'Final plan:\n{get_plan()}')
            return f"Final plan:\n{get_plan()}"

        except Exception as e:
            logging.error(f"Error in make_plan: {e}")
            return """
            ACTION: reason PARAMS: Analyze market conditions
            ACTION: stockdata PARAMS: SPY
            ACTION: insight PARAMS: Market overview
            """

    def _validate_action_format(self, response: str) -> bool:
        """Validate if response follows the required action format."""
        action_pattern = r'^ACTION:\s*(insight|research|reason|stockdata|select):([^>]+)>'
        return bool(re.match(action_pattern, response.strip(), re.IGNORECASE))

    def update_task_progress(self, task, status, advance=0):
        loading_bar.dynamic_update(f'Updating task: {task}', operation="update_task_progress")
        
        # Create task if it doesn't exist
        if task not in self.task_progresses:
            self.task_progresses[task] = self.progress.add_task(
                f"[cyan]{task}", 
                total=100,
                status="Pending"
            )
        
        # Update task progress
        task_id = self.task_progresses[task]
        current_progress = self.progress.tasks[task_id].completed
        
        # Only update if we're actually making progress
        if advance > 0 or status != self.progress.tasks[task_id].fields['status']:
            self.progress.update(
                task_id,
                advance=advance,
                status=status,
                completed=min(current_progress + advance, 100)
            )
            
            # Update overall progress
            self._update_overall_progress()
        
        loading_bar.dynamic_update(f'Task updated: {task} - {status}', operation="update_task_progress")

    def _update_overall_progress(self):
        """Update overall progress based on individual task completion."""
        if not self.task_progresses:
            return
            
        total_progress = sum(
            self.progress.tasks[task_id].completed 
            for task_id in self.task_progresses.values()
        )
        avg_progress = total_progress / len(self.task_progresses)
        
        self.progress.update(
            self.overall_task,
            completed=avg_progress,
            status=f"Overall Progress: {avg_progress:.0f}%"
        )

class PromptManager:
    def __init__(self, agent):
        self.agent = agent

    def get_stock_selection_prompt(self):
        """Generates prompt for initial stock selection."""
        return f"""
        You are a financial agent aiming to maximize returns over a 20-year horizon. Your goal is to identify high-risk, high-reward stocks with the greatest long-term potential.
        Available tickers: {self.agent.tickers}
        Current portfolio: {self.agent.get_portfolio()}
        Previous actions: {self.agent.previous_actions}
        Selected Stocks: {self.agent.portfolio_manager.pending_stocks}
        
        Follow these steps to narrow down the stock selection:
        1. Categorize stocks into sectors and risk levels.
        2. Identify emerging trends and potentially disruptive technologies.
        3. Analyze each stock's growth potential, competitive advantage, and financial health.
        4. Consider macroeconomic factors and long-term industry outlooks.
        5. Assign a risk-reward score to each stock.
        6. Progressively eliminate lower-scoring stocks while maintaining diversification.
        7. Continue narrowing until you have between 30 and 120 high-potential stocks.
        8. When you have your final selection, use the <select:"TICKERS"> action.

        Available actions:
        <insight:TICKER>     # Get insights for a specific stock
        <research:TICKER>    # Get detailed research
        <reason:"QUERY">     # Analyze specific aspects
        <stockdata:TICKER>   # Get market data
        <select:"TICKERS">   # Submit your final stock selection (comma-separated, e.g. "AAPL,MSFT,GOOGL")

        Use the select action only when you have your final list of 30-120 stocks.
        For intermediate analysis, use the other actions to gather information.
        """

    def get_research_prompt(self):
        """Generates prompt for research and analysis."""
        return f"""
        You are a sophisticated financial agent tasked with maximizing returns through comprehensive analysis and data-driven decision-making. Analyze the following high-potential stocks:
        Active tickers: {self.agent.active_tickers}
        Current portfolio: {self.agent.get_portfolio()}

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

        Respond with one of these actions:
        'insight' --> ${r'{ticker and/or query}'}
        'research' --> ${r'{ticker and/or query}'}
        'reason' --> query --> {r'${YOUR QUERY HERE}'}
        'stockdata' --> ${r'{ticker or tickers}'}

        Use insight and research judiciously due to higher cost. Prioritize reasoning and stockdata for initial analysis.
        Previous plan: {self.agent.plan}

        What your going to want to do is use reasoning a lot to do math and various finance related things and give yourself all the info needed to countiue inside the query.
        """

    def get_reasoning_prompt(self, context):
        """Generates prompt for reasoning steps with specific context."""
        return f"""
        Based on the current context and analysis:
        {context}

        Consider the following aspects:
        1. Market conditions and trends
        2. Technical indicators and price action
        3. Fundamental company metrics
        4. Risk factors and potential catalysts
        5. Portfolio impact and diversification

        Previous actions and decisions:
        {self.agent.previous_actions[-5:] if self.agent.previous_actions else "No previous actions"}

        Current portfolio state:
        {self.agent.get_portfolio()}

        Provide detailed reasoning for the next steps, considering:
        - Risk management and position sizing
        - Market timing and entry/exit points
        - Portfolio balance and sector exposure
        - Long-term strategic alignment
        
        Available functions:
        'insight' --> ${r'{ticker}'}
        'research' --> ${r'{ticker}'}
        'reason' --> query --> {r'${YOUR QUERY HERE}'}
        'stockdata' --> ${r'{ticker}'}

        Analyze the situation and provide clear, actionable reasoning for the next steps.
        """

class ActionPhase:
    def __init__(self, phase_name: str, prompt: str, output: str):
        self.phase_name = phase_name
        self.prompt = prompt
        self.output = output
        self.timestamp = time.time()
        
    def format_action_template(self):
        """Return formatted action template for consistent responses"""
        return """
        Valid action formats:
        <insight:TICKER>     # For stock insights (1-5 letter ticker)
        <research:TICKER>    # For detailed research (1-5 letter ticker)
        <reason:"QUERY">     # For analysis (quote multi-word queries)
        <stockdata:TICKER>   # For market data (1-5 letter ticker)
        <select:"TICKERS">   # For stock selection (comma-separated tickers, e.g. "AAPL,MSFT,GOOGL")
        """

    def __str__(self):
        return (
            f"Phase: {self.phase_name}\n"
            f"Prompt: {self.prompt}\n"
            f"Output: {self.output}\n"
            f"Action Templates:\n{self.format_action_template()}"
        )

class ActionHistory:
    def __init__(self):
        self.phases: list[ActionPhase] = []
        self.current_phase = None
        
        # Updated pattern to include select action
        self.action_pattern = r'''
            <                           # Opening bracket
            (?P<action>                 # Action type
                insight|research|reason|stockdata|select
            )
            :                          # Separator
            (?P<param>                 # Parameter
                "[^"]*"               # Quoted string
                |                     # or
                [A-Z,]{1,300}        # Ticker symbols (increased length for multiple tickers)
            )
            >                         # Closing bracket
        '''

    def start_phase(self, phase_name: str, prompt: str):
        """Start a new phase with action templates"""
        self.current_phase = phase_name
        phase = ActionPhase(phase_name, prompt, "")
        self.phases.append(phase)
        return len(self.phases) - 1

    def add_output(self, output: str, phase_idx: int = None):
        if phase_idx is None:
            if self.phases:
                phase_idx = len(self.phases) - 1
            else:
                raise ValueError("No active phase to add output to")
        
        self.phases[phase_idx].output = output

    def get_context(self, window_size: int = 3) -> str:
        """Get formatted context with action templates"""
        recent_phases = self.phases[-window_size:] if self.phases else []
        context = []
        
        for phase in recent_phases:
            context.append(
                f"[{phase.phase_name}]\n"
                f"Prompt: {phase.prompt[:200]}...\n"
                f"Output: {phase.output[:200]}...\n"
                f"Valid Actions:\n{phase.format_action_template()}\n"
                f"{'='*50}"
            )
        
        return "\n".join(context)

    def validate_action(self, action: str, param: str) -> bool:
        """Validate action parameters"""
        if action in ['insight', 'research', 'stockdata']:
            return bool(re.match(r'^[A-Z]{1,5}$', param))
        elif action == 'reason':
            return bool(re.match(r'^"[A-Za-z0-9\s\-_]{1,100}"$', param))
        elif action == 'select':
            # Remove quotes if present
            param = param.strip('"')
            # Validate comma-separated ticker format
            tickers = param.split(',')
            return all(
                re.match(r'^[A-Z]{1,5}$', ticker.strip()) 
                for ticker in tickers
            ) and len(tickers) <= 120
        return False

    def clear(self):
        self.phases = []
        self.current_phase = None

# Update Agent class initialization
class Agent:
    def __init__(self, initial_balance=100, risk_tolerance=0.02):

        self.balance = initial_balance
        
        # Inialize components
        self.chatgpt = ChatGPT4o()
        self.perplexity = Perplexity()
        self.stock_data = StockData()
        self.ollama = Ollama()
        self.market_data_agent = MarketDataAgent()  # Add MarketDataAgent
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Spinner
        self.console = Console()
        self.tasks = { 
            }
        
        # Get the Stock info from the Stock Data
        self.stock_data.read_stock_data()

        # Initalize starting values
        self.portfolio = {}
        self.new_recommendations = []
        self.history = []  # Initialize empty history list
        self.trade_history = []  # Separate list for trade history
        self.risk_tolerance = risk_tolerance  # Initial risk tolerance
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
            "stockdata": self.stockdata  # Changed from lambda to method
        }

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

        self.prompt_manager = PromptManager(self)
        
        self.action_history = ActionHistory()
        self.portfolio_manager = Portfolio()
        
        # Initialize progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            expand=True
        )
        self.overall_task = self.progress.add_task(
            "[cyan]Overall Progress", 
            total=100,
            status="Starting"
        )
        self.task_progresses = {}
        
    def buy(self, ticker, shares, price, min=0, max=0):
        if ticker.empty():
            return
        loading_bar.dynamic_update(f'Buying {ticker}', operation="buy")
        if min == max == 0:
            loading_bar.dynamic_update(f'Calculating min and max', operation="buy")
            loading_bar.dynamic_update(f'Predicting future volume', operation="buy")
            predicted_volume = self.predict_future_volume(ticker)
            loading_bar.dynamic_update(f'Predicting future volume', operation="buy")
            loading_bar.dynamic_update(f'Calculating dynamic lead time', operation="buy")
            dynamic_lead_time = self.calculate_dynamic_lead_time(ticker)
            loading_bar.dynamic_update(f'Calculating dynamic lead time', operation="buy")
            loading_bar.dynamic_update(f'Calculating adaptive safety stock', operation="buy")
            safety_stock = self.calculate_adaptive_safety_stock(ticker)
            loading_bar.dynamic_update(f'Calculating adaptive safety stock', operation="buy")
            loading_bar.dynamic_update(f'Calculating base min', operation="buy")
            base_min = (predicted_volume * dynamic_lead_time) + safety_stock
            loading_bar.dynamic_update(f'Calculating base min', operation="buy")
            loading_bar.dynamic_update(f'Optimizing reorder quantity', operation="buy")
            reorder_qty = self.optimize_reorder_quantity(ticker, base_min)
            loading_bar.dynamic_update(f'Optimizing reorder quantity', operation="buy")
            loading_bar.dynamic_update(f'Adjusting levels for sentiment', operation="buy")
            sentiment_adjusted_min, sentiment_adjusted_max = self.adjust_levels_for_sentiment(
                ticker,
                base_min,
                base_min + reorder_qty
            )
            loading_bar.dynamic_update(f'Adjusting levels for sentiment', operation="buy")
            loading_bar.dynamic_update(f'Finalizing buy', operation="buy")
            min, max = round(sentiment_adjusted_min), round(sentiment_adjusted_max)
            loading_bar.dynamic_update(f'Finalizing buy', operation="buy")
        loading_bar.dynamic_update(f'Saving information', operation="buy")
        self.new_recommendations.append(f"buy {shares} {ticker} at {price} with a min of {min} and a max of {max}")
        loading_bar.dynamic_update(f'Saving information', operation="buy")
        loading_bar.dynamic_update(f'Buy action completed: Buy {shares} of {ticker} at {price} with a min of {min} and a max of {max}', operation="buy")

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
        loading_bar.dynamic_update(f'Selling {ticker}', operation="sell")
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
        loading_bar.dynamic_update("Sell action complete", operation="sell")

    def hold(self, ticker):
        loading_bar.dynamic_update(f'Holding {ticker}', operation="hold")
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
        loading_bar.dynamic_update("Hold action complete", operation="hold")
        
    
    def research(self):
        loading_bar.dynamic_update("Researching stock", operation="research")
        """Get detailed research analysis using Perplexity."""
        if not self.action_inputs:
            return {"error": "No ticker provided", "sentiment": "neutral", "sentiment_score": 0.0}
            
        query = f"Provide detailed research analysis for query: {self.action_inputs}. Include financial metrics, competitive analysis, and growth prospects."
        response, status = self.perplexity.query_perplexity(query=query)
        if status:
            return self.perplexity.evaluate_response(response)
        loading_bar.dynamic_update("Stock research complete", operation="research")
        return {"error": "Failed to get research", "sentiment": "neutral", "sentiment_score": 0.0}

    def insight(self):
        loading_bar.dynamic_update("Gathering market insights", operation="insight")
        """Get market insights for a specific ticker using ChatGPT."""
        if not self.action_inputs:
            return {"error": "No ticker provided", "sentiment": "neutral", "sentiment_score": 0.0}
            
        query = f"Provide market insights and sentiment analysis for {self.action_inputs}. Focus on recent trends, news, and market sentiment."
        response, status = self.chatgpt.query_OpenAI(query=query)
        if status:
            return response
        loading_bar.dynamic_update("Insights gathered", operation="insight")
        return {"error": "Failed to get insights", "sentiment": "neutral", "sentiment_score": 0.0}

    def spinner_animation(self, plan_output=None, current_stage=None):
        loading_bar.dynamic_update("Animating spinner", operation="spinner_animation")
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
        loading_bar.dynamic_update("Spinner animation complete", operation="spinner_animation")

    def initialize_plan_tasks(self, plan_output):
        loading_bar.dynamic_update("Initializing plan tasks", operation="initialize_plan_tasks")
        plan_steps = plan_output.split('\n')
        for step in plan_steps:
            task_name = step.strip()
            if task_name:
                self.task_progresses[task_name] = self.progress.add_task(f"[cyan]{task_name}", total=100, status="Pending")
        loading_bar.dynamic_update("Plan tasks initialized", operation="initialize_plan_tasks")

    def update_plan_progress(self, current_stage):
        loading_bar.dynamic_update("Updating plan progress", operation="update_plan_progress")
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
        loading_bar.dynamic_update("Plan progress updated", operation="update_plan_progress")


    def update_task_progress(self, task, status, advance=0):
        loading_bar.dynamic_update(f'Updating task: {task}', operation="update_task_progress")
        if task not in self.task_progresses:
            self.task_progresses[task] = self.progress.add_task(f"[cyan]{task}", total=100, status="Pending")
        self.progress.update(self.task_progresses[task], advance=advance, status=status)
        loading_bar.dynamic_update(f'{task} updated', operation="update_task_progress")

    def initialize_progress(self):
        loading_bar.dynamic_update("Initializing progress tracking", operation="initialize_progress")
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
        loading_bar.dynamic_update("Progress tracking initialized", operation="initialize_progress")

    def update_most_recent_action(self, action):
        loading_bar.dynamic_update("Updating most recent action", operation="update_most_recent_action")
        self.most_recent_action = action
        self.spinner_animation(current_stage=self.most_recent_action)
        loading_bar.dynamic_update("Action updated", operation="update_most_recent_action")

    def finalize_execution(self):
        """Finalize the execution and clean up."""
        loading_bar.dynamic_update("Finalizing execution", operation="finalize")
        
        try:
            # Update final status in loading bar
            loading_bar.dynamic_update(
                "✅ Execution complete - Check Reports directory for details", 
                operation="finalize"
            )
            
            # Log completion
            logging.info("Agent execution completed successfully")
            
            # Save final state if needed
            if hasattr(self, 'portfolio_manager'):
                self.portfolio_manager.save_state()
            
            # Final cleanup
            self._cleanup()
            
        except Exception as e:
            logging.error(f"Error in finalize_execution: {e}")
            loading_bar.dynamic_update(
                f"❌ Error during finalization: {str(e)}", 
                operation="finalize"
            )

    def _cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Clean up any temporary files or resources
            pass
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def return_stock_data(self):
        loading_bar.dynamic_update("Returning stock data", operation="return_stock_data")
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
        loading_bar.dynamic_update("Stock data returned", operation="return_stock_data")
        return self.stock_data.get_stock_data(self.action_inputs)
        
    def perceive_environment(self, stock_data, perplexity_insights, chatgpt_insights):
        loading_bar.dynamic_update("Perceiving environment", operation="perceive_environment")
        self.environment = {
            "stock_data": stock_data,
            "perplexity_insights": perplexity_insights,
            "chatgpt_insights": chatgpt_insights,
        }
        loading_bar.dynamic_update("Updating technical indicators", operation="perceive_environment")
        self._update_technical_indicators()
        loading_bar.dynamic_update("Environment perceived", operation="perceive_environment")


    def get_portfolio(self):
        loading_bar.dynamic_update("Loading portfolio data", operation="get_portfolio")
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
        loading_bar.dynamic_update("Portfolio data loaded", operation="get_portfolio")

    def load_previous_actions(self):
        loading_bar.dynamic_update("Loading previous actions", operation="load_previous_actions")
        try:
            # Use pathlib for portfolio path handling
            actions_files = list(CONVERSION_PATH.rglob("actions.csv"))
            if actions_files:
                # Get the most recent actions file
                latest_actions = max(actions_files, key=lambda p: p.stat().st_mtime)
                loading_bar.dynamic_update("Previous actions loaded", operation="load_previous_actions")
                return pd.read_csv(latest_actions)
            return pd.DataFrame()  # Return empty DataFrame if no actions found
        except Exception as e:
            logging.error(f"Error reading actions: {e}")
            return pd.DataFrame()


    def plan_actions(self):
        loading_bar.dynamic_update("Planning actions", operation="plan_actions")
        loading_bar.dynamic_update("Generating plan", operation="Planning")
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
        loading_bar.dynamic_update("Plan generated", operation="Planning")
        loading_bar.dynamic_update("Actions planned", operation="plan_actions")

    def decide_action(self):
        loading_bar.dynamic_update("Deciding action", operation="decide_action")
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
      
        loading_bar.dynamic_update("Action decided", operation="decide_action")
        return

    def run_phase(self):
        loading_bar.dynamic_update("Running phase", operation="run_phase")
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
            loading_bar.dynamic_update("Phase run complete", operation="run_phase")
            rank = Ranking(self.stock_data, self, self.perplexity, self.chatgpt) # TODO: Fix this
            return rank.rank_tickers()
        except KeyError:
            logging.error(f"Invalid phase: {self.phase}")
            return 
        
    def pick_tickers(self):
        loading_bar.dynamic_update("Starting stock selection process", operation="pick_tickers")
        
        def analyze_market_sectors():
            # Get initial market overview from ChatGPT
            market_query = """
            Analyze the current market conditions and identify:
            1. The most promising market sectors
            2. Key trends and developments
            3. Potential growth areas
            4. Risk factors to consider
            
            Format your response to highlight specific sectors and explain WHY they are interesting.
            """
            market_insight, status = self.chatgpt.query_OpenAI(query=market_query)
            if not status:
                logging.error("Failed to get market insight from ChatGPT")
                return None
                
            # Get detailed sector analysis from Perplexity
            sector_query = f"""
            Based on this market overview:
            {market_insight}
            
            Provide detailed analysis of each mentioned sector:
            1. Key players and market leaders
            2. Emerging companies with potential
            3. Market share distribution
            4. Growth metrics and valuations
            """
            sector_analysis, status = self.perplexity.query_perplexity(query=sector_query)
            if not status:
                logging.error("Failed to get sector analysis from Perplexity")
                return None
                
            return {'market_insight': market_insight, 'sector_analysis': sector_analysis}
            
        def identify_potential_tickers(sector_info):
            # Extract potential tickers from sector analysis
            initial_prompt = f"""
            Based on this market and sector analysis:
            
            Market Overview:
            {sector_info['market_insight']}
            
            Sector Analysis:
            {sector_info['sector_analysis']}
            
            From our available tickers:
            {self.tickers}
            
            Identify potential stocks to analyze, considering:
            1. Market leaders in promising sectors
            2. Emerging players with growth potential
            3. Undervalued companies with strong fundamentals
            4. Innovation leaders and disruptive companies
            
            Format your response as:
            <select:"TICKER1,TICKER2,..."> with 40-60 initial tickers for deeper analysis
            """
            
            response = self.ollama.query_ollama(initial_prompt)
            actions = self.get_actions(response)
            selected_tickers = set()
            
            for action_name, _ in actions:
                if action_name == 'select':
                    tickers = self.action_inputs.strip('"').split(',')
                    selected_tickers.update(ticker.strip() for ticker in tickers)
            
            return list(selected_tickers)
            
        def analyze_selected_tickers(tickers):
            analyzed_tickers = {}
            
            for ticker in tickers:
                loading_bar.dynamic_update(f"Analyzing {ticker}", operation="pick_tickers.analysis")
                
                # Get stock data
                stock_data = self.stock_data.get_stock_data([ticker])
                if not stock_data or ticker not in stock_data:
                    continue
                    
                data_df = stock_data[ticker]
                recent_data = data_df.tail(5)
                
                # Get company insights from ChatGPT
                company_query = f"""
                Analyze {ticker} considering:
                - Recent price: ${float(recent_data['Close'].iloc[-1]):.2f}
                - Price change: {float(recent_data['Close'].pct_change().iloc[-1] * 100):.2f}%
                - Volume: {int(recent_data['Volume'].iloc[-1]):,}
                
                Focus on:
                1. Company's market position
                2. Growth potential
                3. Competitive advantages
                4. Risks and challenges
                """
                company_insight, _ = self.chatgpt.query_OpenAI(query=company_query)
                
                # Get detailed analysis from Perplexity
                analysis_query = f"""
                Provide detailed analysis of {ticker} including:
                1. Financial health and metrics
                2. Industry position and market share
                3. Growth trajectory and catalysts
                4. Risk assessment
                """
                detailed_analysis, _ = self.perplexity.query_perplexity(query=analysis_query)
                
                analyzed_tickers[ticker] = {
                    'stock_data': stock_data[ticker].to_dict(),
                    'company_insight': company_insight,
                    'detailed_analysis': detailed_analysis
                }
            
            return analyzed_tickers
            
        def make_final_selection(analyzed_tickers):
            selection_prompt = f"""
            We've analyzed {len(analyzed_tickers)} potential stocks.
            
            For each stock, we have:
            {json.dumps({t: {
                'insight': a['company_insight'][:200] + "...",
                'analysis': a['detailed_analysis'][:200] + "..."
            } for t, a in analyzed_tickers.items()}, indent=2)}
            
            Select the most promising 30-120 stocks considering:
            1. Diversification across sectors
            2. Mix of established leaders and growth potential
            3. Risk-reward balance
            4. Market conditions and trends
            
            Provide your selection as:
            <select:"TICKER1,TICKER2,...">
            
            Include a brief explanation of why each stock was selected.
            """
            
            response = self.ollama.query_ollama(selection_prompt)
            actions = self.get_actions(response)
            
            for action_name, _ in actions:
                if action_name == 'select':
                    final_tickers = self.action_inputs.strip('"').split(',')
                    if 30 <= len(final_tickers) <= 120:
                        self.active_tickers = [t.strip() for t in final_tickers]
                        return True
            
            return False
        
        try:
            # Step 1: Analyze market sectors
            loading_bar.dynamic_update("Analyzing market sectors", operation="pick_tickers.sectors")
            sector_info = analyze_market_sectors()
            if not sector_info:
                raise ValueError("Failed to analyze market sectors")
            
            # Step 2: Identify potential tickers
            loading_bar.dynamic_update("Identifying potential tickers", operation="pick_tickers.identify")
            potential_tickers = identify_potential_tickers(sector_info)
            if not potential_tickers:
                raise ValueError("Failed to identify potential tickers")
            
            # Step 3: Analyze selected tickers
            loading_bar.dynamic_update("Analyzing selected tickers", operation="pick_tickers.analyze")
            analyzed_tickers = analyze_selected_tickers(potential_tickers)
            if not analyzed_tickers:
                raise ValueError("Failed to analyze selected tickers")
            
            # Step 4: Make final selection
            loading_bar.dynamic_update("Making final selection", operation="pick_tickers.select")
            if not make_final_selection(analyzed_tickers):
                raise ValueError("Failed to make final selection")
            
            loading_bar.dynamic_update(
                f"✅ Selected {len(self.active_tickers)} stocks for analysis",
                operation="pick_tickers.complete"
            )
            return self.active_tickers
            
        except Exception as e:
            logging.error(f"Error in pick_tickers: {e}")
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
        """
        Generates initial prompt for stock selection process.
        Used in pick_tickers() method.
        """
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
        'insight' --> ${r'{ticker}'}
        'research' --> ${r'{ticker}'}
        'reason' --> query --> {r'${YOUR QUERY HERE}'}
        'stockdata' --> ${r'{ticker}'}

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
        loading_bar.dynamic_update("Starting research and insight analysis", operation="research_and_insight")
        
        results = {}
        chain_of_thought = []
        
        def analyze_ticker(ticker):
            loading_bar.dynamic_update(f"Starting analysis chain for {ticker}", operation="analyze_ticker")
            
            # Get stock data first to have it available throughout the analysis
            stock_data = self.stock_data.get_stock_data([ticker])
            if not stock_data or ticker not in stock_data:
                logging.error(f"Could not get stock data for {ticker}")
                return None
                
            # Format stock data summary for conversation
            data_df = stock_data[ticker]
            recent_data = data_df.tail(5)  # Last 5 days
            data_summary = {
                'latest_close': float(recent_data['Close'].iloc[-1]),
                'price_change': float(recent_data['Close'].pct_change().iloc[-1] * 100),
                'volume': int(recent_data['Volume'].iloc[-1]),
                'avg_volume': int(recent_data['Volume'].mean()),
                'high_52w': float(data_df['High'].rolling(window=252).max().iloc[-1]),
                'low_52w': float(data_df['Low'].rolling(window=252).min().iloc[-1])
            }
            
            ticker_results = {
                'analysis_chain': [],
                'stock_data': data_summary,
                'raw_data': stock_data[ticker].to_dict()  # Store raw data for detailed analysis
            }
            
            # Initial reasoning about what we need to know, now including stock data context
            initial_reasoning = self.ollama.query_ollama(f"""
                I need to analyze {ticker} thoroughly. Here's the current market data:
                
                Current Price: ${data_summary['latest_close']:.2f}
                24h Change: {data_summary['price_change']:.2f}%
                Volume: {data_summary['volume']:,}
                Avg Volume: {data_summary['avg_volume']:,}
                52W High: ${data_summary['high_52w']:.2f}
                52W Low: ${data_summary['low_52w']:.2f}
                
                Let's break this down step by step:
                1. What does this price action and volume tell us?
                2. What fundamental data should we look at next?
                3. What technical indicators would be most relevant?
                4. What market sentiment factors should we consider?
                5. What specific questions should we ask ChatGPT and Perplexity?
                
                For each step, explain WHY we need this information and HOW it will help our decision.
                Then, specify which action to take next using one of these formats:
                <stockdata:TICKER> - Get market data
                <research:TICKER> - Get detailed Perplexity research
                <insight:TICKER> - Get ChatGPT market insights
                <reason:"QUERY"> - Analyze specific aspects
            """)
            
            chain_of_thought.append(f"Initial reasoning for {ticker} with market data: {initial_reasoning}")
            
            # Parse and execute the suggested actions
            actions = self.get_actions(initial_reasoning)
            
            for action_name, action_func in actions:
                try:
                    self.action_inputs = ticker
                    result = action_func()
                    if result:  # Only store result if we got one
                        ticker_results[action_name] = result
                        ticker_results['analysis_chain'].append(f"{action_name}: {str(result)[:200]}...")
                    else:
                        logging.warning(f"No result from {action_name} for {ticker}")
                        continue
                    
                    # After each action, reason about what we learned and what to do next
                    reflection = self.ollama.query_ollama(f"""
                        Based on the {action_name} results for {ticker}:
                        
                        Current Market Context:
                        - Price: ${data_summary['latest_close']:.2f} ({data_summary['price_change']:.2f}%)
                        - Volume: {data_summary['volume']:,} vs Avg: {data_summary['avg_volume']:,}
                        - 52W Range: ${data_summary['low_52w']:.2f} - ${data_summary['high_52w']:.2f}
                        
                        Last Action Result:
                        {str(result)[:500] if result else "No result"}
                        
                        1. What key insights did we gain from the {action_name}?
                        2. How does this align with the market data we have?
                        3. What questions remain unanswered?
                        4. What additional data would help us make a better decision?
                        5. Should we:
                           a) Get more market data (stockdata)
                           b) Get expert insights (research/insight)
                           c) Analyze current information (reason)
                           d) Make a decision
                        
                        Provide your next action in the correct format:
                        <stockdata:TICKER> or <research:TICKER> or <insight:TICKER> or <reason:"QUERY">
                    """)
                    
                    chain_of_thought.append(f"Reflection after {action_name} with market context: {reflection}")
                    
                    # Get and execute next action based on reflection
                    next_actions = self.get_actions(reflection)
                    for next_action_name, next_action_func in next_actions:
                        try:
                            self.action_inputs = ticker
                            next_result = next_action_func()
                            if next_result:
                                ticker_results[f"{next_action_name}_followup"] = next_result
                                ticker_results['analysis_chain'].append(f"{next_action_name}_followup: {str(next_result)[:200]}...")
                        except Exception as e:
                            logging.error(f"Error in follow-up action {next_action_name} for {ticker}: {e}")
                            continue
                        
                except Exception as e:
                    logging.error(f"Error in action {action_name} for {ticker}: {e}")
                    continue
            
            # Final decision synthesis with market context
            decision_prompt = f"""
            We've gathered comprehensive data about {ticker}:
            
            Market Data Summary:
            - Current Price: ${data_summary['latest_close']:.2f}
            - 24h Change: {data_summary['price_change']:.2f}%
            - Volume: {data_summary['volume']:,}
            - Avg Volume: {data_summary['avg_volume']:,}
            - 52W Range: ${data_summary['low_52w']:.2f} - ${data_summary['high_52w']:.2f}
            
            Analysis Chain:
            {json.dumps(ticker_results['analysis_chain'], indent=2)}
            
            Based on all this information:
            1. How does the current price compare to historical ranges?
            2. Is the volume indicating unusual activity?
            3. Synthesize the key findings from our analysis
            4. Identify any potential risks
            5. Evaluate the growth potential
            6. Make a final recommendation with position size
            
            Provide your decision in this format:
            DECISION: [BUY/SELL/HOLD]
            CONFIDENCE: [Low/Medium/High]
            POSITION_SIZE: [1-1000]
            RATIONALE: [Detailed explanation including market data context]
            """
            
            final_decision = self.ollama.query_ollama(decision_prompt)
            ticker_results['final_decision'] = final_decision
            
            loading_bar.dynamic_update(f"Completed analysis chain for {ticker}", operation="analyze_ticker")
            return ticker_results
        
        # Analyze each ticker
        for ticker in self.active_tickers:
            results[ticker] = analyze_ticker(ticker)
            
            # Extract decision and add to portfolio if appropriate
            try:
                decision_text = results[ticker]['final_decision']
                if 'DECISION: BUY' in decision_text:
                    # Extract position size from decision
                    position_match = re.search(r'POSITION_SIZE: (\d+)', decision_text)
                    if position_match:
                        position_size = int(position_match.group(1))
                        self.portfolio_manager.add_to_pending(ticker, position_size, decision_text)
            except Exception as e:
                logging.error(f"Error processing decision for {ticker}: {e}")
        
        # Final portfolio review
        portfolio_review_prompt = f"""
        We've analyzed {len(self.active_tickers)} stocks and made these decisions:
        {json.dumps({t: results[t]['final_decision'] for t in self.active_tickers}, indent=2)}
        
        Current portfolio state:
        {self.portfolio_manager.get_portfolio_summary()}
        
        Please review:
        1. Is our portfolio sufficiently diversified?
        2. Are our position sizes appropriate?
        3. Do we need to adjust any decisions?
        4. What risks should we monitor?
        
        Provide specific recommendations for any needed adjustments.
        """
        
        final_review = self.ollama.query_ollama(portfolio_review_prompt)
        results['portfolio_review'] = final_review
        
        if self._is_portfolio_complete():
            loading_bar.dynamic_update(
                "✅ Portfolio construction complete!\n"
                f"Selected {len(self.portfolio_manager.pending_stocks)} stocks\n"
                "Run 'get_portfolio_summary()' for details",
                operation="research_and_insight.complete"
            )
            self.portfolio_manager.confirm_portfolio()
        
        return results

    def _is_portfolio_complete(self) -> bool:
        """Check if portfolio meets completion criteria"""
        pending = self.portfolio_manager.pending_stocks
        if not 30 <= len(pending) <= 120:
            return False
            
        # Check sector diversification
        sectors = self._get_stock_sectors(pending.keys())
        if len(sectors) < 5:  # Minimum 5 sectors
            return False
            
        # Check position sizes
        total_shares = sum(details['shares'] for details in pending.values())
        for details in pending.values():
            if details['shares'] / total_shares > 0.2:  # No position > 20%
                return False
                
        return True

    def _calculate_position_size(self, sentiment_score: float, technical_score: float) -> int:
        """Calculate position size based on analysis scores"""
        combined_score = (sentiment_score + technical_score) / 2
        if combined_score < 0.4:
            return 0
            
        base_position = 100  # Base position size
        position_size = int(base_position * combined_score)
        return min(position_size, 1000)  # Cap at 1000 shares

    def get_actions(self, output: str) -> list:
        """Parse actions from output text."""
        loading_bar.dynamic_update("Starting action analysis", operation="get_actions")
        
        try:
            # First try to extract structured actions
            pattern = r'<(insight|research|reason|stockdata|select):([^>]+)>'
            matches = re.finditer(pattern, output)
            actions = []
            
            for match in matches:
                action_key = match.group(1).lower()
                param = match.group(2).strip().strip('"')
                if action_key in self.actions:
                    actions.append((action_key, self.actions[action_key]))
                    self.action_inputs = param
                    loading_bar.dynamic_update(f"Found valid {action_key} action", operation="get_actions")
                    return actions

            # If no structured actions found, create a focused reasoning prompt
            context_prompt = f"""
            Current Progress:
            - Selected stocks: {self.portfolio_manager.pending_stocks if hasattr(self, 'portfolio_manager') else []}
            - Target: 30-120 high-potential stocks
            - Previous analysis: {output[:200]}...
            
            Based on this context, please provide ONE specific action:
            <insight:TICKER> - Get insights for a specific stock
            <research:TICKER> - Get detailed research
            <stockdata:TICKER> - Get market data
            <select:TICKER1,TICKER2,...> - Submit final selection (comma-separated)
            
            Respond with exactly ONE action in the format shown above.
            """
            
            self.action_inputs = context_prompt
            return [('reason', self.reason)]
                
        except Exception as e:
            logging.error(f"Error parsing actions: {str(e)}")
            # Create error recovery prompt
            recovery_prompt = "Please provide ONE specific action in the format <action:parameter>"
            self.action_inputs = recovery_prompt
            return [('reason', self.reason)]

    def query_ollama_until_action(self, prompt, max_attempts=5):
        """Query Ollama repeatedly until we get a response containing valid actions."""
        attempt = 1
        while attempt <= max_attempts:
            loading_bar.dynamic_update(
                f"Attempt {attempt}/{max_attempts} to get actionable response from Ollama", 
                operation="query_ollama.attempt"
            )
            
            response = self.ollama.query_ollama(prompt)
            loading_bar.dynamic_update(
                "Checking response for actions...", 
                operation="query_ollama.check"
            )
            
            actions = self.get_actions(response)
            if actions:
                loading_bar.dynamic_update(
                    f"✅ Found {len(actions)} valid actions on attempt {attempt}", 
                    operation="query_ollama.success"
                )
                return response, actions
            
            loading_bar.dynamic_update(
                f"❌ No valid actions found in response. Retrying...\n"
                f"Response preview: {response[:100]}...", 
                operation="query_ollama.retry"
            )
            attempt += 1
            
        loading_bar.dynamic_update(
            "⚠️ Max attempts reached. Forcing reasoning action.", 
            operation="query_ollama.max_attempts"
        )
        # Return last response with a forced reasoning action
        return response, [('reason', self.reason)]

    def stockdata(self):
        """Get stock data for the current action_inputs ticker."""
        if not self.action_inputs:
            return {"error": "No ticker provided"}
        
        try:
            # Handle both single ticker and comma-separated tickers
            tickers = [t.strip() for t in self.action_inputs.split(',')]
            return self.stock_data.get_stock_data(tickers)
        except Exception as e:
            logging.error(f"Error getting stock data: {e}")
            return {"error": f"Failed to get stock data: {str(e)}"}

    def _analyze_technicals(self, data):
        loading_bar.dynamic_update("Starting technical analysis", operation="analyze_technicals")
        
        try:
            # Calculate technical indicators
            data = self._calculate_indicators(data)
            
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
            
            loading_bar.dynamic_update("Technical analysis complete", operation="analyze_technicals")
            return max(0, min(1, score + 0.5))
            
        except Exception as e:
            logging.error(f"Error in technical analysis: {e}")
            loading_bar.dynamic_update("Technical analysis failed, using neutral score", operation="analyze_technicals")
            return 0.5  # Return neutral score on error

    def _calculate_indicators(self, data):
        """Calculate technical indicators for analysis."""
        try:
            # Calculate SMAs
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Forward fill any NaN values
            data = data.fillna(method='ffill')
            # Back fill any remaining NaN values at the start
            data = data.fillna(method='bfill')
            
            return data
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    def _predict_price(self, data):
        loading_bar.dynamic_update("Predicting future price", operation="_predict_price")
        loading_bar.dynamic_update("Predicting future price", operation="_predict_price")
        X = data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].values
        y = data['Close'].shift(-1).dropna().values
        
        X = self.scaler.fit_transform(X[:-1])
        y = y[:-1]
        
        self.model.fit(X, y)
        
        next_day = self.scaler.transform(data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1))
        prediction = self.model.predict(next_day)[0]
        
        current_price = data['Close'].iloc[-1]
        predicted_return = (prediction - current_price) / current_price
        
        loading_bar.dynamic_update("Price prediction complete", operation="_predict_price")
        return max(0, min(1, (predicted_return + 0.1) / 0.2))

    def act(self, decisions):
        loading_bar.dynamic_update("Starting trade execution", operation="act")
        
        for decision in decisions:
            action = decision.get("action")
            ticker = decision.get("ticker")
            loading_bar.dynamic_update(f"Processing {action} for {ticker}", operation=f"act.{action}")
            
            if action == "hold":
                print("Action: Hold. No transactions made.")
                ticker = decision["ticker"]
                self.hold(ticker)
                continue
                
            ticker = decision["ticker"]
            price = decision["price"]
            shares = decision["shares"]

            if action == "buy":
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
        
        loading_bar.dynamic_update("Trade execution complete", operation="act")

    def learn(self):
        """Learn from recent trading history and adjust strategy."""
        loading_bar.dynamic_update("Learning from recent trades", operation="learn")
        
        try:
            # Get recent trade recommendations
            trades_dir = DATA_PATH / 'Trades'
            if not trades_dir.exists():
                logging.info("No trade history found - starting fresh")
                # Initialize default values when no history exists
                self.risk_tolerance = 0.02  # Default risk tolerance
                
                summary = """
                Learning Phase Summary (Initial)
                ====================
                No trading history available
                Using default parameters:
                - Risk Tolerance: 0.02
                - Strategy: Conservative initial approach
                
                Ready to begin trading with default settings.
                """
                print("\n" + summary)
                loading_bar.dynamic_update("Initialized with default settings", operation="learn")
                return
            
            # Get most recent trade file
            trade_files = list(trades_dir.glob('trade_recommendations_*.csv'))
            if not trade_files:
                logging.info("No trade files found - using default settings")
                return
            
            latest_trade_file = max(trade_files, key=lambda x: x.stat().st_mtime)
            
            try:
                trades_df = pd.read_csv(latest_trade_file)
                recent_trades = trades_df.to_dict('records')
            except Exception as e:
                logging.error(f"Error reading trade file: {e}")
                recent_trades = []
            
            if recent_trades:
                # Existing logic for analyzing trades
                avg_sentiment = sum(float(t['sentiment_score']) for t in recent_trades 
                                 if t['sentiment_score'] != 'N/A') / len(recent_trades)
                avg_technical = sum(float(t['technical_score']) for t in recent_trades 
                                 if t['technical_score'] != 'N/A') / len(recent_trades)
                
                # Adjust risk tolerance based on scores
                if avg_sentiment > 0.6 and avg_technical > 0.6:
                    self.risk_tolerance = min(0.05, self.risk_tolerance * 1.1)
                    loading_bar.dynamic_update("Increased risk tolerance", operation="learn")
                elif avg_sentiment < 0.4 or avg_technical < 0.4:
                    self.risk_tolerance = max(0.01, self.risk_tolerance * 0.9)
                    loading_bar.dynamic_update("Decreased risk tolerance", operation="learn")
                
                # Generate learning summary with historical data
                summary = f"""
                Learning Phase Summary (Historical)
                ====================
                Analyzed {len(recent_trades)} recent trades
                Average Sentiment Score: {avg_sentiment:.2f}
                Average Technical Score: {avg_technical:.2f}
                Updated Risk Tolerance: {self.risk_tolerance:.3f}
                
                Key Insights:
                - {'Increased' if self.risk_tolerance > 0.03 else 'Decreased'} risk tolerance based on performance
                - Most traded sectors: {self._get_most_traded_sectors(recent_trades)}
                """
                
                print("\n" + summary)
                
            else:
                logging.info("No valid trade history - using default settings")
                loading_bar.dynamic_update("Using default settings", operation="learn")
            
            loading_bar.dynamic_update("Learning phase complete", operation="learn")
            
        except Exception as e:
            logging.error(f"Error in learning phase: {e}")
            loading_bar.dynamic_update(f"Learning phase error: {str(e)}", operation="learn")

    def _get_most_traded_sectors(self, trades):
        """Helper method to identify most traded sectors."""
        try:
            # This would need integration with a sector classification system
            return "Sector analysis not implemented"
        except Exception as e:
            logging.error(f"Error analyzing sectors: {e}")
            return "Sector analysis failed"

    def get_portfolio_value(self):
        loading_bar.dynamic_update("Calculating portfolio value", operation="get_portfolio_value")
        loading_bar.dynamic_update("Calculating portfolio value", operation="get_portfolio_value")
        total_value = self.balance
        for ticker, shares in self.portfolio.items():
            if ticker in self.environment["stock_data"]:
                current_price = self.environment["stock_data"][ticker]['Close'].iloc[-1]
                total_value += shares * current_price
        loading_bar.dynamic_update("Portfolio value calculated", operation="get_portfolio_value")
        return total_value

    def get_performance_report(self):
        loading_bar.dynamic_update("Generating performance report", operation="get_performance_report")
        loading_bar.dynamic_update("Generating performance report", operation="get_performance_report")
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
        
        try:
            ranking = Ranking()
            ranking_report = ranking.rank_tickers()
        except Exception as e:
            ranking_report = f"Failed to generate ranking: {e}"
        report += (
            "Performance Report:\n"
            "# ...other performance metrics...\n\n"
            "Ticker Ranking:\n"
            f"{ranking_report}"
        )
        loading_bar.dynamic_update("Report generated", operation="get_performance_report")
        return report
    
    def _augment_prompt_with_stock_data(self, prompt: str) -> str:
        """Augment the prompt with stock data for any mentioned tickers."""
        additional_context = ""
        # Iterate over all known tickers
        for ticker in self.tickers:
            if ticker in prompt:
                stock_df = self.stock_data.get_stock_data([ticker]).get(ticker)
                if stock_df is not None and not stock_df.empty:
                    # Get the most recent date and its data
                    last_date = stock_df.index[-1]
                    last_row = stock_df.iloc[-1]
                    additional_context += (f"\nTicker: {ticker} | Date: {last_date} | Close: {last_row['Close']} | Volume: {last_row['Volume']}")
        if additional_context:
            prompt += "\n\nAdditional Stock Data:" + additional_context
        return prompt

    def reason(self, query=None):
        loading_bar.dynamic_update("Starting reasoning process", operation="reason")
        prompt = query or self.action_inputs
        # Augment the prompt with stock data if any ticker is mentioned
        augmented_prompt = self._augment_prompt_with_stock_data(prompt)
        phase_idx = self.action_history.start_phase("Reasoning", augmented_prompt)
        response = self.ollama.query_ollama(augmented_prompt)
        self.action_history.add_output(response, phase_idx)
        loading_bar.dynamic_update("Reasoning process complete", operation="reason")
        return response
    
    def test_outputs(self):
        loading_bar.dynamic_update("Testing outputs", operation="test_outputs")
        loading_bar.dynamic_update("Testing outputs", operation="test_outputs")
 #       if self.plan == "":
  #          raise "Plan not generated."
    #    if self.active_tickers == []:
     #       raise "Tickers not selected."
  #      if self.previous_actions == {}:
   #         logging.info("No previous actions.")
#        if self.current_actions == {}:
 #           raise "No current actions."
        if self.tickers == []:
            raise "No tickers available."
        loading_bar.dynamic_update("Outputs tested", operation="test_outputs")

    def execute_trades(self):
        loading_bar.dynamic_update("Saving trade recommendations", operation="execute_trades")
        
        try:
            # Create trades directory if it doesn't exist
            trades_dir = DATA_PATH / 'Trades'
            trades_dir.mkdir(exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            trades_file = trades_dir / f'trade_recommendations_{timestamp}.csv'
            
            # Prepare trade data
            trade_data = []
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            for ticker, details in portfolio_summary['pending'].items():
                trade_data.append({
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'action': 'BUY',  # Default to buy for initial portfolio
                    'shares': details['shares'],
                    'reason': details['reason'],
                    'sentiment_score': details.get('sentiment', 'N/A'),
                    'technical_score': details.get('technical', 'N/A'),
                    'status': 'PENDING'
                })
            
            # Save to CSV
            if trade_data:
                df = pd.DataFrame(trade_data)
                df.to_csv(trades_file, index=False)
                loading_bar.dynamic_update(
                    f"✅ Saved {len(trade_data)} trade recommendations to {trades_file}", 
                    operation="execute_trades"
                )
                
                # Generate summary report
                summary = f"""
                Trade Recommendations Summary
                ===========================
                Total Trades: {len(trade_data)}
                File Location: {trades_file}
                Timestamp: {timestamp}
                
                Portfolio Overview:
                - Pending Stocks: {len(portfolio_summary['pending'])}
                - Total Position Count: {sum(d['shares'] for d in portfolio_summary['pending'].values())}
                
                Next Steps:
                1. Review trades in {trades_file}
                2. Execute approved trades manually
                3. Update portfolio status after execution
                """
                
                # Save summary
                summary_file = trades_dir / f'trade_summary_{timestamp}.txt'
                with open(summary_file, 'w') as f:
                    f.write(summary)
                
                print("\n" + summary)
                
            else:
                loading_bar.dynamic_update(
                    "No trades to save", 
                    operation="execute_trades"
                )
                
        except Exception as e:
            logging.error(f"Error saving trade recommendations: {e}")
            loading_bar.dynamic_update(
                f"❌ Error saving trades: {str(e)}", 
                operation="execute_trades"
            )

    def _generate_trade_report(self, trade_data):
        """Generate a detailed report for each trade recommendation."""
        report = []
        for trade in trade_data:
            report.append(f"""
            Trade Recommendation for {trade['ticker']}
            =====================================
            Action: {trade['action']}
            Shares: {trade['shares']}
            
            Analysis:
            - Sentiment Score: {trade['sentiment_score']}
            - Technical Score: {trade['technical_score']}
            
            Reasoning:
            {trade['reason']}
            
            Status: {trade['status']}
            """)
        return "\n".join(report)

    # In Agent.begin()
    def begin(self):
        with loading_bar:
            phases = [
                ("Strategic Planning", self.plan_actions),
                ("Stock Selection", self.pick_tickers),
                ("Validation", self.test_outputs),
                ("Research Analysis", self.research_and_insight),
                ("Trade Execution", self.execute_trades),
                ("Learning Phase", self.learn),
                # Add new phase for ranking
                ("Ranking Analysis", self.generate_rankings)  # Add this line
            ]
            
            for phase_name, phase_function in phases:
                loading_bar.dynamic_update(f"Starting {phase_name}", operation=f"begin.{phase_name.lower().replace(' ', '_')}")
                phase_function()
                loading_bar.dynamic_update(f"Completed {phase_name}", operation=f"begin.{phase_name.lower().replace(' ', '_')}")

        self._generate_comprehensive_report()
        self.finalize_execution()
        loading_bar.dynamic_update("✅ Agent execution completed", operation="begin")

    def _generate_comprehensive_report(self):
        """Generate a comprehensive report of the agent's activities and decisions."""
        loading_bar.dynamic_update("Generating comprehensive report", operation="generate_report")
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            reports_dir = DATA_PATH / 'Reports'
            reports_dir.mkdir(exist_ok=True)
            
            report_file = reports_dir / f'comprehensive_report_{timestamp}.txt'
            
            # Gather all relevant data
            portfolio_summary = self.portfolio_manager.get_portfolio_summary() if hasattr(self, 'portfolio_manager') else {}
            
            report_sections = [
                "=================================",
                "Comprehensive Trading Report",
                "=================================",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "\nTrading Plan:",
                "-------------",
                str(self.plan) if hasattr(self, 'plan') else "No plan generated",
                "\nSelected Tickers:",
                "----------------",
                ", ".join(self.active_tickers) if self.active_tickers else "No tickers selected",
                "\nPortfolio Summary:",
                "-----------------",
                f"Pending Trades: {len(portfolio_summary.get('pending', {}))}",
                f"Current Holdings: {len(portfolio_summary.get('holdings', {}))}",
                "\nRisk Analysis:",
                "-------------",
                f"Current Risk Tolerance: {self.risk_tolerance:.3f}",
                "\nNext Steps:",
                "-----------",
                "1. Review trade recommendations in the Trades directory",
                "2. Execute approved trades manually",
                "3. Update portfolio status after execution",
                "4. Monitor performance and adjust strategy as needed"
            ]
            
            report_content = "\n".join(report_sections)
            
            # Save report
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            # Print summary to console
            print("\nComprehensive Report Generated:")
            print(f"Saved to: {report_file}")
            print("\nKey Highlights:")
            print(f"- Active Tickers: {len(self.active_tickers)}")
            print(f"- Pending Trades: {len(portfolio_summary.get('pending', {}))}")
            print(f"- Risk Tolerance: {self.risk_tolerance:.3f}")
            
            loading_bar.dynamic_update("Report generation complete", operation="generate_report")
            
        except Exception as e:
            logging.error(f"Error generating comprehensive report: {e}")
            loading_bar.dynamic_update("Error generating report", operation="generate_report")

    def generate_rankings(self):
        loading_bar.dynamic_update("Generating ticker rankings", operation="generate_rankings")
        try:
            ranking = Ranking(self.stock_data, self, self.perplexity, self.chatgpt)
            if self.active_tickers and len(self.active_tickers) > 0:
                ranked = ranking.rank_tickers(self.active_tickers)
            else:
                ranked = ranking.rank_tickers()
            loading_bar.dynamic_update("Rankings generated and saved", operation="generate_rankings")
        except Exception as e:
            logging.error(f"Error generating rankings: {e}")
            loading_bar.dynamic_update(f"Error generating rankings: {e}", operation="generate_rankings")

    def query_market(self, query: str) -> Tuple[str, list]:
        """Query the market data agent for financial analysis."""
        loading_bar.dynamic_update("Querying market data agent", operation="query_market")
        response, reasoning = self.market_data_agent.analyze_query(query)
        loading_bar.dynamic_update("Market query complete", operation="query_market")
        return response, reasoning

    def get_market_insights(self) -> dict:
        """Get daily market insights from conversation history."""
        return self.market_data_agent.get_daily_insights()



def main():
    logging.basicConfig(level=logging.INFO)

    agent = Agent()
    agent.begin()
    
    # Begin ticker ranking process with improved logging
    logging.info("Starting ticker ranking process")
    rankings = agent.generate_rankings()

    if rankings:
        logging.info(f"Ticker ranking process completed successfully. Found {len(rankings)} ranked tickers.")
        print("\nTop ranked tickers (by composite score):")
        for entry in rankings[:10]:
            print(f"{entry['ticker']}: {entry['composite']} (Analysis Time: {entry['analysis_time']})")
    else:
        logging.warning("No ranked tickers generated.")



class Portfolio:
    def __init__(self):
        self.holdings = {}
        self.pending_stocks = {}
        self.transaction_history = []
        # Update path to match project structure
        self.DATA_PATH = Path(__file__).parent.parent.parent / 'Data'
        self.PORTFOLIO_PATH = self.DATA_PATH / 'Portfolio'  # Changed from Databases/Portfolio

    def add_to_pending(self, ticker: str, shares: int, reason: str):
        self.pending_stocks[ticker] = {'shares': shares, 'reason': reason}

    def confirm_portfolio(self):
        self.holdings.update(self.pending_stocks)
        self.pending_stocks = {}

    def get_portfolio_summary(self):
        return {
            'holdings': self.holdings,
            'pending': self.pending_stocks
        }

    def save_state(self):
        """Save portfolio state and trading information."""
        loading_bar.dynamic_update("Saving portfolio state", operation="save_state")
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Create portfolio directory if it doesn't exist
            self.PORTFOLIO_PATH.mkdir(parents=True, exist_ok=True)
            
            # Save only portfolio-related state
            portfolio_data = {
                'timestamp': timestamp,
                'holdings': self.holdings,
                'pending_stocks': self.pending_stocks,
                'transaction_history': self.transaction_history
            }
            
            # Save portfolio state using correct path
            state_file = self.PORTFOLIO_PATH / f'portfolio_state_{timestamp}.json'
            with open(state_file, 'w') as f:
                json.dump(portfolio_data, f, indent=4)
                
            # Generate summary
            summary = f"""
            Portfolio State Save Summary
            ==========================
            Timestamp: {timestamp}
            
            Portfolio State:
            - Holdings: {len(self.holdings)}
            - Pending Trades: {len(self.pending_stocks)}
            - Transaction History: {len(self.transaction_history)}
            
            File Saved:
            - Portfolio: {state_file}
            """
            
            # Save summary to same directory
            summary_file = self.PORTFOLIO_PATH / f'save_summary_{timestamp}.txt'
            with open(summary_file, 'w') as f:
                f.write(summary)
                
            loading_bar.dynamic_update("Portfolio state saved successfully", operation="save_state")
            return True
            
        except Exception as e:
            logging.error(f"Error saving portfolio state: {e}")
            loading_bar.dynamic_update(f"Error saving portfolio state: {str(e)}", operation="save_state")
            return False

def finalize_execution(self):
    """Finalize the execution and clean up."""
    loading_bar.dynamic_update("Finalizing execution", operation="finalize")
    
    try:
        # Update final status in loading bar
        loading_bar.dynamic_update(
            "✅ Execution complete - Check Reports directory for details", 
            operation="finalize"
        )
        
        # Log completion
        logging.info("Agent execution completed successfully")
        
        # Save final portfolio state if there are any holdings or pending trades
        if hasattr(self, 'portfolio_manager'):
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            if portfolio_summary['holdings'] or portfolio_summary['pending']:
                self.portfolio_manager.save_state()
            else:
                logging.info("No portfolio state to save - empty portfolio")
        
        # Final cleanup
        self._cleanup()
        
    except Exception as e:
        logging.error(f"Error in finalize_execution: {e}")
        loading_bar.dynamic_update(
            f"❌ Error during finalization: {str(e)}", 
            operation="finalize"
        )

class Ranking:
    def __init__(self, stock_data: StockData, agent: Agent, perplexity: Perplexity, chatgpt: ChatGPT4o):
        self.stock_data = stock_data
        self.agent = agent
        self.perplexity = perplexity
        self.chatgpt = chatgpt
        self.ollama = Ollama()
        self.ranking_path = DATA_PATH / 'Rankings'
        self.ranking_path.mkdir(exist_ok=True)

        # Multi-Agent Configuration with Dynamic Weights
        self.analysis_weights = {
            'sentiment': {'perplexity': 0.6, 'ollama': 0.4},
            'fundamental': {'chatgpt': 0.5, 'ollama': 0.5},
            'technical': {'ollama': 1.0}
        }
        
        # Adaptive Cost Structure
        self.api_costs = {
            'perplexity': 0.02, 
            'chatgpt': 0.03,
            'ollama': 0.000001
        }

        # State Management
        self.valid_tickers = self._validate_tickers()
        self.technical_cache = LRUCache(maxsize=1000)
        
        # Quality Benchmarks from Research [2][4]
        self.quality_metrics = {
            'sentiment': {'perplexity': 0.82, 'ollama': 0.78},
            'fundamental': {'chatgpt': 0.85, 'ollama': 0.79},
            'technical': {'ollama': 0.88}
        }

    def _validate_tickers(self) -> List[str]:
        """Advanced ticker validation with liquidity filters"""
        threshold = pd.Timestamp.now() - pd.DateOffset(days=14)
        return [
            t for t in self.stock_data.tickers
            if (data := self.stock_data.get_stock_data([t]).get(t)) is not None
            and len(data) >= 30
            and data.index[-1] >= threshold
            and data['Volume'].mean() > 100000
        ]

    def _select_llm(self, analysis_type: str) -> str:
        """Cost-optimized LLM selection with quality awareness"""
        candidates = list(self.analysis_weights[analysis_type].keys())
        weights = [
            (self.quality_metrics[analysis_type][llm]**2) / (self.api_costs[llm] + 1e-8)
            for llm in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _agent_analysis(self, ticker: str) -> Dict[str, float]:
        """Optimized analysis pipeline with local-first strategy"""
        scores = {}
        
        # Technical Analysis (Local Priority)
        tech_analysis = self._local_technical_analysis(ticker)
        if tech_analysis['confidence'] > 0.7:
            scores['technical'] = tech_analysis['score']
        else:
            scores['technical'] = self._ollama_technical(ticker)

        # Fundamental Analysis
        llm = self._select_llm('fundamental')
        scores['fundamental'] = getattr(self, f"_{llm}_fundamental")(ticker)

        # Sentiment Analysis
        llm = self._select_llm('sentiment')
        scores['sentiment'] = getattr(self, f"_{llm}_sentiment")(ticker)

        return scores

    def _local_technical_analysis(self, ticker: str) -> dict:
        """Local technical analysis with cached results"""
        if ticker in self.technical_cache:
            return self.technical_cache[ticker]

        data = self.stock_data.get_stock_data([ticker])[ticker]
        analysis = {
            'rsi': self._calculate_rsi(data),
            'macd': self._calculate_macd(data),
            'volume_trend': data['Volume'].pct_change(5).iloc[-1],
            'price_momentum': data['Close'].pct_change(30).iloc[-1],
            'score': self._technical_score(data),
            'confidence': self._technical_confidence(data)
        }
        
        self.technical_cache[ticker] = analysis
        return analysis

    def _ollama_technical(self, ticker: str) -> float:
        """Fallback to Ollama for complex technical analysis"""
        prompt = f"""Perform advanced technical analysis on {ticker} considering:
        1. Price/volume divergence
        2. Institutional accumulation patterns
        3. Market structure analysis
        4. Liquidity conditions
        
        Output format: Score:-10 to 10|Rationale:<text>"""
        
        response = self.ollama.query_ollama(prompt)
        return self._parse_ollama_score(response)

    def _perplexity_sentiment(self, ticker: str) -> float:
        """Market sentiment analysis with Perplexity"""
        response = self.perplexity.query_perplexity(
            f"Analyze real-time sentiment for {ticker} from news and social media"
        )
        return self.perplexity.analyze_sentiment(response)[1] * 20 - 10  # Scale to [-10,10]

    def _ollama_fundamental(self, ticker: str) -> float:
        """Fundamental analysis with chain-of-thought reasoning"""
        prompt = f"""Analyze {ticker} fundamentals using this framework:
        1. Calculate key financial ratios (P/E, EV/EBITDA, ROIC)
        2. Evaluate growth sustainability
        3. Assess competitive advantages
        4. Compare to sector peers
        
        Final Score:-10 to 10|Rationale:<text>"""
        
        response = self.ollama.reason(prompt)
        return self._parse_ollama_score(response)

    def rank_tickers(self, tickers=None):
        
        tickers = list(self.stock_data.tickers)

        if not tickers:
            print("No valid tickers found. Saving empty ranking list.")
            self.save_rankings([])
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        rankings = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(self._process_ticker, t): t for t in tickers}
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("[cyan]Ranking Tickers", total=len(futures))
                for future in as_completed(futures):
                    result = future.result()
                    rankings.append(result)
                    progress.update(task, advance=1, description=f"Processed {result['ticker']}")

        sorted_rankings = sorted(rankings, key=lambda x: x['composite'], reverse=True)
        self.save_rankings(sorted_rankings)
        return sorted_rankings

    def _process_ticker(self, ticker: str) -> Dict:
        """Full analysis pipeline for a single ticker"""
        scores = self._agent_analysis(ticker)
        return {
            'ticker': ticker,
            'composite': sum(scores.values()),
            **scores,
            'analysis_time': pd.Timestamp.now().isoformat()
        }

    def _parse_ollama_score(self, text: str) -> float:
        """Robust score parsing from Ollama responses"""
        patterns = [
            r"Score:\s*(-?\d+\.?\d*)",
            r"Rating:\s*(-?\d+\.?\d*)",
            r"Recommendation:\s*(-?\d+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return max(-10, min(10, float(match.group(1))))
        return 0.0

    def save_rankings(self, ranked_tickers):
        try:
            from datetime import datetime
            self.ranked_file = self.ranking_path / f"rankings_{datetime.now().date()}.csv"
            import pandas as pd
            df = pd.DataFrame(ranked_tickers)
            df.to_csv(self.ranked_file, index=False)
            logging.info(f"Saved rankings to {self.ranked_file}")
        except Exception as e:
            logging.error(f"Error saving rankings: {e}")

    def _generate_html_report(self, df: pd.DataFrame):
        """Create visual report with key metrics"""
        report = [
            "<html><head><title>Stock Rankings</title>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head>",
            "<body><h1>Top 100 Ranked Stocks</h1>",
            "<div id='chart'></div>",
            "<script>",
            "const data = [",
            "{",
            "x: " + str(df['ticker'].tolist()) + ",",
            "y: " + str(df['composite'].tolist()) + ",",
            "type: 'bar',",
            "marker: {color: '" + str(df['composite'].apply(lambda x: 'green' if x > 0 else 'red').tolist()) + "'}",
            "}];",
            "Plotly.newPlot('chart', data);",
            "</script></body></html>"
        ]
        
        with open(self.ranking_path / "rankings.html", "w") as f:
            f.write("\n".join(report))

    def _calculate_rsi(self, data) -> float:
        """Calculate the Relative Strength Index (RSI) for the given data."""
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_macd(self, data) -> float:
        """Calculate the MACD for the given data."""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd.iloc[-1]

    def _technical_score(self, data) -> float:
        """Compute a simple technical score based on RSI and MACD."""
        rsi = self._calculate_rsi(data)
        macd = self._calculate_macd(data)
        score = (rsi - 50) / 10 + macd / 5
        return score

    def _technical_confidence(self, data) -> float:
        """Return a technical confidence score; 1.0 if sufficient data, else lower."""
        if len(data) >= 30:
            return 1.0
        else:
            return 0.5

    def _chatgpt_fundamental(self, ticker: str) -> float:
        """Analyze fundamental metrics using ChatGPT and return a score between -10 and 10."""
        query = f"Provide a fundamental analysis score for {ticker} (scale -10 to 10) based on financial metrics and growth potential."
        response, status = self.chatgpt.query_OpenAI(query=query)
        return self._parse_ollama_score(response)

    def _ollama_fundamental(self, ticker: str) -> float:
        """Analyze fundamentals using Ollama and return a score between -10 and 10."""
        query = f"Provide a fundamental analysis for {ticker} in terms of financial health, growth potential, and competitive advantage. Give a score between -10 and 10."
        response = self.ollama.query_ollama(query)
        return self._parse_ollama_score(response)

    def _ollama_sentiment(self, ticker: str) -> float:
        """Analyze market sentiment using Ollama and return a score between -10 (bearish) and 10 (bullish)."""
        query = f"Analyze the market sentiment for {ticker} using news and social media. Score from -10 (bearish) to 10 (bullish)."
        response = self.ollama.query_ollama(query)
        return self._parse_ollama_score(response)

    def _chatgpt_sentiment(self, ticker: str) -> float:
        """Analyze market sentiment using ChatGPT and return a score between -10 and 10."""
        query = f"Provide market sentiment for {ticker} (scale -10 to 10) based on recent news."
        response, status = self.chatgpt.query_OpenAI(query=query)
        return self._parse_ollama_score(response)

class MarketDataAgent:
    def __init__(self):
        self.stock_data = StockData()
        self.ollama = Ollama()
        self.perplexity = Perplexity()
        self.chatgpt = ChatGPT4o()
        self.conversation_path = CONVERSION_PATH
        self.conversation_path.mkdir(parents=True, exist_ok=True)
        
    def _get_conversation_file(self) -> Path:
        """Get the conversation file path for today."""
        today = datetime.now().strftime("%Y%m%d")
        return self.conversation_path / f"market_conversation_{today}.json"
    
    def _log_conversation(self, query: str, reasoning_chain: List[str], response: str):
        """Log a conversation with timestamp."""
        file_path = self._get_conversation_file()
        timestamp = datetime.now().isoformat()
        
        try:
            if file_path.exists():
                conversations = json.loads(file_path.read_text())
            else:
                conversations = []
                
            conversations.append({
                "timestamp": timestamp,
                "query": query,
                "reasoning_chain": reasoning_chain,
                "response": response
            })
            
            file_path.write_text(json.dumps(conversations, indent=2))
        except Exception as e:
            logging.error(f"Error logging conversation: {e}")

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis."""
        try:
            # Calculate SMAs
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            return data.fillna(method='ffill').fillna(method='bfill')
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    def _analyze_technicals(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Analyze technical indicators and return a score and reasoning."""
        try:
            data = self._calculate_technical_indicators(data)
            latest = data.iloc[-1]
            score = 0
            reasoning = []
            
            # SMA crossover
            if latest['SMA_20'] > latest['SMA_50']:
                score += 0.2
                reasoning.append("Bullish: 20-day SMA above 50-day SMA")
            else:
                score -= 0.2
                reasoning.append("Bearish: 20-day SMA below 50-day SMA")
            
            # RSI
            if 30 < latest['RSI'] < 70:
                score += 0.1
                reasoning.append(f"Neutral RSI: {latest['RSI']:.2f}")
            elif latest['RSI'] <= 30:
                score += 0.2
                reasoning.append(f"Oversold RSI: {latest['RSI']:.2f}")
            else:
                score -= 0.2
                reasoning.append(f"Overbought RSI: {latest['RSI']:.2f}")
            
            # MACD
            if latest['MACD'] > latest['Signal']:
                score += 0.2
                reasoning.append("Bullish: MACD above signal line")
            else:
                score -= 0.2
                reasoning.append("Bearish: MACD below signal line")
            
            return max(0, min(1, score + 0.5)), reasoning
            
        except Exception as e:
            logging.error(f"Error in technical analysis: {e}")
            return 0.5, ["Error performing technical analysis"]

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a market-related query using chain-of-thought reasoning."""
        loading_bar.dynamic_update("Starting market analysis", operation="analyze_query")
        
        try:
            # Extract tickers from query
            tickers = self._extract_tickers(query)
            
            # Get market data
            market_data = {ticker: self.stock_data.get_stock_data([ticker])[ticker] 
                         for ticker in tickers}
            
            # Technical analysis
            technical_scores = {}
            technical_reasoning = {}
            for ticker, data in market_data.items():
                score, reasoning = self._analyze_technicals(data)
                technical_scores[ticker] = score
                technical_reasoning[ticker] = reasoning
            
            # Sentiment analysis
            sentiment_scores = {}
            for ticker in tickers:
                perplexity_response = self.perplexity.query_perplexity(
                    f"Analyze market sentiment for {ticker}"
                )
                sentiment_scores[ticker] = self.perplexity.analyze_sentiment(perplexity_response)[1]
            
            # Chain of thought reasoning with Ollama
            prompt = f"""Analyze the following market query: {query}
            Technical Analysis: {technical_reasoning}
            Sentiment Scores: {sentiment_scores}
            
            Provide a detailed analysis with:
            1. Key market factors
            2. Technical indicators interpretation
            3. Market sentiment assessment
            4. Potential risks and opportunities
            5. Actionable recommendations
            """
            
            reasoning_chain = []
            response = self.ollama.reason(prompt, reasoning_chain)
            
            # Log the conversation
            self._log_conversation(query, reasoning_chain, response)
            
            loading_bar.dynamic_update("Analysis complete", operation="analyze_query")
            
            return {
                "query": query,
                "tickers": tickers,
                "technical_analysis": {
                    "scores": technical_scores,
                    "reasoning": technical_reasoning
                },
                "sentiment_analysis": sentiment_scores,
                "reasoning_chain": reasoning_chain,
                "response": response
            }
            
        except Exception as e:
            logging.error(f"Error analyzing query: {e}")
            return {"error": str(e)}

    def _extract_tickers(self, query: str) -> List[str]:
        """Extract stock tickers from a query using regex."""
        # Basic pattern for stock tickers (can be enhanced)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        return list(set(re.findall(ticker_pattern, query)))

    def get_conversation_history(self, days: int = 1) -> List[Dict[str, Any]]:
        """Get conversation history for the specified number of days."""
        history = []
        today = datetime.now()
        
        for i in range(days):
            date = today - timedelta(days=i)
            file_path = self.conversation_path / f"market_conversation_{date.strftime('%Y%m%d')}.json"
            
            if file_path.exists():
                try:
                    conversations = json.loads(file_path.read_text())
                    history.extend(conversations)
                except Exception as e:
                    logging.error(f"Error reading conversation history: {e}")
        
        return history

    def get_daily_insights(self) -> Dict[str, Any]:
        """Get insights from today's conversations."""
        today_file = self._get_conversation_file()
        if not today_file.exists():
            return {"error": "No conversations found for today"}
            
        try:
            conversations = json.loads(today_file.read_text())
            
            # Extract key themes and insights
            all_text = " ".join([
                f"{conv['query']} {conv['response']}"
                for conv in conversations
            ])
            
            prompt = f"""Analyze these market conversations and provide:
            1. Key market themes discussed
            2. Important market insights
            3. Potential action items
            4. Areas needing further analysis
            
            Conversations: {all_text}"""
            
            reasoning_chain = []
            insights = self.ollama.reason(prompt, reasoning_chain)
            
            return {
                "conversation_count": len(conversations),
                "insights": insights,
                "reasoning_chain": reasoning_chain
            }
            
        except Exception as e:
            logging.error(f"Error getting daily insights: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    main()
