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
        
    def _initialize_sentiment_analyzer(self):
        """Initialize VADER sentiment analyzer."""
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()

    def analyze_sentiment(self, statement):
        """Analyze sentiment of a statement using VADER."""
        sentiment_scores = self.sia.polarity_scores(statement)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            sentiment = "Bullish"
        elif compound_score <= -0.05:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        normalized_score = compound_score * 2
        return sentiment, normalized_score

    def read_api(self):
        try:
            with open(self.api_key_path, 'r') as file:
                self.api_key = file.read().strip()
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is empty")
        return self.api_key

    
    def query_OpenAI(self, model="Chatgpt-4o", query="", max_tokens=150, temperature=0.5, role=""):
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
        
        return output, status

    def evaluate_response(self, response):
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
        

class Perplexity:
    def __init__(self):
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
        
    def read_api(self) -> str:
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.read().strip()
                if not api_key:
                    raise ValueError("Perplexity API key is empty")
                return api_key
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")

    def query_perplexity(self, model: str = "", query: str = "", 
                        max_tokens: int = 150, temperature: float = 0.5, 
                        role: str = "") -> Tuple[str, bool]:
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
        return sentiment, normalized_score

    def evaluate_response(self, response: Union[str, Dict]) -> Dict:
        """
        Evaluate response from Perplexity API with enhanced error handling.
        """
        try:
            if not response:
                return {"error": "No response received", "sentiment": "neutral", "sentiment_score": 0.0}
            
            text = response if isinstance(response, str) else str(response)
            sentiment, score = self.analyze_sentiment(text)
            
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
        self.data_path = STOCK_DATA_PATH / 'Stock-Data'
        self.maintain_stock_data()
        self.stock_data = {}  # Initialize empty dictionary
        self.read_stock_data()  # Now populate it
        self.tickers = list(self.stock_data.keys())

    def get_stock_data(self, tickers):
        data = {}
        for ticker in tickers:
            data[ticker] = self.stock_data[ticker]
        return data
    
    def read_stock_data(self):
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
                
            return stock_data
            
        except Exception as e:
            logging.error(f"Error reading stock data: {e}")
            return {}


    def maintain_stock_data(self) -> Optional[bool]:
        """
        Run update_data() from the Data Management module.
        
        Returns:
            Optional[bool]: True if update was successful, False if an error occurred, None if update_data() doesn't return a status.
        """
        try:


            logging.info("Starting data update process")
            result = update_data()
            logging.info("Data update process completed")
            
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


    def query_ollama(self, prompt: str) -> str:
        try:
            response = ollama.generate(
                model = 'llama3.2:1b',
                prompt=prompt,
#z                max_tokens=120000,
#                temperature=0.5,
                system="You are a financial agent making decisions based on market analysis."
            )
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
        try:
            response = ollama.generate(
                model=self.model,
                prompt=f"{prompt}\n\nIMPORTANT: Your response MUST start with 'ACTION:' followed by one of these actions: {', '.join(self.action_handlers.keys())}. Then, add 'PARAMS:' followed by the necessary parameters separated by commas.",
                system="You are a financial agent making decisions based on market analysis. Always respond with an action and parameters in the specified format."
            )
            return response['response']
        except Exception as e:
            logging.error(f"Error querying Ollama: {e}")
            return "ACTION: error PARAMS: An error occurred during the query."

    def make_plan(self, query: str) -> str:
        if not query:
            raise ValueError("Query cannot be empty")

        start_time = time.time()
        plan = []
        
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

            return f"Final plan:\n{get_plan()}"

        except Exception as e:
            logging.error(f"Error in make_plan: {e}")
            return f"Error creating plan: {str(e)}"


class Agent:
    def __init__(self, initial_balance=100, risk_tolerance=0.02):
        self.balance = initial_balance
        
        # Inialize components
        self.chatgpt = ChatGPT4o()
        self.perplexity = Perplexity()
        self.stock_data = StockData()
        self.ollama = Ollama()

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


    def buy(self, ticker, shares, price, min=0, max=0):
        if min == max == 0:
            predicted_volume = self.predict_future_volume(ticker)
            dynamic_lead_time = self.calculate_dynamic_lead_time(ticker)
            safety_stock = self.calculate_adaptive_safety_stock(ticker)
            base_min = (predicted_volume * dynamic_lead_time) + safety_stock

            reorder_qty = self.optimize_reorder_quantity(ticker, base_min)
            sentiment_adjusted_min, sentiment_adjusted_max = self.adjust_levels_for_sentiment(
                ticker,
                base_min,
                base_min + reorder_qty
            )
            min, max = round(sentiment_adjusted_min), round(sentiment_adjusted_max)

        self.new_recommendations.append(f"buy {shares} {ticker} at {price} with a min of {min} and a max of {max}")

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

    def hold(self, ticker):
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
        
    
    def research(self):
        """Get detailed research analysis using Perplexity."""
        if not self.action_inputs:
            return {"error": "No ticker provided", "sentiment": "neutral", "sentiment_score": 0.0}
            
        query = f"Provide detailed research analysis for {self.action_inputs}. Include financial metrics, competitive analysis, and growth prospects."
        response, status = self.perplexity.query_perplexity(query=query)
        if status:
            return self.perplexity.evaluate_response(response)
        return {"error": "Failed to get research", "sentiment": "neutral", "sentiment_score": 0.0}

    def insight(self):
        """Get market insights for a specific ticker using ChatGPT."""
        if not self.action_inputs:
            return {"error": "No ticker provided", "sentiment": "neutral", "sentiment_score": 0.0}
            
        query = f"Provide market insights and sentiment analysis for {self.action_inputs}. Focus on recent trends, news, and market sentiment."
        response, status = self.chatgpt.query_OpenAI(query=query)
        if status:
            return response
        return {"error": "Failed to get insights", "sentiment": "neutral", "sentiment_score": 0.0}

    def spinner_animation(self, plan_output=None, current_stage=None):
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

    def initialize_plan_tasks(self, plan_output):
        plan_steps = plan_output.split('\n')
        for step in plan_steps:
            task_name = step.strip()
            if task_name:
                self.task_progresses[task_name] = self.progress.add_task(f"[cyan]{task_name}", total=100, status="Pending")

    def update_plan_progress(self, current_stage):
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


    def update_task_progress(self, task, status, advance=0):
        if task not in self.task_progresses:
            self.task_progresses[task] = self.progress.add_task(f"[cyan]{task}", total=100, status="Pending")
        self.progress.update(self.task_progresses[task], advance=advance, status=status)

    def initialize_progress(self):
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

    def update_most_recent_action(self, action):
        self.most_recent_action = action
        self.spinner_animation(current_stage=self.most_recent_action)

    def finalize_execution(self):
        for task_id in self.task_progresses.values():
            self.progress.update(task_id, completed=100, status="[bold green]Completed[/bold green]")
        self.progress.update(self.overall_task, completed=100, status="[bold green]Completed[/bold green]")
        self.progress.stop()
        self.console.print(Panel.fit("[bold green]All financial tasks completed successfully.[/bold green]", border_style="green"))

    def return_stock_data(self):
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
        return self.stock_data.get_stock_data(self.action_inputs)
        
    def perceive_environment(self, stock_data, perplexity_insights, chatgpt_insights):
        self.environment = {
            "stock_data": stock_data,
            "perplexity_insights": perplexity_insights,
            "chatgpt_insights": chatgpt_insights,
        }
        self._update_technical_indicators()


    def get_portfolio(self):
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

    def load_previous_actions(self):
        try:
            # Use pathlib for portfolio path handling
            actions_files = list(CONVERSION_PATH.rglob("actions.csv"))
            if actions_files:
                # Get the most recent actions file
                latest_actions = max(actions_files, key=lambda p: p.stat().st_mtime)
                return pd.read_csv(latest_actions)
            return pd.DataFrame()  # Return empty DataFrame if no actions found
        except Exception as e:
            logging.error(f"Error reading actions: {e}")
            return pd.DataFrame()

    def plan_actions(self):
        self.update_most_recent_action("Planning actions")
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
        self.spinner_animation(plan_output=self.plan, current_stage="Planning")

    def decide_action(self):
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
      
        return

    def run_phase(self):
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
            return
        except KeyError:
            logging.error(f"Invalid phase: {self.phase}")
            return 


    def pick_tickers(self):
        self.update_most_recent_action("Selecting tickers")
        
        initial_prompt = self._generate_initial_prompt()
        output = self.ollama.query_ollama(initial_prompt)
        self.previous_actions.append(output)
        
        selected_stocks = set(self.tickers)
        iteration_count = 0
        max_iterations = (60*60)/5  # Prevent infinite loops
        
        while iteration_count < max_iterations:
            actions = self.get_actions(output)
            
            for action_name, action_func in actions:
                if action_name == 'reason':
                    refined_output = action_func()
                    self.previous_actions.append(refined_output)
                    
                    if self._should_stop_narrowing(refined_output, selected_stocks):
                        return
                    
                    new_selection = self.extract_selected_stocks(refined_output)
                    if new_selection:
                        selected_stocks = new_selection
                
                elif action_name in ['insight', 'research', 'stockdata']:
                    action_func()  # Execute but don't store results unless needed
            
            if 30 <= len(selected_stocks) <= 120:
                if self._confirm_optimal_set(selected_stocks):
                    self.active_tickers = list(selected_stocks)
                    return
            
            output = self.ollama.query_ollama("Continue narrowing down the stock selection.")
            iteration_count += 1
        
        self._handle_max_iterations_reached(selected_stocks)

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
        'reason' --> query --> ${{YOUR QUERY HERE}}
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
        self.ollama.reason("Maximum iterations reached. Finalizing stock selection.")


    def research_and_insight(self):
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

        return decisions
    
    def get_actions(self, output):
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

        return actions

    def stockdata(self, ticker):
        """Retrieve historical stock data for analysis."""
        if ticker in self.stock_data.stock_data:
            data = self.stock_data.get_stock_data([ticker])
            return {ticker: data[ticker]}
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
        chatgpt_sentiment = self.environment["chatgpt_insights"].get(ticker, "")
        perplexity_sentiment = self.environment["perplexity_insights"].get(ticker, "")
        
        sentiment_score = 0.5  # Neutral by default
        if "bullish" in chatgpt_sentiment:
            sentiment_score += 0.25
        if "bearish" in chatgpt_sentiment:
            sentiment_score -= 0.25
        
        return max(0, min(1, sentiment_score))

    def _analyze_technicals(self, data):
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
        
        return max(0, min(1, score + 0.5))

    def _predict_price(self, data):
        X = data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].values
        y = data['Close'].shift(-1).dropna().values
        
        X = self.scaler.fit_transform(X[:-1])
        y = y[:-1]
        
        self.model.fit(X, y)
        
        next_day = self.scaler.transform(data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1))
        prediction = self.model.predict(next_day)[0]
        
        current_price = data['Close'].iloc[-1]
        predicted_return = (prediction - current_price) / current_price
        
        return max(0, min(1, (predicted_return + 0.1) / 0.2))

    def _calculate_buy_amount(self, ticker, data):
        current_price = data['Close'].iloc[-1]
        max_shares = int(self.balance * self.risk_tolerance / current_price)
        if max_shares > 0:
            return {"action": "buy", "ticker": ticker, "shares": max_shares, "price": current_price}
        return None

    def _calculate_sell_amount(self, ticker, data):
        current_price = data['Close'].iloc[-1]
        shares = self.portfolio.get(ticker, 0)
        if shares > 0:
            return {"action": "sell", "ticker": ticker, "shares": shares, "price": current_price}
        return None

    def act(self, decisions):
        for decision in decisions:
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

    def learn(self):
        recent_trades = self.history[-10:]
        if recent_trades:
            profit_ratio = sum(1 for trade in recent_trades if "Sold" in trade) / len(recent_trades)
            if profit_ratio > 0.6:
                self.risk_tolerance = min(0.05, self.risk_tolerance * 1.1)
            elif profit_ratio < 0.4:
                self.risk_tolerance = max(0.01, self.risk_tolerance * 0.9)

    def get_portfolio_value(self):
        total_value = self.balance
        for ticker, shares in self.portfolio.items():
            if ticker in self.environment["stock_data"]:
                current_price = self.environment["stock_data"][ticker]['Close'].iloc[-1]
                total_value += shares * current_price
        return total_value

    def get_performance_report(self):
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
        
        return report
    
    def reason(self, query=None):
        if query == None:
            query = self.action_inputs
        response = self.ollama.query_ollama(query)
        return response
    
    def test_outputs(self):
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

    def excute_trades(self):
        self.update_most_recent_action("Executing trades")
        self.act(self.decisons)
        self.spinner_animation(current_stage="ExecuteTrades")


    def begin(self):
        self.spinner_animation(current_stage="Initialization")
        self.phase = 0
        self.plan_actions()
        self.spinner_animation(plan_output=self.plan, current_stage="Planning")

        self.phase = 1
        self.pick_tickers()
        self.spinner_animation(current_stage="PickingTickers")

        self.phase = 2
        self.test_outputs()
        self.spinner_animation(current_stage="Validation")

        self.phase = 3
        self.research_and_insight()
        self.spinner_animation(current_stage="ResearchAndInsight")

        self.phase = 4
        self.excute_trades()
        self.spinner_animation(current_stage="ExecuteTrades")

        self.phase = 5
        self.learn()
        self.spinner_animation(current_stage="Learning")

        # Finalize
        self.finalize_execution()


def main():
    logging.basicConfig(level=logging.INFO)

    agent = Agent()
    agent.begin()

if __name__ == "__main__":
    main()
