import re
import openai
import os
import sys
import time
import logging
import math
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
import ollama
import logging
import random
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from loading import DynamicLoadingBar

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

# Initialize the global loading bar
loading_bar = DynamicLoadingBar()

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
        loading_bar.dynamic_update("Instance initialization complete", operation="ChatGPT4o.__init__")
        
    def _initialize_sentiment_analyzer(self):
        """Initialize VADER sentiment analyzer."""
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()

    def analyze_sentiment(self, statement):
        loading_bar.dynamic_update("Starting sentiment analysis", operation="analyze_sentiment")
        sentiment_scores = self.sia.polarity_scores(statement)
        compound_score = sentiment_scores['compound']
         
        if compound_score >= 0.05:
            sentiment = "Bullish"
        elif compound_score <= -0.05:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        normalized_score = compound_score * 2
        loading_bar.dynamic_update("Sentiment analysis complete", operation="analyze_sentiment")
        return sentiment, normalized_score

    def read_api(self):
        loading_bar.dynamic_update("Reading API key", operation="read_api")
        try:
            with open(self.api_key_path, 'r') as file:
                self.api_key = file.read().strip()
        except FileNotFoundError:
            loading_bar.dynamic_update("API key file not found", operation="read_api")
            raise ValueError(f"API key file not found at {self.api_key_path}")
        
        if not self.api_key:
            loading_bar.dynamic_update("Empty API key", operation="read_api")
            raise ValueError("OpenAI API key is empty")
        loading_bar.dynamic_update("API key loaded successfully", operation="read_api")
        loading_bar.dynamic_update("API key read complete", operation="read_api")
        return self.api_key

    
    def query_OpenAI(self, model="Chatgpt-4o", query="", max_tokens=150, temperature=0.5, role=""):
        loading_bar.dynamic_update("Preparing OpenAI query", operation="query_OpenAI")
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
        
        loading_bar.dynamic_update("OpenAI query complete", operation="query_OpenAI")
        return output, status

    def evaluate_response(self, response):
        loading_bar.dynamic_update("Evaluating response", operation="evaluate_response")
        """
        Evaluate response from ChatGPT-4o API and return insights with sentiment analysis.
        """
        try:
            if not response:
                return "No response received."
            if "error" in str(response).lower():
                return "The response contains an error."
            
            sentiment, score = self.analyze_sentiment(str(response))
            loading_bar.dynamic_update("Response evaluation complete", operation="evaluate_response")
            return {
                "text": str(response),
                "sentiment": sentiment,
                "sentiment_score": score
            }
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            loading_bar.dynamic_update("Response evaluation complete", operation="evaluate_response")
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
        loading_bar.dynamic_update("Instance initialization complete", operation="Perplexity.__init__")
        
    def read_api(self) -> str:
        loading_bar.dynamic_update("Reading Perplexity API key", operation="read_api")
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.read().strip()
                if not api_key:
                    raise ValueError("Perplexity API key is empty")
                return api_key
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")
        loading_bar.dynamic_update("Perplexity API key read", operation="read_api")

    def query_perplexity(self, model: str = "", query: str = "", 
                        max_tokens: int = 150, temperature: float = 0.5, 
                        role: str = "") -> Tuple[str, bool]:
        loading_bar.dynamic_update("Querying Perplexity", operation="query_perplexity")
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
            loading_bar.dynamic_update("Perplexity query complete", operation="query_perplexity")
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
        loading_bar.dynamic_update("Starting sentiment analysis", operation="analyze_sentiment")
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
        loading_bar.dynamic_update("Sentiment analysis complete", operation="analyze_sentiment")
        return sentiment, normalized_score

    def evaluate_response(self, response: Union[str, Dict]) -> Dict:
        loading_bar.dynamic_update("Evaluating Perplexity response", operation="evaluate_response")
        """
        Evaluate response from Perplexity API with enhanced error handling.
        """
        try:
            if not response:
                return {"error": "No response received", "sentiment": "neutral", "sentiment_score": 0.0}
            
            text = response if isinstance(response, str) else str(response)
            sentiment, score = self.analyze_sentiment(text)
            loading_bar.dynamic_update("Response evaluation complete", operation="evaluate_response")
            
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
        loading_bar.dynamic_update("Generating plan with Ollama", operation="query_ollama_for_plan")
        try:
            loading_bar.dynamic_update("Generating plan, Querying Ollama", operation="query_ollama_for_plan")
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial agent making decisions based on market analysis. Always respond with an action and parameters in the specified format.'
                    },
                    {
                        'role': 'user',
                        'content': f"{prompt}\n\nIMPORTANT: Your response MUST start with 'ACTION:' followed by one of these actions: {', '.join(self.action_handlers.keys())}. Then, add 'PARAMS:' followed by the necessary parameters separated by commas."
                    }
                ],
                stream=False
            )
            print("Formatted content:", response['message']['content'])
            loading_bar.dynamic_update("Plan generation complete", operation="query_ollama_for_plan")
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error querying Ollama: {e}")
            return "ACTION: error PARAMS: An error occurred during the query."



    def make_plan(self, query: str) -> str:
        loading_bar.dynamic_update("Making plan", operation="make_plan")
        loading_bar.dynamic_update("Generating plan", operation="make_plan")
        if not query:
            raise ValueError("Query cannot be empty")

        loading_bar.dynamic_update("Initializing start time", operation="make_plan")
        start_time = time.time()
        loading_bar.dynamic_update("Initializing start time", operation="make_plan")
        loading_bar.dynamic_update("Initializing plan", operation="make_plan")
        plan = []
        loading_bar.dynamic_update("Initializing plan", operation="make_plan")
        
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
            loading_bar.dynamic_update("Plan made", operation="make_plan")
            return f"Final plan:\n{get_plan()}"

        except Exception as e:
            logging.error(f"Error in make_plan: {e}")
            return f"Error creating plan: {str(e)}"
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

        self.prompt_manager = PromptManager(self)
        
        self.action_history = ActionHistory()
        self.portfolio_manager = Portfolio()
        
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
        loading_bar.dynamic_update("Finalizing execution", operation="finalize_execution")
        for task_id in self.task_progresses.values():
            self.progress.update(task_id, completed=100, status="[bold green]Completed[/bold green]")
        self.progress.update(self.overall_task, completed=100, status="[bold green]Completed[/bold green]")
        self.progress.stop()
        self.console.print(Panel.fit("[bold green]All financial tasks completed successfully.[/bold green]", border_style="green"))
        loading_bar.dynamic_update("Execution finalized", operation="finalize_execution")

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
            return
        except KeyError:
            logging.error(f"Invalid phase: {self.phase}")
            return 
        
    def pick_tickers(self):
        loading_bar.dynamic_update("Starting stock selection process", operation="pick_tickers")
        
        try:
            prompt = self.prompt_manager.get_stock_selection_prompt()
            output = self.ollama.query_ollama(prompt)
            self.previous_actions.append(output)
            
            iteration_count = 0
            max_iterations = (60*60)//5  # Prevent infinite loops
            selected_stocks = set()
            
            while iteration_count < max_iterations:
                loading_bar.dynamic_update(
                    f"Stock selection iteration {iteration_count + 1}/{max_iterations}", 
                    operation="pick_tickers.iteration"
                )
                
                actions = self.get_actions(output)
                for action_name, action_func in actions:
                    if action_name == 'select':
                        # Parse selected stocks from the parameter
                        tickers = self.action_inputs.strip('"').split(',')
                        selected_stocks.update(ticker.strip() for ticker in tickers)
                        
                        if 30 <= len(selected_stocks) <= 120:
                            loading_bar.dynamic_update(
                                f"Valid stock selection found: {len(selected_stocks)} stocks", 
                                operation="pick_tickers.complete"
                            )
                            return list(selected_stocks)
                    else:
                        # Handle other actions normally
                        refined_output = action_func(self.action_inputs)
                        self.previous_actions.append(refined_output)
                
                output = self.ollama.query_ollama(self._generate_refinement_prompt(selected_stocks))
                iteration_count += 1
            
            loading_bar.dynamic_update(
                "Max iterations reached, using emergency selection", 
                operation="pick_tickers.timeout"
            )
            return self._emergency_stock_selection()
            
        except Exception as e:
            logging.error(f"Error in pick_tickers: {e}")
            loading_bar.dynamic_update(
                "Error in stock selection, falling back to emergency selection", 
                operation="pick_tickers.error"
            )
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
        for ticker in self.active_tickers:
            loading_bar.dynamic_update(f"Analyzing {ticker}", operation="research_and_insight.ticker_analysis")
            
            # Get analysis results
            sentiment_score = self._analyze_sentiment(ticker)
            technical_score = self._analyze_technicals(self.stock_data.get_stock_data(ticker))
            
            # Calculate position size based on scores
            position_size = self._calculate_position_size(sentiment_score, technical_score)
            
            if position_size > 0:
                reason = f"Sentiment: {sentiment_score:.2f}, Technical: {technical_score:.2f}"
                self.portfolio_manager.add_to_pending(ticker, position_size, reason)
            
            results[ticker] = {
                'sentiment': sentiment_score,
                'technical': technical_score,
                'position_size': position_size
            }

        # Get next action from Ollama with context
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        context = self.action_history.get_context()
        
        next_action_prompt = f"""
        Previous Analysis Context:
        {context}

        Current Portfolio Status:
        {portfolio_summary}

        Based on the analysis and current portfolio, what action should be taken next?
        Consider:
        1. Portfolio balance and diversification
        2. Risk management
        3. Entry/exit timing
        4. Market conditions

        Respond with one of these actions:
        <insight:TICKER>     # Get more insights
        <research:TICKER>    # Deep dive analysis
        <reason:"QUERY">     # Analyze specific aspect
        <select:"TICKERS">   # Finalize stock selection
        """
        
        response = self.ollama.query_ollama(next_action_prompt)
        self.action_history.add_output(response)
        
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
        """Parse actions from output text with robust parameter binding and progress tracking."""
        loading_bar.dynamic_update("Starting action analysis", operation="get_actions")
        
        try:
            loading_bar.dynamic_update(
                f"Analyzing response for actions...\n"
                f"Response preview: {output[:100]}...", 
                operation="get_actions.analyze"
            )
            
            # Define action mapping with validation requirements
            action_map = {
                'insight': (self.insight, r'^[A-Z]{1,5}$'),  # Stock ticker format
                'research': (self.research, r'^[A-Z]{1,5}$'),
                'reason': (self.reason, r'^[A-Za-z0-9\s\-_]{1,100}$'),  # Query format
                'stockdata': (self.stock_data.get_stock_data, r'^[A-Z]{1,5}$')
            }
            
            # Improved regex pattern with named groups and flexible parameter capture
            pattern = r'(?P<action>insight|research|reason|stockdata)\s*.*?\$\{(?P<param>[^\}]+)\}'
            actions = []
            
            # Find all matches
            matches = re.finditer(pattern, output, re.IGNORECASE)
            for match in matches:
                action_key = match.group('action').lower()
                param = match.group('param').strip()
                
                if action_key in action_map:
                    func, param_pattern = action_map[action_key]
                    
                    loading_bar.dynamic_update(
                        f"Validating {action_key} parameter: {param}", 
                        operation=f"get_actions.validate.{action_key}"
                    )
                    
                    # Validate parameter format
                    if re.match(param_pattern, param):
                        actions.append((action_key, func))
                        self.action_inputs = param
                        
                        loading_bar.dynamic_update(
                            f"✅ Valid {action_key} action found\n"
                            f"Parameter: {param}\n"
                            f"Context: ...{output[max(0, match.start()-30):match.end()]}", 
                            operation=f"get_actions.found.{action_key}"
                        )
                    else:
                        loading_bar.dynamic_update(
                            f"⚠️ Invalid parameter format for {action_key}: {param}", 
                            operation=f"get_actions.invalid.{action_key}"
                        )
            
            # If no valid actions found, treat output as reasoning query with history context
            if not actions:
                loading_bar.dynamic_update(
                    "No valid actions found, treating as reasoning query",
                    operation="get_actions.fallback"
                )
                context = self.action_history.get_context()
                self.action_inputs = f"""Previous context:
                {context}\n\nCurrent output:
                {output} valid responses: 
                    {r"'insight'"} --> ${r'{ticker}'}
                    {r"'research'"} --> ${r'{ticker}'}
                    {r"'reason'"} --> query --> {r'${YOUR QUERY HERE}'}
                    {r"'stockdata'"} --> ${r'{ticker}'}"""
                actions = [('reason', self.reason)]
            
            loading_bar.dynamic_update(
                f"Action analysis complete\n"
                f"Found {len(actions)} valid actions", 
                operation="get_actions.complete"
            )
            return actions
            
        except Exception as e:
            error_msg = f"Error parsing actions: {str(e)}"
            loading_bar.dynamic_update(error_msg, operation="get_actions.error")
            logging.error(error_msg)
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

    def stockdata(self, ticker):
        loading_bar.dynamic_update(f'Getting data for {ticker}', operation="stockdata")
        """Retrieve historical stock data for analysis."""
        if ticker in self.stock_data.stock_data:
            data = self.stock_data.get_stock_data([ticker])
            loading_bar.dynamic_update("Data retrieved", operation="stockdata")
            return {ticker: data[ticker]}
        loading_bar.dynamic_update("Data retrieval failed", operation="stockdata")
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
        loading_bar.dynamic_update(f"Analyzing sentiment for {ticker}", operation="analyze_sentiment")
        
        chatgpt_sentiment = self.environment["chatgpt_insights"].get(ticker, "")
        perplexity_sentiment = self.environment["perplexity_insights"].get(ticker, "")
        
        sentiment_score = 0.5  # Neutral by default
        if "bullish" in chatgpt_sentiment:
            sentiment_score += 0.25
        if "bearish" in chatgpt_sentiment:
            sentiment_score -= 0.25
        
        loading_bar.dynamic_update(f"Completed sentiment analysis for {ticker}", operation="analyze_sentiment")
        return max(0, min(1, sentiment_score))

    def _analyze_technicals(self, data):
        loading_bar.dynamic_update("Starting technical analysis", operation="analyze_technicals")
        
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
        loading_bar.dynamic_update("Learning from recent trades", operation="learn")
        loading_bar.dynamic_update("Learning from recent trades", operation="learn")
        recent_trades = self.history[-10:]
        if recent_trades:
            profit_ratio = sum(1 for trade in recent_trades if "Sold" in trade) / len(recent_trades)
            if profit_ratio > 0.6:
                self.risk_tolerance = min(0.05, self.risk_tolerance * 1.1)
            elif profit_ratio < 0.4:
                self.risk_tolerance = max(0.01, self.risk_tolerance * 0.9)
        loading_bar.dynamic_update("Learning complete", operation="learn")

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
        
        loading_bar.dynamic_update("Report generated", operation="get_performance_report")
        return report
    
    def reason(self, query=None):
        loading_bar.dynamic_update("Starting reasoning process", operation="reason")
        
        # Add phase to history before querying
        phase_idx = self.action_history.start_phase("Reasoning", query or self.action_inputs)
        
        response = self.ollama.query_ollama(query or self.action_inputs)
        
        # Add response to history
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
        loading_bar.dynamic_update("Executing trades", operation="execute_trades")
        self.update_most_recent_action("Executing trades")
        self.act(self.decisons)
        self.spinner_animation(current_stage="ExecuteTrades")
        loading_bar.dynamic_update("Trades executed", operation="execute_trades")

    # In Agent.begin()
    def begin(self):
        with loading_bar:
            phases = [
                ("Strategic Planning", self.plan_actions),
                ("Stock Selection", self.pick_tickers),
                ("Validation", self.test_outputs),
                ("Research Analysis", self.research_and_insight),
                ("Trade Execution", self.execute_trades),
                ("Learning Phase", self.learn)
            ]
            
            for phase_name, phase_function in phases:
                loading_bar.dynamic_update(f"Starting {phase_name}", operation=f"begin.{phase_name.lower().replace(' ', '_')}")
                phase_function()
                loading_bar.dynamic_update(f"Completed {phase_name}", operation=f"begin.{phase_name.lower().replace(' ', '_')}")

        self._generate_comprehensive_report()
        self.finalize_execution()
        loading_bar.dynamic_update("Agent begun", operation="begin")



def main():
    logging.basicConfig(level=logging.INFO)

    agent = Agent()
    agent.begin()

class Portfolio:
    def __init__(self):
        self.holdings = {}
        self.pending_stocks = {}
        self.transaction_history = []

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

if __name__ == "__main__":
    main()
