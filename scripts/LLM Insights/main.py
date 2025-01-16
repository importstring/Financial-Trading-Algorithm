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

# Get project root directory
def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    # Navigate up until we find the project root (where .git usually exists)
    for parent in [current_file, *current_file.parents]:
        if (parent / '.git').exists():
            return parent
        # Fallback: if we find 'Financial-Trading-Algorithm' directory
        if parent.name == 'Financial-Trading-Algorithm':
            return parent
    raise FileNotFoundError("Could not find project root directory")

# Set up project paths
PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / 'Data'
STOCK_DATA_PATH = DATA_PATH / 'Stock-Data'
DATA_MANAGEMENT_PATH = DATA_PATH / 'Data-Management'
API_KEYS_PATH = PROJECT_ROOT / 'API_Keys'
PORTFOLIO_PATH = DATA_PATH / 'Databases' / 'Portfolio'

# Add Data Management to system path
sys.path.append(str(DATA_MANAGEMENT_PATH))
from maintenance import update_data #type: ignore

"""
Stock Data: 
- Update with update_data() 
- Inside {path}
Financial Research:
- Fintool
Trading Insights 
- Chatgpt-4o API Key
Agent
- Lhama locally
Goal:
Given all this data we need a model that can take this and take an action in the stock market
"""

base_directory = os.path.dirname(os.path.abspath(__file__))    
data_directory = os.path.join(base_directory, 'data')

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
        
        if compound_score >= 0.05: #INPROGRESS: Change the threshold to a dynamic value - 4/10
            sentiment = "bullish"
        elif compound_score <= -0.05:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        normalized_score = compound_score * 2
        return sentiment, normalized_score

    def read_api(self):
        try:
            with open(self.api_key_path, 'r') as file:
                self.api_key = file.read().strip()
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")
    
        if not openai.api_key:
            raise ValueError("OpenAI API key is empty")
    
    def query_OpenAI(self, model="", query="", max_tokens=150, temperature=0.5, role=""):
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
        self.data_path = STOCK_DATA_PATH
        self.maintain_stock_data()
        self.stock_data = self.read_stock_data()
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
        # Standardized format for all classes
        self.tests = 1
        self.checkmark = "✅"
        self.crossmark = "❌"
        
        # For Ollama
        self.path_to_ollama = 'placeholder_directory' 
        self.ollama = self.load_ollama()
    
    def load_ollama(self):
        pass

    def query_ollama(self, query):
        pass

    def get_next_action(self, query):
        pass


class Agent:
    def __init__(self, initial_balance=100, risk_tolerance=0.02):
        self.balance = initial_balance
        
        # Inialize components
        self.chatgpt = ChatGPT4o()
        self.perplexity = Perplexity()
        self.stock_data = StockData()
        self.ollama = Ollama()
        
        # Get the Stock info from the Stock Data
        self.stock_data.read_stock_data()

        # Initalize starting values
        self.portfolio = {}
        self.history = []
        self.plan = "" 
        self.previoud_actions = []
        self.environment = None
        self.tickers = StockData.tickers
        
        self.action_inputs = None # Inputs for the current action
        self.actions = {
            "buy": self.buy(), # Buy a stock --> {ticker, shares, price} --> API
            "sell": self.sell, # Sell a stock --> {ticker, shares, price} --> API
            "hold": self.hold, # Hold the stock --> {ticker} --> Database
            "research": self.research(), # Research the stock --> {ticker} --> 'perplexity' 
            'insight': self.insight(), # Get insights on the stock --> {ticker} --> 'chatgpt'
            "reason": self.reason(), # Reiterate and start a new action --> Prompt
            "stockdata": self.get_stock_data() 
        }

        self.risk_tolerance = risk_tolerance
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

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
        

    def __research__(self):
        ticker = self.action_inputs
        pass

    def perceive_environment(self, stock_data, perplexity_insights, chatgpt_insights):
        self.environment = {
            "stock_data": stock_data,
            "perplexity_insights": perplexity_insights,
            "chatgpt_insights": chatgpt_insights,
        }
        self._update_technical_indicators()

    def _update_technical_indicators(self):
        for ticker, data in self.environment["stock_data"].items():
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'], data['Signal'] = self._calculate_macd(data['Close'])

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal

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

    def plan_actions(self):
        
        prompt = (
            f"""
            You are a financial agent that tries to make as much money as possible.
            You have {len(self.tickers)} valid tickers available to trade:
            {self.tickers}
            Here is your current portfolio:
            {self.get_portfolio()}
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
        self.plan = self.ollama.query_ollama(prompt)

    def decide_action(self):
        '''
        Phase 0: Plan out actions
        Phase 1: Get insights and research on which tickers might be interesting to trade
        Phase 2: Save
        Phase 3:  Narrow down the amount of trades using the actions
            "research": self.research(), # Research the stock --> ticker --> 'Perplexity' 
            'insight': self.insight(), # Get insights on the stock --> ticker --> 'ChatGPT'
            "reason": self.reason() # Reiterate and start a new action --> Prompt
            'stockdata': self.get_stock_data() # Get stock data for array --> array of tickers output stock data
        Phase 4: Excute
        Phase 5: Learn & Justify/Final notes
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
        except KeyError:
            logging.error(f"Invalid phase: {self.phase}")
            return None
        
    def pick_tickers(self):
        string = (
        f"""        
        You are a financial agent that tries to make as much money as possible.
        You have {len(self.tickers)} valid tickers available to trade:
        {self.tickers}
        Here is your current portfolio:
        {self.get_portfolio()}
        You can request data for tickers and they will be provided to you.
        Your goal is to determine which tickers might be interesting to potentially research or invest in by using the 
        'insight' function to get insights on the tickers and the 'research' function to get more detailed information on the tickers.
        Additionally you can reason which allows you to ask yourself a question and then your response will be provided to.
        To excute the 'insight' function:
        'insight' --> {r"${ticker}"}
        To excute the 'research' function:
        'research' --> {r"${ticker}"}
        To excute the 'reason' function:
        'reason' --> query --> {r'${YOUR QUERY HERE}'} - Fill in 'YOUR QUERY HERE' with the query you would like to ask yourself
        The first thing you should likely do is get insights on which markets are most profitable while in addition taking research from online. To do this
        'research' --> query --> {r'${YOUR QUERY HERE}'} - Fill in 'YOUR QUERY HERE' with the query you would like to ask the AI.
        'insight' --> query --> {r'${YOUR QUERY HERE}'} - Fill in 'YOUR QUERY HERE' with the query you would like to ask the AI.
        Then you should move to reasoning to narrow down which tickers to trade.
        'reason' --> query --> {r'${YOUR QUERY HERE}'} - Fill in 'YOUR QUERY HERE' with the query you would like to ask yourself
        You may want stock data so then you can use the 'stockdata' function to get the stock data for the tickers you have selected.
        'stockdata' --> {r"${ticker}"}
        You made a plan in a previous phase to act as a general guide for your actions but since new data will arrive it shouldn't be followed exactly.
        Here it is:
        {self.plan}
        """
        )
        output = self.ollama.query_ollama(string)
        actions = self.get_actions(output)
        
        results = []
        for action, param in actions:
            result = action(param)
            results.append(result)
    
        # Process the results as needed
        return results
    
    def get_actions(self, output):
        
        actions_mapping = {
            'insight': self.insight,
            'research': self.research,
            'reason': self.reason,
            'stockdata': self.stockdata
        }

        actions = []
        for line in output.splitlines():
            for key, action in actions_mapping.items():
                if key in line.lower():
                    # Extract the parameter (e.g., ticker or query) from the line
                    param_start = line.find("$") + 1
                    param_end = line.find("}") if "}" in line else len(line)
                    param = line[param_start:param_end].strip()
                    actions.append((action, param))

        return actions

    def insight(self, param):
        """Get market insights for a specific ticker using ChatGPT."""
        query = f"Provide market insights and sentiment analysis for {param}. Focus on recent trends, news, and market sentiment."
        response, status = self.chatgpt.query_OpenAI(query=query)
        if status:
            return self.chatgpt.evaluate_response(response)
        return {"error": "Failed to get insights", "sentiment": "neutral", "sentiment_score": 0.0}

    def research(self, param):
        """Get detailed research analysis for a specific ticker using Perplexity."""
        query = f"Provide detailed research analysis for {param}. Include financial metrics, competitive analysis, and growth prospects."
        response, status = self.perplexity.query_perplexity(query=query)
        if status:
            return self.perplexity.evaluate_response(response)
        return {"error": "Failed to get research", "sentiment": "neutral", "sentiment_score": 0.0}

    def reason(self, query):
        """Process a reasoning query about market conditions and strategy."""
        prompt = f"Based on the following query, provide strategic analysis: {query}"
        response = self.ollama.query_ollama(prompt)
        return {"reasoning": response}

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

def main():
    logging.basicConfig(level=logging.INFO)

    agent = Agent()
    agent.begin_trading()





    # Create logic for perplexity and Chatgpt Queries
    chatgpt_query = "Any insights on the recent stock market trends? Focus on specific stocks you are bullish in based on your recent intereactions."
    perplexity_query = "What are the recent trends in the stock market that I could use to make profitable long term investments? Use data to back up your answers."

    # Decode a stock list to investigate
    pass

    # TODO: Create logic to get the agent to pick the next action
    # TODO: Loop until a decision is made
    """
    Agent Options:
    - Research
        - ChatGPT [Market Research on emotions of traders] (Sediment Analysis)
            - Prompts
        - Perplexity [Market Research on trends and live data] (Sediment Analysis)
            - Prompts
    - Buy [To Early for now]
    - Sell [To Early for now]
    - 
    """


#    # Example workflow
#    stock_info = stock_data.read_stock_data()
#    chatgpt_query = "Analyze recent trends in tech stocks"
#    perplexity_query = "Get financial reports for AAPL, GOOGL, MSFT"

#    chatgpt_response, status = chatgpt.query_OpenAI(
        #query='What stocks should I buy do you think?'
    #)
    #perplexity_response, status = perplexity.query_perplexity(
        #query='What stock is profitable to buy?'
#    )

 #   chatgpt_response, status = chatgpt.query_OpenAI(chatgpt_query)
  #  perplexity_response, status = perplexity.query_perplexity(
   #     query=perplexity_query,
    #    max_tokens=200,
     #   temperature=0.7
    #)

    #chatgpt_insights = chatgpt.evaluate_response(chatgpt_response)
    #perplexity_insights = perplexity.evaluate_response(perplexity_response)

    #agent.perceive_environment(stock_info, perplexity_insights, chatgpt_insights)
    #decisions = agent.decide_action()
#    agent.act(decisions)
 #   agent.learn()
  #  print(agent.get_performance_report())

if __name__ == "__main__":
    main()