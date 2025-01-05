import openai
import os
import sys
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import glob
import requests

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
        self.OpenAI_tests = 1
        self.checkmark = "✅"
        self.crossmark = "❌"
        self.api_key_path = os.path.join(base_directory, 'API_Keys/OpenAI.txt')
        self.default_model = "ChatGPT-4o"
        self.default_role = (
            "You are a financial analyst who has talked with millions of people about the stock market. You are here to provide insights about the stock market based on your interactions."
            )
        self.api_key = self.read_api()
        
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
                print(f"Test {self.OpenAI_tests}: {self.checkmark if test[0] else self.crossmark} {test[1]}")
                self.OpenAI_tests += 1
            
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
        Evaluate response from ChatGPT-4o API and return insights.
        """

        try:
            if not response:
                return "No response received."
            if "error" in response.lower():
                return "The response contains an error."
            return f"Insights: {response}"
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            return "Error during evaluation."
        


class Fintool:
    def __init__(self):
        self.api_key_path = os.path.join(base_directory, 'API_Keys/Fintool.txt')
        self.api_key = self.read_api()
        
    def read_api(self):
        try:
            with open(self.api_key_path, 'r') as file:
                self.api_key = file.read().strip()
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {self.api_key_path}")
    
        if not self.api_key:
            raise ValueError("Fintool API key is empty")
    
    def query_fintool(self, user_query):
        """
        Send query to Fintool API and return response.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"https://api.fintool.com/query?query={user_query}", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error querying Fintool API: {e}")
            return None

    def evaluate_response(self, response):
        """
        Evaluate response from Fintool API and return insights.
        """
        try:
            if not response:
                return "No response received."
            if "error" in response:
                return f"Error in response: {response['error']}"
            return f"Insights: {response.get('data', 'No data available')}"
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            return "Error during evaluation."

class StockData:
    def __init__(self):
        self.data_path = "/Users/simon/Financial-Trading-Algorithm/Data/Stock-Data"
        self.maintain_stock_data()
        self.stock_data = self.read_stock_data()

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
            # Get all CSV files in the data path
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            
            for file_path in csv_files:
                # Extract ticker from filename
                ticker = os.path.basename(file_path).replace('.csv', '')
                
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
            sys.path.append('/Users/simon/Financial-Trading-Algorithm/Data/Data-Management')
            from main import update_data

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
        
class Agent:
    """
    Agent should be powered by llama
    """
    def __init__(self):
        self.llm = self.initialize_llama()

    def initialize_llama(self):
        # Initialize and return the Llama model
        # This is a placeholder and should be replaced with actual Llama initialization
        pass

    def process_data(self, stock_data, fintool_insights, chatgpt_insights):
        # Process the data using the Llama model
        # This is a placeholder and should be replaced with actual processing logic
        pass

    def make_decision(self):
        # Make a decision based on the processed data
        # This is a placeholder and should be replaced with actual decision-making logic
        pass

def main():
    logging.basicConfig(level=logging.INFO)

    stock_data = StockData()
    chatgpt = ChatGPT4o()
    fintool = Fintool()
    agent = Agent()

    # Example workflow
    stock_info = stock_data.read_stock_data()
    chatgpt_query = "Analyze recent trends in tech stocks"
    fintool_query = "Get financial reports for AAPL, GOOGL, MSFT"

    chatgpt_response = chatgpt.query_chatgpt(chatgpt_query)
    fintool_response = fintool.query_fintool(fintool_query)

    chatgpt_insights = chatgpt.evaluate_response(chatgpt_response)
    fintool_insights = fintool.evaluate_response(fintool_response)

    agent.process_data(stock_info, fintool_insights, chatgpt_insights)
    decision = agent.make_decision()

    logging.info(f"Agent's decision: {decision}")

if __name__ == "__main__":
    main()