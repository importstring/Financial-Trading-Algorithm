import openai
from openai import OpenAI
from typing import Dict, Tuple, Optional
import os

def query_OpenAI(query: str, model: str = "", max_tokens: int = 4000, temperature: float = 0.5, role: str = "") -> Tuple[str, bool]:
    """Query OpenAI API with advanced features for financial analysis."""
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from os.path import join, dirname
            filepath = join(dirname(__file__), "API_Keys", "OpenAI.txt")
            with open(filepath, "r") as f:
                api_key = f.read().strip()
        except Exception as e:
            return "OpenAI API key not found in environment variables or file.", False
        if not api_key:
            return "OpenAI API key not found in environment variables or file.", False
    client = OpenAI(api_key=api_key)

    # Validate and set model
    supported_models: Dict[str, Dict[str, int]] = {
        "gpt-4-1106-preview": {"max_tokens": 128000},
        "gpt-4-0125-preview": {"max_tokens": 128000},
        "gpt-4-vision-preview": {"max_tokens": 4096},
        "gpt-4": {"max_tokens": 8192},
        "gpt-3.5-turbo-0125": {"max_tokens": 16385},
        "gpt-3.5-turbo": {"max_tokens": 4096},
    }
    
    if not model:
        model = "gpt-4-0125-preview"
    
    if model not in supported_models:
        return f"Unsupported model: {model}. Available models: {list(supported_models.keys())}", False
    
    # Set token limit
    max_tokens = min(max_tokens, supported_models[model]["max_tokens"])
    
    if not query.strip():
        return "Invalid query provided. Please check the input.", False

    # Set default role if not provided
    if not role:
        role = "You are a financial analyst with extensive experience in the stock market. You provide insights based on your current knowledge."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": query}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stream=False,
            presence_penalty=0,
            frequency_penalty=0
        )
        output = response.choices[0].message.content
        status = True
        
    except openai.BadRequestError as e:
        output, status = f"Invalid request: {e}", False
    except openai.AuthenticationError:
        output, status = "Authentication failed. Check your API key.", False
    except openai.RateLimitError:
        output, status = "Rate limit exceeded. Please try again later.", False
    except openai.APITimeoutError:
        output, status = "Request timed out. Please try again.", False
    except openai.APIConnectionError:
        output, status = "Connection error. Please check your internet connection.", False
    except Exception as e:
        output, status = f"An unexpected error occurred: {e}", False
    
    return output, status

# Example usage
if __name__ == "__main__":
    result, success = query_OpenAI("What are the key factors affecting stock market volatility?")
    if success:
        print("Response:", result)
    else:
        print("Error:", result)
