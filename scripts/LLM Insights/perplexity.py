import requests
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Tuple, Dict
import logging

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_perplexity(
    query: str,
    api_key: str = "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",
    cost_tracker: Dict[str, float] = None
) -> Tuple[str, float]:
    """
    Auto-versioning Perplexity query function with dynamic model selection
    
    Args:
        query: User's question/query
        api_key: Perplexity API key (default placeholder)
        cost_tracker: Optional dict to accumulate costs
    
    Returns:
        Tuple of (response_text, cost)
    """
    
    # 1. API Setup
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Perplexity-Version": current_date,
        "Content-Type": "application/json"
    }
    
    try:
        # 2. Model Discovery
        models_response = requests.get(
            "https://api.perplexity.ai/models",
            headers=headers,
            timeout=10
        )
        models = models_response.json().get('data', [])
        
        # 3. Model Selection
        model = _select_model(query, models)
        
        # 4. Execute Query
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json={
                "model": model['id'],
                "messages": [{"role": "user", "content": query}],
                "temperature": 0.7,
                "max_tokens": 4096
            },
            timeout=15
        )
        response.raise_for_status()
        
        # 5. Cost Tracking
        cost = model.get('cost', 0)
        if cost_tracker is not None:
            cost_tracker[model['id']] = cost_tracker.get(model['id'], 0) + cost
            
        return response.json()['choices'][0]['message']['content'], cost
        
    except Exception as e:
        logging.error(f"Query failed: {str(e)}")
        return f"Error: {str(e)}", 0.0

def _select_model(query: str, available_models: list) -> dict:
    """Select best model based on query complexity and availability"""
    try:
        # Use local Ollama for complexity analysis
        ollama_resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"Classify query complexity (1-5): {query}",
                "stream": False
            },
            timeout=5
        )
        complexity = int(ollama_resp.json()['response'].strip())
    except:
        complexity = 3  # Fallback
        
    # Model selection logic
    if complexity <= 2:
        return next((m for m in available_models if "small" in m['id']), available_models[0])
    elif complexity == 3:
        return next((m for m in available_models if "medium" in m['id']), available_models[0])
    elif complexity >= 4:
        return next((m for m in available_models if "large" in m['id']), available_models[0])
