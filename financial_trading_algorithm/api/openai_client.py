from typing import Dict, Any, Optional, List
from .base_client import BaseAPIClient
import logging
from openai import OpenAI
import asyncio

class OpenAIClient(BaseAPIClient):
    """Client for interacting with OpenAI API."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        super().__init__('openai')
        
        # Get model configuration
        self.model = self.config.get('API', 'openai_model')
        
        # Initialize OpenAI client with connection pooling
        self.client = self.connection_manager.get_openai_client()
        if not self.client:
            raise ValueError("Failed to initialize OpenAI client")
    
    async def query(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.5,
        system_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query OpenAI API with retry and error handling.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0-1)
            system_role: Optional system role prompt
            
        Returns:
            Dictionary with response and status
        """
        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._handle_rate_limit(
                self._make_completion_request,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"OpenAI query failed: {error_info}")
            return {"error": error_info, "status": False}
    
    async def _make_completion_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Make a completion request to OpenAI.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Response randomness
            
        Returns:
            API response
        """
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if not response or not response.choices:
                raise ValueError("Empty response from OpenAI")
                
            return {
                "content": response.choices[0].message.content,
                "status": True,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI completion request failed: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using OpenAI.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        prompt = f"""
        Analyze the sentiment of this text and provide a score from -1 (very negative) to 1 (very positive):
        
        Text: {text}
        
        Respond in this format:
        Score: <number>
        Confidence: <number 0-1>
        Analysis: <brief explanation>
        """
        
        try:
            response = await self.query(
                prompt=prompt,
                temperature=0.3,
                system_role="You are a financial sentiment analyzer."
            )
            
            if response.get("status", False):
                content = response["content"]
                
                # Parse response
                import re
                score_match = re.search(r"Score:\s*([-\d.]+)", content)
                confidence_match = re.search(r"Confidence:\s*([\d.]+)", content)
                analysis_match = re.search(r"Analysis:\s*(.+)", content)
                
                if score_match and confidence_match and analysis_match:
                    return {
                        "score": float(score_match.group(1)),
                        "confidence": float(confidence_match.group(1)),
                        "analysis": analysis_match.group(1).strip(),
                        "status": True
                    }
                
            return {
                "error": "Failed to parse sentiment analysis",
                "status": False
            }
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Sentiment analysis failed: {error_info}")
            return {"error": error_info, "status": False}
    
    async def analyze_financial_text(
        self,
        text: str,
        analysis_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Analyze financial text for insights.
        
        Args:
            text: Financial text to analyze
            analysis_type: Type of analysis ('technical', 'fundamental', 'comprehensive')
            
        Returns:
            Dictionary with analysis results
        """
        system_roles = {
            'technical': "You are a technical analysis expert.",
            'fundamental': "You are a fundamental analysis expert.",
            'comprehensive': "You are a comprehensive financial analyst."
        }
        
        prompts = {
            'technical': """
                Analyze this text for technical analysis insights:
                1. Identify price patterns and trends
                2. Note volume patterns
                3. Highlight technical indicators
                4. Provide trading implications
                
                Text: {text}
                
                Respond with:
                Patterns: <key patterns>
                Indicators: <relevant indicators>
                Volume Analysis: <volume insights>
                Trading Signals: <bullish/bearish signals>
                Confidence: <0-1>
            """,
            'fundamental': """
                Analyze this text for fundamental insights:
                1. Identify key financial metrics
                2. Note growth indicators
                3. Highlight competitive advantages
                4. Assess financial health
                
                Text: {text}
                
                Respond with:
                Metrics: <key metrics>
                Growth: <growth analysis>
                Competitive Position: <advantages/disadvantages>
                Financial Health: <assessment>
                Confidence: <0-1>
            """,
            'comprehensive': """
                Provide a comprehensive analysis of this text:
                1. Technical factors
                2. Fundamental factors
                3. Market sentiment
                4. Risk assessment
                
                Text: {text}
                
                Respond with:
                Technical Analysis: <key points>
                Fundamental Analysis: <key points>
                Sentiment: <market sentiment>
                Risks: <key risks>
                Recommendation: <buy/sell/hold>
                Confidence: <0-1>
            """
        }
        
        try:
            response = await self.query(
                prompt=prompts[analysis_type].format(text=text),
                temperature=0.3,
                system_role=system_roles[analysis_type]
            )
            
            if response.get("status", False):
                return {
                    "analysis": response["content"],
                    "type": analysis_type,
                    "status": True
                }
            
            return {
                "error": "Failed to analyze financial text",
                "status": False
            }
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Financial text analysis failed: {error_info}")
            return {"error": error_info, "status": False} 