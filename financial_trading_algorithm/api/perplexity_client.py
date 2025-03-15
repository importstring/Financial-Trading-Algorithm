from typing import Dict, Any, Optional, List, Tuple
from .base_client import BaseAPIClient
import logging
import json
import asyncio

class PerplexityClient(BaseAPIClient):
    """Client for interacting with Perplexity API."""
    
    def __init__(self):
        """Initialize Perplexity client."""
        super().__init__('perplexity')
        
        # Get model configuration
        self.model = self.config.get('API', 'perplexity_model')
        
        # API endpoint
        self.api_url = "https://api.perplexity.ai/chat/completions"
    
    async def query(
        self,
        query: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_role: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Query Perplexity API with retry and error handling.
        
        Args:
            query: User query
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0-1)
            system_role: Optional system role prompt
            
        Returns:
            Tuple of (response text, success status)
        """
        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": query})
        
        try:
            response = await self._handle_rate_limit(
                self._make_completion_request,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response[1]:  # If successful
                return response[0], True
            return f"Error: {response[0]}", False
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Perplexity query failed: {error_info}")
            return str(error_info), False
    
    async def _make_completion_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Tuple[str, bool]:
        """
        Make a completion request to Perplexity.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Response randomness
            
        Returns:
            Tuple of (response text, success status)
        """
        try:
            json_data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = await self._make_request(
                url=self.api_url,
                method='POST',
                json_data=json_data
            )
            
            if not response or 'choices' not in response:
                raise ValueError("Invalid response from Perplexity")
                
            return response['choices'][0]['message']['content'], True
            
        except Exception as e:
            self.logger.error(f"Perplexity completion request failed: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using Perplexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        prompt = f"""
        Analyze the market sentiment in this text. Consider technical factors, market psychology, and economic indicators:
        
        {text}
        
        Provide your analysis in this format:
        Score: <number between -1 and 1>
        Confidence: <number between 0 and 1>
        Factors: <key factors influencing sentiment>
        Outlook: <brief market outlook>
        """
        
        try:
            response, success = await self.query(
                query=prompt,
                temperature=0.3,
                system_role="You are a financial sentiment analysis expert."
            )
            
            if success:
                # Parse response
                import re
                score_match = re.search(r"Score:\s*([-\d.]+)", response)
                confidence_match = re.search(r"Confidence:\s*([\d.]+)", response)
                factors_match = re.search(r"Factors:\s*(.+?)(?=\n|$)", response)
                outlook_match = re.search(r"Outlook:\s*(.+?)(?=\n|$)", response)
                
                if all([score_match, confidence_match, factors_match, outlook_match]):
                    return {
                        "score": float(score_match.group(1)),
                        "confidence": float(confidence_match.group(1)),
                        "factors": factors_match.group(1).strip(),
                        "outlook": outlook_match.group(1).strip(),
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
    
    async def research_stock(
        self,
        ticker: str,
        research_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Perform detailed stock research using Perplexity.
        
        Args:
            ticker: Stock ticker symbol
            research_type: Type of research ('technical', 'fundamental', 'comprehensive')
            
        Returns:
            Dictionary with research results
        """
        research_prompts = {
            'technical': f"""
                Provide detailed technical analysis for {ticker}:
                1. Chart patterns and trends
                2. Support and resistance levels
                3. Technical indicators (RSI, MACD, etc.)
                4. Volume analysis
                5. Price action signals
                
                Format your response as:
                Patterns: <identified patterns>
                Levels: <key price levels>
                Indicators: <indicator readings>
                Volume: <volume analysis>
                Signals: <trading signals>
                Risk: <technical risk assessment>
            """,
            'fundamental': f"""
                Provide detailed fundamental analysis for {ticker}:
                1. Financial health metrics
                2. Growth indicators
                3. Competitive position
                4. Industry analysis
                5. Valuation metrics
                
                Format your response as:
                Metrics: <key financial metrics>
                Growth: <growth analysis>
                Competition: <competitive analysis>
                Industry: <industry overview>
                Valuation: <valuation assessment>
                Risk: <fundamental risk factors>
            """,
            'comprehensive': f"""
                Provide comprehensive analysis for {ticker}:
                1. Technical analysis
                2. Fundamental analysis
                3. Market sentiment
                4. Industry trends
                5. Risk assessment
                
                Format your response as:
                Technical: <technical analysis>
                Fundamental: <fundamental analysis>
                Sentiment: <market sentiment>
                Industry: <industry analysis>
                Risks: <risk factors>
                Recommendation: <investment recommendation>
            """
        }
        
        try:
            response, success = await self.query(
                query=research_prompts[research_type],
                temperature=0.3,
                system_role="You are an expert financial analyst."
            )
            
            if success:
                # Parse response into sections
                sections = {}
                current_section = None
                
                for line in response.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if ':' in line and line.split(':')[0].strip() in [
                        'Technical', 'Fundamental', 'Sentiment', 'Industry',
                        'Risks', 'Recommendation', 'Patterns', 'Levels',
                        'Indicators', 'Volume', 'Signals', 'Risk', 'Metrics',
                        'Growth', 'Competition', 'Valuation'
                    ]:
                        current_section = line.split(':')[0].strip()
                        sections[current_section] = line.split(':', 1)[1].strip()
                    elif current_section:
                        sections[current_section] += ' ' + line
                
                return {
                    "research": sections,
                    "type": research_type,
                    "ticker": ticker,
                    "status": True
                }
            
            return {
                "error": "Failed to complete research",
                "status": False
            }
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Stock research failed: {error_info}")
            return {"error": error_info, "status": False} 